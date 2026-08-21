"""Strict resolved-config and canonical result boundaries for Waterbirds smoke runs."""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    FiniteFloat,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from grit.config import LinearProbeTrainingConfig, SeedSets
from grit.results import ArtifactReference, CodeProvenance, EnvironmentProvenance
from grit.schemas import StrictBoundaryModel
from grit.waterbirds_selection import (
    FrozenWaterbirdsCandidate,
    FrozenWaterbirdsCheckpoint,
    WaterbirdsFinalTestMetricRecord,
    WaterbirdsValidationMetricRecord,
)
from grit.waterbirds_training import WaterbirdsRestorationReceipt

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class WaterbirdsCandidateConfig(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-candidate/v1"]
    protocol_id: Literal["waterbirds_cf/v1"]
    non_reportable: Literal[True]
    method_id: Literal["erm", "grit"]
    dataset_profile: Literal["fixture", "production"]
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    pair_manifest_digest: NonEmptyStr | None
    projection_rank: Annotated[StrictInt, Field(ge=0, le=24)] | None
    relative_singular_value_tolerance: Annotated[StrictFloat, Field(gt=0.0)] | None
    training: LinearProbeTrainingConfig
    seed_sets: SeedSets

    @model_validator(mode="after")
    def _validate_method(self) -> WaterbirdsCandidateConfig:
        if self.method_id == "erm" and any(
            value is not None
            for value in (
                self.pair_manifest_digest,
                self.projection_rank,
                self.relative_singular_value_tolerance,
            )
        ):
            raise ValueError("Waterbirds ERM config cannot contain oracle projection")
        if self.method_id == "grit" and any(
            value is None
            for value in (
                self.pair_manifest_digest,
                self.projection_rank,
                self.relative_singular_value_tolerance,
            )
        ):
            raise ValueError("Waterbirds GRIT config requires oracle projection")
        return self

    def scientific_config_digest(self) -> str:
        return self.canonical_digest()


class WaterbirdsRunResult(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-run-result/v1"]
    result_kind: Literal["ordinary_waterbirds"]
    status: Literal["succeeded"]
    run_id: NonEmptyStr
    resolved_config: WaterbirdsCandidateConfig
    resolved_config_digest: NonEmptyStr
    code: CodeProvenance
    environment: EnvironmentProvenance
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    candidate_selection: FrozenWaterbirdsCandidate
    checkpoint_selection: FrozenWaterbirdsCheckpoint
    restoration: WaterbirdsRestorationReceipt
    final_test_metric: WaterbirdsFinalTestMetricRecord
    artifacts: tuple[ArtifactReference, ...]

    @model_validator(mode="after")
    def _validate_result(self) -> WaterbirdsRunResult:
        config = self.resolved_config
        digest = config.scientific_config_digest()
        if self.resolved_config_digest != config.canonical_digest():
            raise ValueError("Waterbirds result config digest is inconsistent")
        candidate = self.candidate_selection
        if (
            candidate.method_id != config.method_id
            or candidate.scientific_config_digest != digest
            or candidate.projection_rank != config.projection_rank
            or candidate.seed_sets != config.seed_sets
        ):
            raise ValueError("Waterbirds frozen candidate does not match result config")
        checkpoint = self.checkpoint_selection
        if (
            checkpoint.candidate_selection_id != candidate.frozen_selection_id
            or checkpoint.method_id != candidate.method_id
            or checkpoint.checkpoint.run_id != self.run_id
            or checkpoint.checkpoint.candidate_id != candidate.candidate_id
            or checkpoint.checkpoint.scientific_config_digest != digest
            or checkpoint.decision.seed not in config.seed_sets.final
            or checkpoint.decision.projection_rank != config.projection_rank
        ):
            raise ValueError("Waterbirds checkpoint does not match result lifecycle")
        if (
            self.restoration.candidate_selection_id != candidate.frozen_selection_id
            or self.restoration.checkpoint != checkpoint.checkpoint
        ):
            raise ValueError("Waterbirds restoration does not match result checkpoint")
        if not self.validation_metrics:
            raise ValueError("Waterbirds result requires final-run validation history")
        validation_by_id = {item.record_id: item for item in self.validation_metrics}
        if len(validation_by_id) != len(self.validation_metrics):
            raise ValueError("Waterbirds validation record IDs must be unique")
        for metric in self.validation_metrics:
            identity = (
                metric.run_id,
                metric.candidate_id,
                metric.method_id,
                metric.scientific_config_digest,
                metric.seed_stage,
                metric.seed,
                metric.projection_rank,
                metric.dataset_manifest_digest,
                metric.feature_cache_manifest_digest,
            )
            expected = (
                self.run_id,
                candidate.candidate_id,
                config.method_id,
                digest,
                checkpoint.decision.seed_stage,
                checkpoint.decision.seed,
                config.projection_rank,
                config.dataset_manifest_digest,
                config.feature_cache_manifest_digest,
            )
            if identity != expected:
                raise ValueError(
                    "Waterbirds validation history identity is inconsistent"
                )
        contributor = validation_by_id.get(checkpoint.decision.contributing_record_id)
        if contributor is None:
            raise ValueError("Waterbirds checkpoint cites an unavailable metric")
        if (
            contributor.checkpoint_id != checkpoint.checkpoint.checkpoint_id
            or contributor.epoch != checkpoint.checkpoint.epoch
            or contributor.worst_group_accuracy
            != checkpoint.decision.worst_group_accuracy
            or contributor.adjusted_average_accuracy
            != checkpoint.decision.adjusted_average_accuracy
        ):
            raise ValueError("Waterbirds checkpoint decision trace is inconsistent")
        final = self.final_test_metric
        final_identity = (
            final.run_id,
            final.candidate_id,
            final.method_id,
            final.scientific_config_digest,
            final.checkpoint_id,
            final.epoch,
            final.seed,
            final.projection_rank,
            final.dataset_manifest_digest,
            final.feature_cache_manifest_digest,
        )
        expected_final = (
            self.run_id,
            candidate.candidate_id,
            config.method_id,
            digest,
            checkpoint.checkpoint.checkpoint_id,
            checkpoint.checkpoint.epoch,
            checkpoint.decision.seed,
            config.projection_rank,
            config.dataset_manifest_digest,
            config.feature_cache_manifest_digest,
        )
        if final_identity != expected_final:
            raise ValueError("Waterbirds final metric identity is inconsistent")
        return self


MetricName: TypeAlias = Literal[
    "worst_group_accuracy",
    "adjusted_average_accuracy",
    "raw_average_accuracy",
    "grit_minus_erm_worst_group_accuracy",
]


class WaterbirdsMetricSummary(StrictBoundaryModel):
    metric_name: MetricName
    seed_count: Literal[10]
    mean: FiniteFloat
    sample_standard_deviation: Annotated[FiniteFloat, Field(ge=0.0)]
    ci95_lower: FiniteFloat
    ci95_upper: FiniteFloat


def make_waterbirds_metric_summary(
    metric_name: MetricName,
    values: tuple[float, ...],
) -> WaterbirdsMetricSummary:
    if len(values) != 10:
        raise ValueError("Waterbirds final metric summary requires ten seeds")
    mean = fmean(values)
    standard_deviation = stdev(values)
    # Two-sided 95% Student-t critical value with nine degrees of freedom.
    half_width = 2.2621571627409915 * standard_deviation / math.sqrt(10)
    return WaterbirdsMetricSummary(
        metric_name=metric_name,
        seed_count=10,
        mean=mean,
        sample_standard_deviation=standard_deviation,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
    )


class WaterbirdsMethodSmokeSummary(StrictBoundaryModel):
    method_id: Literal["erm", "grit"]
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    result_paths: tuple[NonEmptyStr, ...]
    final_worst_group_accuracies: tuple[float, ...]
    final_adjusted_average_accuracies: tuple[float, ...]
    final_raw_average_accuracies: tuple[float, ...]
    worst_group_summary: WaterbirdsMetricSummary
    adjusted_average_summary: WaterbirdsMetricSummary
    raw_average_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_summary(self) -> WaterbirdsMethodSmokeSummary:
        series: tuple[
            tuple[MetricName, tuple[float, ...], WaterbirdsMetricSummary], ...
        ] = (
            (
                "worst_group_accuracy",
                self.final_worst_group_accuracies,
                self.worst_group_summary,
            ),
            (
                "adjusted_average_accuracy",
                self.final_adjusted_average_accuracies,
                self.adjusted_average_summary,
            ),
            (
                "raw_average_accuracy",
                self.final_raw_average_accuracies,
                self.raw_average_summary,
            ),
        )
        if len(self.result_paths) != 10:
            raise ValueError("Waterbirds smoke summary requires ten final results")
        for metric_name, values, summary in series:
            if len(values) != 10:
                raise ValueError("Waterbirds smoke metric series requires ten seeds")
            if summary != make_waterbirds_metric_summary(metric_name, values):
                raise ValueError("Waterbirds smoke metric summary is inconsistent")
        return self


class WaterbirdsSmokeSummary(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-smoke-result/v1"]
    non_reportable: Literal[True]
    dataset_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    methods: tuple[WaterbirdsMethodSmokeSummary, WaterbirdsMethodSmokeSummary]
    paired_worst_group_differences: tuple[float, ...]
    paired_worst_group_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_paired_summary(self) -> WaterbirdsSmokeSummary:
        if tuple(item.method_id for item in self.methods) != ("erm", "grit"):
            raise ValueError("Waterbirds smoke methods must be ordered ERM then GRIT")
        expected = tuple(
            grit - erm
            for erm, grit in zip(
                self.methods[0].final_worst_group_accuracies,
                self.methods[1].final_worst_group_accuracies,
                strict=True,
            )
        )
        if self.paired_worst_group_differences != expected:
            raise ValueError("Waterbirds paired differences are inconsistent")
        if self.paired_worst_group_summary != make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy", expected
        ):
            raise ValueError("Waterbirds paired summary is inconsistent")
        return self
