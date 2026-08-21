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
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import StrictBoundaryModel
from grit.selection import CheckpointIdentity
from grit.waterbirds_selection import (
    FrozenWaterbirdsCandidate,
    FrozenWaterbirdsCheckpoint,
    WaterbirdsFinalTestMetricRecord,
    WaterbirdsValidationMetricRecord,
)
from grit.waterbirds_training import WaterbirdsRestorationReceipt

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class WaterbirdsCandidateConfig(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-candidate/v2"]
    protocol_id: Literal["waterbirds_cf/v1"]
    non_reportable: Literal[True]
    method_id: Literal["erm", "grit"]
    dataset_profile: Literal["fixture", "production"]
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr | None
    projection_diagnostics_digest: NonEmptyStr | None
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
                self.projection_diagnostics_digest,
                self.projection_rank,
                self.relative_singular_value_tolerance,
            )
        ):
            raise ValueError("Waterbirds ERM config cannot contain oracle projection")
        if self.method_id == "grit" and any(
            value is None
            for value in (
                self.pair_manifest_digest,
                self.projection_diagnostics_digest,
                self.projection_rank,
                self.relative_singular_value_tolerance,
            )
        ):
            raise ValueError("Waterbirds GRIT config requires oracle projection")
        return self

    def scientific_config_digest(self) -> str:
        return self.canonical_digest()


class _WaterbirdsArtifactReference(StrictBoundaryModel):
    artifact_id: NonEmptyStr
    relative_uri: NonEmptyStr
    digest: NonEmptyStr


class WaterbirdsDatasetArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["dataset_manifest"]


class WaterbirdsFeatureArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["feature_manifest"]
    dataset_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]


class WaterbirdsCheckpointArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["selected_linear_checkpoint"]
    checkpoint: CheckpointIdentity


class WaterbirdsPairArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["pair_manifest"]
    dataset_manifest_digest: NonEmptyStr


class WaterbirdsProjectionArtifactReference(_WaterbirdsArtifactReference):
    kind: Literal["projection_diagnostics"]
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    requested_rank: Annotated[StrictInt, Field(ge=0, le=24)]


WaterbirdsArtifactReference = Annotated[
    WaterbirdsDatasetArtifactReference
    | WaterbirdsFeatureArtifactReference
    | WaterbirdsCheckpointArtifactReference
    | WaterbirdsPairArtifactReference
    | WaterbirdsProjectionArtifactReference,
    Field(discriminator="kind"),
]


class WaterbirdsRunResult(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-run-result/v2"]
    result_kind: Literal["ordinary_waterbirds"]
    status: Literal["succeeded"]
    run_id: NonEmptyStr
    final_seed: StrictInt
    resolved_config: WaterbirdsCandidateConfig
    resolved_config_digest: NonEmptyStr
    code: CodeProvenance
    environment: EnvironmentProvenance
    validation_metrics: tuple[WaterbirdsValidationMetricRecord, ...]
    candidate_selection: FrozenWaterbirdsCandidate
    checkpoint_selection: FrozenWaterbirdsCheckpoint
    restoration: WaterbirdsRestorationReceipt
    final_test_metric: WaterbirdsFinalTestMetricRecord
    selected_checkpoint_manifest_digest: NonEmptyStr
    artifacts: tuple[WaterbirdsArtifactReference, ...]

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
            or candidate.dataset_manifest_digest != config.dataset_manifest_digest
            or candidate.feature_cache_manifest_digest
            != config.feature_cache_manifest_digest
            or candidate.normalization != config.normalization
            or candidate.adjusted_weight_spec_digest
            != config.adjusted_weight_spec_digest
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
            or checkpoint.decision.seed != self.final_seed
            or checkpoint.decision.projection_rank != config.projection_rank
            or checkpoint.dataset_manifest_digest != config.dataset_manifest_digest
            or checkpoint.feature_cache_manifest_digest
            != config.feature_cache_manifest_digest
            or checkpoint.normalization != config.normalization
            or checkpoint.adjusted_weight_spec_digest
            != config.adjusted_weight_spec_digest
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
                metric.normalization,
                metric.adjusted_weight_spec_digest,
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
                config.normalization,
                config.adjusted_weight_spec_digest,
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
            final.normalization,
            final.adjusted_weight_spec_digest,
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
            config.normalization,
            config.adjusted_weight_spec_digest,
        )
        if final_identity != expected_final:
            raise ValueError("Waterbirds final metric identity is inconsistent")
        self._validate_artifacts()
        return self

    def _validate_artifacts(self) -> None:
        config = self.resolved_config
        artifacts_by_kind = {artifact.kind: artifact for artifact in self.artifacts}
        if len(artifacts_by_kind) != len(self.artifacts):
            raise ValueError("Waterbirds result artifact kinds must be unique")
        if len({artifact.artifact_id for artifact in self.artifacts}) != len(
            self.artifacts
        ) or len({artifact.relative_uri for artifact in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("Waterbirds result artifact references must be unique")
        required = {
            "dataset_manifest",
            "feature_manifest",
            "selected_linear_checkpoint",
        }
        if config.method_id == "grit":
            required |= {"pair_manifest", "projection_diagnostics"}
        if set(artifacts_by_kind) != required:
            raise ValueError(
                "Waterbirds result required artifact references are missing"
            )
        dataset = artifacts_by_kind["dataset_manifest"]
        feature = artifacts_by_kind["feature_manifest"]
        checkpoint = artifacts_by_kind["selected_linear_checkpoint"]
        if (
            dataset.digest != config.dataset_manifest_digest
            or not isinstance(feature, WaterbirdsFeatureArtifactReference)
            or feature.digest != config.feature_cache_manifest_digest
            or feature.dataset_manifest_digest != config.dataset_manifest_digest
            or feature.normalization != config.normalization
            or not isinstance(checkpoint, WaterbirdsCheckpointArtifactReference)
            or checkpoint.digest != self.selected_checkpoint_manifest_digest
            or checkpoint.checkpoint != self.checkpoint_selection.checkpoint
        ):
            raise ValueError("Waterbirds result artifact lineage is inconsistent")
        if config.method_id == "grit":
            pair = artifacts_by_kind["pair_manifest"]
            projection = artifacts_by_kind["projection_diagnostics"]
            if (
                not isinstance(pair, WaterbirdsPairArtifactReference)
                or pair.digest != config.pair_manifest_digest
                or pair.dataset_manifest_digest != config.dataset_manifest_digest
                or not isinstance(projection, WaterbirdsProjectionArtifactReference)
                or projection.digest != config.projection_diagnostics_digest
                or projection.pair_manifest_digest != config.pair_manifest_digest
                or projection.feature_cache_manifest_digest
                != config.feature_cache_manifest_digest
                or projection.normalization != config.normalization
                or projection.requested_rank != config.projection_rank
            ):
                raise ValueError("Waterbirds GRIT artifact lineage is inconsistent")


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


class WaterbirdsFinalSeedObservation(StrictBoundaryModel):
    seed: StrictInt
    method_id: Literal["erm", "grit"]
    result_path: NonEmptyStr
    metric_record_id: NonEmptyStr
    worst_group_accuracy: FiniteFloat
    adjusted_average_accuracy: FiniteFloat
    raw_average_accuracy: FiniteFloat


class WaterbirdsPairedSeedDifference(StrictBoundaryModel):
    seed: StrictInt
    grit_minus_erm_worst_group_accuracy: FiniteFloat


class WaterbirdsMethodSmokeSummary(StrictBoundaryModel):
    method_id: Literal["erm", "grit"]
    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    final_observations: Annotated[
        tuple[WaterbirdsFinalSeedObservation, ...], Field(min_length=10, max_length=10)
    ]
    worst_group_summary: WaterbirdsMetricSummary
    adjusted_average_summary: WaterbirdsMetricSummary
    raw_average_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_summary(self) -> WaterbirdsMethodSmokeSummary:
        if len(set(self.configured_final_seeds)) != 10:
            raise ValueError("Waterbirds configured final seeds must be unique")
        observed_seeds = tuple(item.seed for item in self.final_observations)
        if observed_seeds != self.configured_final_seeds:
            raise ValueError(
                "Waterbirds final observations require configured seeds exactly once"
            )
        if any(item.method_id != self.method_id for item in self.final_observations):
            raise ValueError("Waterbirds final observation method is inconsistent")
        if len({item.result_path for item in self.final_observations}) != 10 or len(
            {item.metric_record_id for item in self.final_observations}
        ) != 10:
            raise ValueError(
                "Waterbirds final observations must be uniquely attributable"
            )
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
        for metric_name, values, summary in series:
            if len(values) != 10:
                raise ValueError("Waterbirds smoke metric series requires ten seeds")
            if summary != make_waterbirds_metric_summary(metric_name, values):
                raise ValueError("Waterbirds smoke metric summary is inconsistent")
        return self

    @property
    def result_paths(self) -> tuple[str, ...]:
        return tuple(item.result_path for item in self.final_observations)

    @property
    def final_worst_group_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.worst_group_accuracy) for item in self.final_observations
        )

    @property
    def final_adjusted_average_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.adjusted_average_accuracy) for item in self.final_observations
        )

    @property
    def final_raw_average_accuracies(self) -> tuple[float, ...]:
        return tuple(
            float(item.raw_average_accuracy) for item in self.final_observations
        )


class WaterbirdsSmokeSummary(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-smoke-result/v2"]
    non_reportable: Literal[True]
    dataset_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    normalization: Literal["none", "l2"]
    adjusted_weight_spec_digest: NonEmptyStr
    methods: tuple[WaterbirdsMethodSmokeSummary, WaterbirdsMethodSmokeSummary]
    paired_worst_group_differences: tuple[WaterbirdsPairedSeedDifference, ...]
    paired_worst_group_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_paired_summary(self) -> WaterbirdsSmokeSummary:
        if tuple(item.method_id for item in self.methods) != ("erm", "grit"):
            raise ValueError("Waterbirds smoke methods must be ordered ERM then GRIT")
        for method in self.methods:
            if (
                method.dataset_manifest_digest != self.dataset_manifest_digest
                or method.feature_cache_manifest_digest
                != self.feature_cache_manifest_digest
                or method.normalization != self.normalization
                or method.adjusted_weight_spec_digest
                != self.adjusted_weight_spec_digest
            ):
                raise ValueError("Waterbirds smoke method lineage is inconsistent")
        if self.methods[0].configured_final_seeds != (
            self.methods[1].configured_final_seeds
        ):
            raise ValueError("Waterbirds paired methods require identical final seeds")
        erm_by_seed = {
            item.seed: float(item.worst_group_accuracy)
            for item in self.methods[0].final_observations
        }
        grit_by_seed = {
            item.seed: float(item.worst_group_accuracy)
            for item in self.methods[1].final_observations
        }
        expected = tuple(
            WaterbirdsPairedSeedDifference(
                seed=seed,
                grit_minus_erm_worst_group_accuracy=(
                    grit_by_seed[seed] - erm_by_seed[seed]
                ),
            )
            for seed in self.methods[0].configured_final_seeds
        )
        if self.paired_worst_group_differences != expected:
            raise ValueError("Waterbirds paired differences are inconsistent")
        if self.paired_worst_group_summary != make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(
                float(item.grit_minus_erm_worst_group_accuracy) for item in expected
            ),
        ):
            raise ValueError("Waterbirds paired summary is inconsistent")
        return self
