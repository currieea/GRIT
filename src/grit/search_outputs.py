"""Dataset-specific production summaries and top-level experiment index."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, FiniteFloat, StrictInt, StrictStr, model_validator

from grit.schemas import CmnistSelector, StrictBoundaryModel
from grit.search import SearchLineage
from grit.waterbirds_run_contracts import MetricName, WaterbirdsMetricSummary

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class CmnistAccuracySummary(StrictBoundaryModel):
    metric_name: Literal["test_ood_accuracy", "grit_minus_erm_test_ood_accuracy"]
    seed_count: Literal[10]
    mean: FiniteFloat
    sample_standard_deviation: Annotated[FiniteFloat, Field(ge=0.0)]
    ci95_lower: FiniteFloat
    ci95_upper: FiniteFloat


def make_cmnist_accuracy_summary(
    metric_name: Literal[
        "test_ood_accuracy", "grit_minus_erm_test_ood_accuracy"
    ],
    values: tuple[float, ...],
) -> CmnistAccuracySummary:
    if len(values) != 10:
        raise ValueError("CMNIST final metric summary requires ten seeds")
    mean = fmean(values)
    deviation = stdev(values)
    half_width = 2.2621571627409915 * deviation / math.sqrt(10)
    return CmnistAccuracySummary(
        metric_name=metric_name,
        seed_count=10,
        mean=mean,
        sample_standard_deviation=deviation,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
    )


class CmnistFinalSeedObservation(StrictBoundaryModel):
    seed: StrictInt
    method_id: Literal["erm", "grit"]
    selector: CmnistSelector
    result_path: NonEmptyStr
    metric_record_id: NonEmptyStr
    test_ood_accuracy: FiniteFloat


class CmnistMethodSelectorSummary(StrictBoundaryModel):
    method_id: Literal["erm", "grit"]
    selector: CmnistSelector
    lineage: SearchLineage
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    final_observations: Annotated[
        tuple[CmnistFinalSeedObservation, ...], Field(min_length=10, max_length=10)
    ]
    accuracy_summary: CmnistAccuracySummary

    @model_validator(mode="after")
    def _validate_summary(self) -> CmnistMethodSelectorSummary:
        if (
            len(set(self.finalist_candidate_ids)) != 3
            or self.selected_candidate_id not in self.finalist_candidate_ids
        ):
            raise ValueError("CMNIST selected candidate must be a unique finalist")
        if len(set(self.configured_final_seeds)) != 10:
            raise ValueError("CMNIST summary final seeds must be unique")
        if tuple(item.seed for item in self.final_observations) != (
            self.configured_final_seeds
        ):
            raise ValueError("CMNIST summary requires configured seeds exactly once")
        if any(
            item.method_id != self.method_id or item.selector is not self.selector
            for item in self.final_observations
        ):
            raise ValueError("CMNIST final observation identity is inconsistent")
        if len({item.result_path for item in self.final_observations}) != 10 or len(
            {item.metric_record_id for item in self.final_observations}
        ) != 10:
            raise ValueError("CMNIST final observations must be uniquely attributable")
        values = tuple(
            float(item.test_ood_accuracy) for item in self.final_observations
        )
        if self.accuracy_summary != make_cmnist_accuracy_summary(
            "test_ood_accuracy", values
        ):
            raise ValueError("CMNIST accuracy summary is inconsistent")
        return self


class CmnistPairedSeedDifference(StrictBoundaryModel):
    seed: StrictInt
    selector: CmnistSelector
    grit_minus_erm_test_ood_accuracy: FiniteFloat


class CmnistPairedSelectorSummary(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-paired-summary/v1"] = (
        "grit.cmnist-paired-summary/v1"
    )
    selector: CmnistSelector
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    paired_differences: Annotated[
        tuple[CmnistPairedSeedDifference, ...], Field(min_length=10, max_length=10)
    ]
    difference_summary: CmnistAccuracySummary

    @model_validator(mode="after")
    def _validate_pairs(self) -> CmnistPairedSelectorSummary:
        if tuple(item.seed for item in self.paired_differences) != (
            self.configured_final_seeds
        ) or any(
            item.selector is not self.selector for item in self.paired_differences
        ):
            raise ValueError("CMNIST paired differences must align by configured seed")
        values = tuple(
            float(item.grit_minus_erm_test_ood_accuracy)
            for item in self.paired_differences
        )
        if self.difference_summary != make_cmnist_accuracy_summary(
            "grit_minus_erm_test_ood_accuracy", values
        ):
            raise ValueError("CMNIST paired summary is inconsistent")
        return self


class CmnistProductionSummary(StrictBoundaryModel):
    schema_version: Literal["grit.cmnist-production-summary/v1"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    lineage: SearchLineage
    methods: tuple[
        CmnistMethodSelectorSummary,
        CmnistMethodSelectorSummary,
        CmnistMethodSelectorSummary,
        CmnistMethodSelectorSummary,
    ]
    paired_selectors: tuple[
        CmnistPairedSelectorSummary,
        CmnistPairedSelectorSummary,
    ]

    @model_validator(mode="after")
    def _validate_methods(self) -> CmnistProductionSummary:
        expected = (
            ("erm", CmnistSelector.PRIMARY_ROBUST),
            ("erm", CmnistSelector.SECONDARY_SOURCE),
            ("grit", CmnistSelector.PRIMARY_ROBUST),
            ("grit", CmnistSelector.SECONDARY_SOURCE),
        )
        if tuple((item.method_id, item.selector) for item in self.methods) != expected:
            raise ValueError("CMNIST production summaries require canonical ordering")
        if any(item.lineage != self.lineage for item in self.methods):
            raise ValueError("CMNIST production summary lineage is inconsistent")
        if tuple(item.selector for item in self.paired_selectors) != (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            raise ValueError("CMNIST paired selector summaries are misordered")
        by_identity = {
            (item.method_id, item.selector): item for item in self.methods
        }
        for paired in self.paired_selectors:
            erm = by_identity[("erm", paired.selector)]
            grit = by_identity[("grit", paired.selector)]
            if erm.configured_final_seeds != grit.configured_final_seeds:
                raise ValueError("CMNIST paired methods require identical final seeds")
            erm_by_seed = {
                item.seed: float(item.test_ood_accuracy)
                for item in erm.final_observations
            }
            grit_by_seed = {
                item.seed: float(item.test_ood_accuracy)
                for item in grit.final_observations
            }
            expected_pairs = tuple(
                CmnistPairedSeedDifference(
                    seed=seed,
                    selector=paired.selector,
                    grit_minus_erm_test_ood_accuracy=(
                        grit_by_seed[seed] - erm_by_seed[seed]
                    ),
                )
                for seed in erm.configured_final_seeds
            )
            if paired.paired_differences != expected_pairs:
                raise ValueError("CMNIST paired differences are inconsistent")
        return self


class WaterbirdsProductionMethodSummary(StrictBoundaryModel):
    method_id: Literal["erm", "grit"]
    lineage: SearchLineage
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    result_paths_by_seed: tuple[tuple[StrictInt, NonEmptyStr], ...]
    worst_group_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...]
    adjusted_average_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...]
    raw_average_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...]
    worst_group_summary: WaterbirdsMetricSummary
    adjusted_average_summary: WaterbirdsMetricSummary
    raw_average_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_summary(self) -> WaterbirdsProductionMethodSummary:
        if (
            len(set(self.finalist_candidate_ids)) != 3
            or self.selected_candidate_id not in self.finalist_candidate_ids
        ):
            raise ValueError("Waterbirds selected candidate must be a unique finalist")
        if len(set(self.configured_final_seeds)) != 10:
            raise ValueError("Waterbirds production final seeds must be unique")
        series = (
            self.result_paths_by_seed,
            self.worst_group_by_seed,
            self.adjusted_average_by_seed,
            self.raw_average_by_seed,
        )
        if any(
            tuple(seed for seed, _ in values) != self.configured_final_seeds
            for values in series
        ):
            raise ValueError("Waterbirds production series must align by final seed")
        result_paths = tuple(path for _, path in self.result_paths_by_seed)
        if len(set(result_paths)) != 10:
            raise ValueError("Waterbirds final result paths must be unique")
        metrics: tuple[
            tuple[MetricName, tuple[float, ...], WaterbirdsMetricSummary], ...
        ] = (
            (
                "worst_group_accuracy",
                tuple(float(value) for _, value in self.worst_group_by_seed),
                self.worst_group_summary,
            ),
            (
                "adjusted_average_accuracy",
                tuple(float(value) for _, value in self.adjusted_average_by_seed),
                self.adjusted_average_summary,
            ),
            (
                "raw_average_accuracy",
                tuple(float(value) for _, value in self.raw_average_by_seed),
                self.raw_average_summary,
            ),
        )
        from grit.waterbirds_run_contracts import make_waterbirds_metric_summary

        for name, values, summary in metrics:
            if summary != make_waterbirds_metric_summary(name, values):
                raise ValueError("Waterbirds production metric summary is inconsistent")
        return self


class WaterbirdsProductionSummary(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-production-summary/v1"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    lineage: SearchLineage
    methods: tuple[
        WaterbirdsProductionMethodSummary,
        WaterbirdsProductionMethodSummary,
    ]
    paired_worst_group_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...]
    paired_worst_group_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_paired(self) -> WaterbirdsProductionSummary:
        if tuple(item.method_id for item in self.methods) != ("erm", "grit"):
            raise ValueError("Waterbirds production methods must be ERM then GRIT")
        if any(item.lineage != self.lineage for item in self.methods):
            raise ValueError("Waterbirds production summary lineage is inconsistent")
        erm, grit = self.methods
        if erm.configured_final_seeds != grit.configured_final_seeds:
            raise ValueError("Waterbirds paired methods require identical final seeds")
        erm_values = dict(erm.worst_group_by_seed)
        grit_values = dict(grit.worst_group_by_seed)
        expected = tuple(
            (seed, float(grit_values[seed]) - float(erm_values[seed]))
            for seed in erm.configured_final_seeds
        )
        if self.paired_worst_group_by_seed != expected:
            raise ValueError("Waterbirds paired differences are inconsistent")
        from grit.waterbirds_run_contracts import make_waterbirds_metric_summary

        if self.paired_worst_group_summary != make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(value for _, value in expected),
        ):
            raise ValueError("Waterbirds paired summary is inconsistent")
        return self


class WaterbirdsPairedSummaryArtifact(StrictBoundaryModel):
    schema_version: Literal["grit.waterbirds-paired-summary/v1"]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    paired_worst_group_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...]
    paired_worst_group_summary: WaterbirdsMetricSummary

    @model_validator(mode="after")
    def _validate_summary(self) -> WaterbirdsPairedSummaryArtifact:
        if tuple(seed for seed, _ in self.paired_worst_group_by_seed) != (
            self.configured_final_seeds
        ):
            raise ValueError("Waterbirds paired artifact must align by final seed")
        from grit.waterbirds_run_contracts import make_waterbirds_metric_summary

        if self.paired_worst_group_summary != make_waterbirds_metric_summary(
            "grit_minus_erm_worst_group_accuracy",
            tuple(value for _, value in self.paired_worst_group_by_seed),
        ):
            raise ValueError("Waterbirds paired artifact summary is inconsistent")
        return self


class IndexedArtifact(StrictBoundaryModel):
    kind: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: NonEmptyStr
    schema_version: NonEmptyStr | None

    @model_validator(mode="after")
    def _validate_path(self) -> IndexedArtifact:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or self.relative_path != path.as_posix()
        ):
            raise ValueError(
                "experiment-index paths must remain inside the output root"
            )
        if not self.sha256.startswith("sha256:"):
            raise ValueError("experiment-index artifacts require SHA-256 digests")
        return self


class ExperimentIndex(StrictBoundaryModel):
    schema_version: Literal["grit.experiment-index/v1"]
    dataset: Literal["cmnist", "waterbirds_cf"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    resolved_config_digest: NonEmptyStr
    artifacts: tuple[IndexedArtifact, ...]

    @model_validator(mode="after")
    def _validate_artifacts(self) -> ExperimentIndex:
        paths = tuple(item.relative_path for item in self.artifacts)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(
                "experiment-index artifact paths must be nonempty and unique"
            )
        if self.artifacts != tuple(
            sorted(self.artifacts, key=lambda item: item.relative_path)
        ):
            raise ValueError("experiment-index artifacts must be canonically ordered")
        return self


def verify_experiment_index(root: Path, index: ExperimentIndex) -> ExperimentIndex:
    """Reparse an index and verify every addressed local artifact byte-for-byte."""

    validated = ExperimentIndex.model_validate_json(index.canonical_json())
    by_path = {artifact.relative_path: artifact for artifact in validated.artifacts}
    if "search-plan.json" not in by_path or "resolved-config.json" not in by_path:
        raise ValueError("experiment index lacks plan/config authority artifacts")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (*root.rglob("*.json"), *root.rglob("*.yaml"))
        if path != root / "experiment-index.json"
        and not any(
            ".incomplete" in part or ".interrupted-" in part
            for part in path.parts
        )
    }
    if set(by_path) != actual_paths:
        raise ValueError("experiment index does not enumerate the complete output tree")
    for artifact in validated.artifacts:
        path = root / artifact.relative_path
        if not path.is_file() or _file_sha256(path) != artifact.sha256:
            raise ValueError(
                f"experiment-index artifact digest mismatch: {artifact.relative_path}"
            )
    from grit.search import ResolvedProductionSearchConfig, SearchPlan

    plan = SearchPlan.model_validate_json(
        (root / "search-plan.json").read_text(encoding="utf-8")
    )
    resolved = ResolvedProductionSearchConfig.model_validate_json(
        (root / "resolved-config.json").read_text(encoding="utf-8")
    )
    if (
        plan.canonical_digest() != validated.plan_digest
        or resolved.canonical_digest() != validated.resolved_config_digest
        or plan.resolved_config != resolved
        or plan.dataset != validated.dataset
    ):
        raise ValueError("experiment index plan/config identity is inconsistent")
    return validated


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
