"""Dataset-specific production summaries and top-level experiment index."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, FiniteFloat, StrictInt, StrictStr, model_validator

from grit.methods.types import IMPLEMENTED_METHODS, MethodId
from grit.schemas import ORDINARY_CMNIST_SELECTORS, CmnistSelector, StrictBoundaryModel
from grit.search.plan import SearchLineage
from grit.search.waterbirds_contracts import MetricName, WaterbirdsMetricSummary
from grit.selection.waterbirds import WaterbirdsSelector

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]


class CmnistAccuracySummary(StrictBoundaryModel):
    metric_name: Literal["test_ood_accuracy", "grit_minus_erm_test_ood_accuracy"]
    seed_count: Literal[10]
    mean: FiniteFloat
    sample_standard_deviation: Annotated[FiniteFloat, Field(ge=0.0)]
    ci95_lower: FiniteFloat
    ci95_upper: FiniteFloat


def make_cmnist_accuracy_summary(
    metric_name: Literal["test_ood_accuracy", "grit_minus_erm_test_ood_accuracy"],
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
    method_id: MethodId
    selector: CmnistSelector
    result_path: NonEmptyStr
    metric_record_id: NonEmptyStr
    test_ood_accuracy: FiniteFloat


class CmnistMethodSelectorSummary(StrictBoundaryModel):
    method_id: MethodId
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
        if (
            len({item.result_path for item in self.final_observations}) != 10
            or len({item.metric_record_id for item in self.final_observations}) != 10
        ):
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
    """One search tree's report; `selectors` names its track (ordinary or oracle)."""

    schema_version: Literal["grit.cmnist-production-summary/v1"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    lineage: SearchLineage
    selectors: tuple[CmnistSelector, ...] = ORDINARY_CMNIST_SELECTORS
    methods: Annotated[tuple[CmnistMethodSelectorSummary, ...], Field(min_length=1)]
    paired_selectors: tuple[CmnistPairedSelectorSummary, ...]

    @property
    def test_oracle(self) -> bool:
        return self.selectors == (CmnistSelector.TEST_ORACLE,)

    @model_validator(mode="after")
    def _validate_methods(self) -> CmnistProductionSummary:
        if self.selectors not in (
            ORDINARY_CMNIST_SELECTORS,
            (CmnistSelector.TEST_ORACLE,),
        ):
            raise ValueError(
                "CMNIST summaries use both ordinary selectors or only test_oracle"
            )
        observed_methods = tuple(dict.fromkeys(item.method_id for item in self.methods))
        if tuple(sorted(observed_methods, key=IMPLEMENTED_METHODS.index)) != (
            observed_methods
        ):
            raise ValueError("CMNIST production methods are misordered")
        expected = tuple(
            (method, selector)
            for method in observed_methods
            for selector in self.selectors
        )
        if tuple((item.method_id, item.selector) for item in self.methods) != expected:
            raise ValueError("CMNIST production summaries require canonical ordering")
        if any(item.lineage != self.lineage for item in self.methods):
            raise ValueError("CMNIST production summary lineage is inconsistent")
        expected_paired_selectors = (
            self.selectors
            if "erm" in observed_methods and "grit" in observed_methods
            else ()
        )
        if tuple(item.selector for item in self.paired_selectors) != (
            expected_paired_selectors
        ):
            raise ValueError("CMNIST paired selector summaries are misordered")
        by_identity = {(item.method_id, item.selector): item for item in self.methods}
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


class RotatedMnistAccuracySummary(StrictBoundaryModel):
    metric_name: Literal["test_r90_accuracy", "grit_minus_erm_test_r90_accuracy"]
    seed_count: Literal[10]
    mean: FiniteFloat
    sample_standard_deviation: Annotated[FiniteFloat, Field(ge=0.0)]
    ci95_lower: FiniteFloat
    ci95_upper: FiniteFloat


def make_rotated_mnist_accuracy_summary(
    metric_name: Literal["test_r90_accuracy", "grit_minus_erm_test_r90_accuracy"],
    values: tuple[float, ...],
) -> RotatedMnistAccuracySummary:
    if len(values) != 10:
        raise ValueError("RotatedMNIST final summary requires ten seeds")
    mean = fmean(values)
    deviation = stdev(values)
    half_width = 2.2621571627409915 * deviation / math.sqrt(10)
    return RotatedMnistAccuracySummary(
        metric_name=metric_name,
        seed_count=10,
        mean=mean,
        sample_standard_deviation=deviation,
        ci95_lower=mean - half_width,
        ci95_upper=mean + half_width,
    )


class RotatedMnistFinalSeedObservation(StrictBoundaryModel):
    seed: StrictInt
    method_id: MethodId
    selector: CmnistSelector
    result_path: NonEmptyStr
    metric_record_id: NonEmptyStr
    test_r90_accuracy: FiniteFloat


class RotatedMnistMethodSelectorSummary(StrictBoundaryModel):
    method_id: MethodId
    selector: CmnistSelector
    lineage: SearchLineage
    selected_candidate_id: NonEmptyStr
    finalist_candidate_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    final_observations: Annotated[
        tuple[RotatedMnistFinalSeedObservation, ...],
        Field(min_length=10, max_length=10),
    ]
    accuracy_summary: RotatedMnistAccuracySummary

    @model_validator(mode="after")
    def _validate_summary(self) -> RotatedMnistMethodSelectorSummary:
        if (
            len(set(self.finalist_candidate_ids)) != 3
            or self.selected_candidate_id not in self.finalist_candidate_ids
        ):
            raise ValueError("selected candidate must be one of three finalists")
        if tuple(item.seed for item in self.final_observations) != (
            self.configured_final_seeds
        ):
            raise ValueError("final observations must align by configured seed")
        if any(
            item.method_id != self.method_id or item.selector is not self.selector
            for item in self.final_observations
        ):
            raise ValueError("final observation identity is inconsistent")
        if (
            len({item.result_path for item in self.final_observations}) != 10
            or len({item.metric_record_id for item in self.final_observations}) != 10
        ):
            raise ValueError("final observations must be uniquely attributable")
        values = tuple(
            float(item.test_r90_accuracy) for item in self.final_observations
        )
        if self.accuracy_summary != make_rotated_mnist_accuracy_summary(
            "test_r90_accuracy", values
        ):
            raise ValueError("RotatedMNIST accuracy summary is inconsistent")
        return self


class RotatedMnistPairedSeedDifference(StrictBoundaryModel):
    seed: StrictInt
    selector: CmnistSelector
    grit_minus_erm_test_r90_accuracy: FiniteFloat


class RotatedMnistPairedSelectorSummary(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-paired-summary/v1"] = (
        "grit.rotated-mnist-paired-summary/v1"
    )
    selector: CmnistSelector
    configured_final_seeds: Annotated[
        tuple[StrictInt, ...], Field(min_length=10, max_length=10)
    ]
    paired_differences: Annotated[
        tuple[RotatedMnistPairedSeedDifference, ...],
        Field(min_length=10, max_length=10),
    ]
    difference_summary: RotatedMnistAccuracySummary

    @model_validator(mode="after")
    def _validate_pairs(self) -> RotatedMnistPairedSelectorSummary:
        if tuple(item.seed for item in self.paired_differences) != (
            self.configured_final_seeds
        ) or any(
            item.selector is not self.selector for item in self.paired_differences
        ):
            raise ValueError("paired differences must align by configured seed")
        values = tuple(
            float(item.grit_minus_erm_test_r90_accuracy)
            for item in self.paired_differences
        )
        if self.difference_summary != make_rotated_mnist_accuracy_summary(
            "grit_minus_erm_test_r90_accuracy", values
        ):
            raise ValueError("RotatedMNIST paired summary is inconsistent")
        return self


class RotatedMnistProductionSummary(StrictBoundaryModel):
    schema_version: Literal["grit.rotated-mnist-production-summary/v1"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    lineage: SearchLineage
    methods: tuple[
        RotatedMnistMethodSelectorSummary,
        RotatedMnistMethodSelectorSummary,
        RotatedMnistMethodSelectorSummary,
        RotatedMnistMethodSelectorSummary,
    ]
    paired_selectors: tuple[
        RotatedMnistPairedSelectorSummary, RotatedMnistPairedSelectorSummary
    ]

    @model_validator(mode="after")
    def _validate_methods(self) -> RotatedMnistProductionSummary:
        expected = tuple(
            (method, selector)
            for method in ("erm", "grit")
            for selector in (
                CmnistSelector.PRIMARY_ROBUST,
                CmnistSelector.SECONDARY_SOURCE,
            )
        )
        if tuple((item.method_id, item.selector) for item in self.methods) != expected:
            raise ValueError("RotatedMNIST summaries must be ERM then GRIT")
        if any(item.lineage != self.lineage for item in self.methods):
            raise ValueError("RotatedMNIST summary lineage is inconsistent")
        if tuple(item.selector for item in self.paired_selectors) != (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        ):
            raise ValueError("RotatedMNIST paired selectors are misordered")
        by_identity = {(item.method_id, item.selector): item for item in self.methods}
        for paired in self.paired_selectors:
            erm = by_identity[("erm", paired.selector)]
            grit = by_identity[("grit", paired.selector)]
            if erm.configured_final_seeds != grit.configured_final_seeds:
                raise ValueError("paired methods require identical final seeds")
            erm_values = {
                item.seed: float(item.test_r90_accuracy)
                for item in erm.final_observations
            }
            grit_values = {
                item.seed: float(item.test_r90_accuracy)
                for item in grit.final_observations
            }
            expected_pairs = tuple(
                RotatedMnistPairedSeedDifference(
                    seed=seed,
                    selector=paired.selector,
                    grit_minus_erm_test_r90_accuracy=(
                        grit_values[seed] - erm_values[seed]
                    ),
                )
                for seed in erm.configured_final_seeds
            )
            if paired.paired_differences != expected_pairs:
                raise ValueError("RotatedMNIST paired differences are inconsistent")
        return self


class WaterbirdsProductionMethodSummary(StrictBoundaryModel):
    method_id: MethodId
    selector: WaterbirdsSelector
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
        from grit.search.waterbirds_contracts import make_waterbirds_metric_summary

        for name, values, summary in metrics:
            if summary != make_waterbirds_metric_summary(name, values):
                raise ValueError("Waterbirds production metric summary is inconsistent")
        return self


class WaterbirdsProductionSummary(StrictBoundaryModel):
    """Ten-seed summaries for one Waterbirds tree; one selector per tree.

    `selector: test_oracle` summaries are the paper's oracle-validation envelope and
    are reported only under that heading.
    """

    schema_version: Literal["grit.waterbirds-production-summary/v2"]
    reportable: Literal[True]
    plan_digest: NonEmptyStr
    lineage: SearchLineage
    selector: WaterbirdsSelector
    methods: Annotated[
        tuple[WaterbirdsProductionMethodSummary, ...], Field(min_length=1)
    ]
    paired_worst_group_by_seed: tuple[tuple[StrictInt, FiniteFloat], ...] | None
    paired_worst_group_summary: WaterbirdsMetricSummary | None

    @property
    def test_oracle(self) -> bool:
        return self.selector == "test_oracle"

    @property
    def paired(self) -> bool:
        return self.paired_worst_group_by_seed is not None

    @model_validator(mode="after")
    def _validate_paired(self) -> WaterbirdsProductionSummary:
        method_ids = tuple(item.method_id for item in self.methods)
        if len(set(method_ids)) != len(method_ids) or method_ids != tuple(
            sorted(method_ids, key=IMPLEMENTED_METHODS.index)
        ):
            raise ValueError("Waterbirds production methods must be unique and ordered")
        if any(item.lineage != self.lineage for item in self.methods):
            raise ValueError("Waterbirds production summary lineage is inconsistent")
        if any(item.selector != self.selector for item in self.methods):
            raise ValueError("Waterbirds production summary mixes selectors")
        seeds = {item.configured_final_seeds for item in self.methods}
        if len(seeds) != 1:
            raise ValueError("Waterbirds paired methods require identical final seeds")
        by_method = {item.method_id: item for item in self.methods}
        pairable = "erm" in by_method and "grit" in by_method
        if pairable != (self.paired_worst_group_by_seed is not None) or pairable != (
            self.paired_worst_group_summary is not None
        ):
            raise ValueError(
                "Waterbirds paired differences exist exactly when ERM and GRIT are "
                "both summarized"
            )
        if not pairable:
            return self
        erm_values = dict(by_method["erm"].worst_group_by_seed)
        grit_values = dict(by_method["grit"].worst_group_by_seed)
        expected = tuple(
            (seed, float(grit_values[seed]) - float(erm_values[seed]))
            for seed in by_method["erm"].configured_final_seeds
        )
        if self.paired_worst_group_by_seed != expected:
            raise ValueError("Waterbirds paired differences are inconsistent")
        from grit.search.waterbirds_contracts import make_waterbirds_metric_summary

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
        from grit.search.waterbirds_contracts import make_waterbirds_metric_summary

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
    dataset: Literal["cmnist", "waterbirds_cf", "rotated_mnist"]
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
            ".incomplete" in part or ".interrupted-" in part for part in path.parts
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
    from grit.search.plan import ResolvedProductionSearchConfig, SearchPlan

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
