"""Minimal strict experiment configuration boundaries for Milestone 3."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    PositiveInt,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from grit.schemas import CmnistSelector, StrictBoundaryModel, canonical_digest_value

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]


class CmnistDatasetConfig(StrictBoundaryModel):
    """Dataset identity without choosing the unresolved partition algorithm."""

    dataset_id: Literal["cmnist"]
    construction_method_id: NonEmptyStr
    training_split_names: tuple[Literal["train_e01", "train_e02"], ...]
    validation_split_names: tuple[Literal["val_e01", "val_e02", "val_e05"], ...]
    final_test_split_name: Literal["test_ood"]

    @model_validator(mode="after")
    def _validate_required_splits(self) -> CmnistDatasetConfig:
        expected_training = ("train_e01", "train_e02")
        expected_validation = ("val_e01", "val_e02", "val_e05")
        if self.training_split_names != expected_training:
            raise ValueError(f"training_split_names must be {expected_training!r}")
        if self.validation_split_names != expected_validation:
            raise ValueError(f"validation_split_names must be {expected_validation!r}")
        return self


class FrozenFeatureConfig(StrictBoundaryModel):
    """The only representation admitted by the initial contract boundary."""

    kind: Literal["frozen_features"]
    encoder_id: NonEmptyStr
    normalization: Literal["none", "l2"]


class DisabledPairsConfig(StrictBoundaryModel):
    kind: Literal["disabled"]


class OraclePairsConfig(StrictBoundaryModel):
    """Configuration identity only; this does not implement pair construction."""

    kind: Literal["oracle"]
    construction_id: NonEmptyStr
    source_partition_ids: tuple[NonEmptyStr, ...]
    pair_count: PositiveInt

    @model_validator(mode="after")
    def _validate_sources(self) -> OraclePairsConfig:
        expected_sources = ("train_e01_sources", "train_e02_sources")
        if self.source_partition_ids != expected_sources:
            raise ValueError(
                "initial CMNIST oracle pairs require the approved training source "
                f"partitions {expected_sources!r}"
            )
        if self.pair_count != 256:
            raise ValueError("initial CMNIST oracle pairs require pair_count=256")
        return self


PairsConfig: TypeAlias = Annotated[
    DisabledPairsConfig | OraclePairsConfig,
    Field(discriminator="kind"),
]


class DisabledProjectionConfig(StrictBoundaryModel):
    kind: Literal["disabled"]


class LinearProjectionConfig(StrictBoundaryModel):
    """Projection settings only; Milestone 3 performs no projection mathematics."""

    kind: Literal["linear_pair_difference"]
    requested_rank: Annotated[StrictInt, Field(ge=0, le=24)]
    center_differences: Literal[False]


ProjectionConfig: TypeAlias = Annotated[
    DisabledProjectionConfig | LinearProjectionConfig,
    Field(discriminator="kind"),
]


class ErmAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["erm"]


class GritAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["grit"]


AlgorithmConfig: TypeAlias = Annotated[
    ErmAlgorithmConfig | GritAlgorithmConfig,
    Field(discriminator="kind"),
]


class SeedSets(StrictBoundaryModel):
    """Prespecified search stages; scheduling remains outside Milestone 3."""

    tuning: tuple[StrictInt, ...]
    confirmation: tuple[StrictInt, ...]
    final: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def _validate_seed_sets(self) -> SeedSets:
        named = {
            "tuning": self.tuning,
            "confirmation": self.confirmation,
            "final": self.final,
        }
        expected_counts = {"tuning": 3, "confirmation": 2, "final": 10}
        for name, values in named.items():
            if not values:
                raise ValueError(f"{name} seed set must not be empty")
            if len(values) != expected_counts[name]:
                raise ValueError(
                    f"{name} seed set must contain {expected_counts[name]} seeds"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"{name} seed set contains duplicates")
        if set(self.tuning) & set(self.confirmation):
            raise ValueError("tuning and confirmation seeds must be disjoint")
        if set(self.tuning) & set(self.final):
            raise ValueError("tuning and final seeds must be disjoint")
        if set(self.confirmation) & set(self.final):
            raise ValueError("confirmation and final seeds must be disjoint")
        return self


class OrdinarySelectionConfig(StrictBoundaryModel):
    selector: CmnistSelector


class CmnistTestOracleSelectionConfig(StrictBoundaryModel):
    selector: Literal["test_ood_accuracy"]
    test_oracle: Literal[True]


class _CommonCmnistExperimentConfig(StrictBoundaryModel):
    schema_version: Literal["grit.experiment/v1"]
    experiment_name: NonEmptyStr
    protocol_id: Literal["cmnist/v1"]
    dataset: CmnistDatasetConfig
    representation: FrozenFeatureConfig
    pairs: PairsConfig
    projection: ProjectionConfig
    algorithm: AlgorithmConfig
    seed_sets: SeedSets

    def scientific_config_digest(self) -> str:
        """Identify trainable candidate fields independently of selector and seeds."""

        candidate_fields = self.model_dump(
            mode="json",
            exclude={
                "diagnostic_selection",
                "experiment_name",
                "run_kind",
                "seed_sets",
                "selection",
            },
        )
        return canonical_digest_value(candidate_fields)

    @model_validator(mode="after")
    def _validate_algorithm_components(self) -> _CommonCmnistExperimentConfig:
        if isinstance(self.algorithm, ErmAlgorithmConfig):
            if not isinstance(self.pairs, DisabledPairsConfig):
                raise ValueError("ERM requires pairs.kind='disabled'")
            if not isinstance(self.projection, DisabledProjectionConfig):
                raise ValueError("ERM requires projection.kind='disabled'")
        else:
            if not isinstance(self.pairs, OraclePairsConfig):
                raise ValueError("initial GRIT requires pairs.kind='oracle'")
            if not isinstance(self.projection, LinearProjectionConfig):
                raise ValueError(
                    "initial GRIT requires projection.kind='linear_pair_difference'"
                )
        return self


class OrdinaryExperimentConfig(_CommonCmnistExperimentConfig):
    run_kind: Literal["ordinary"]
    selection: OrdinarySelectionConfig


class CmnistTestOracleExperimentConfig(_CommonCmnistExperimentConfig):
    run_kind: Literal["cmnist_test_oracle_diagnostic"]
    diagnostic_selection: CmnistTestOracleSelectionConfig


ExperimentConfig: TypeAlias = Annotated[
    OrdinaryExperimentConfig | CmnistTestOracleExperimentConfig,
    Field(discriminator="run_kind"),
]

_EXPERIMENT_CONFIG_ADAPTER: TypeAdapter[ExperimentConfig] = TypeAdapter(
    ExperimentConfig
)


def parse_experiment_config_json(payload: str) -> ExperimentConfig:
    """Parse canonical or authored JSON through the discriminated strict root."""

    return _EXPERIMENT_CONFIG_ADAPTER.validate_json(payload)
