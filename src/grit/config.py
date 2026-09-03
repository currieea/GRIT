"""Minimal strict experiment configuration boundaries for Milestone 3."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    FiniteFloat,
    PositiveInt,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    model_validator,
)

from grit.schemas import CmnistSelector, StrictBoundaryModel, canonical_digest_value

NonEmptyStr: TypeAlias = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt: TypeAlias = Annotated[StrictInt, Field(ge=0)]
Probability: TypeAlias = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]

OPENAI_CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
OPENAI_CLIP_WEIGHTS_IDENTITY = (
    "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
)
OPENAI_CLIP_PREPROCESSING_ID = f"openai-clip-vit-b32-preprocess@{OPENAI_CLIP_REVISION}"


class CmnistSourceCounts(StrictBoundaryModel):
    train_e01: PositiveInt
    train_e02: PositiveInt
    validation: PositiveInt
    test: PositiveInt


class CmnistDatasetConfig(StrictBoundaryModel):
    """Dataset identity bound to the approved deterministic partition algorithm."""

    dataset_id: Literal["cmnist"]
    construction_method_id: Literal["cmnist-stratified-hash-v1"]
    construction_seed: StrictInt
    label_flip_prob: Probability
    source_counts: CmnistSourceCounts
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
    encoder_id: Literal["openai-clip-vit-b32", "synthetic-fake-512"]
    encoder_revision: NonEmptyStr
    weights_identity: NonEmptyStr
    preprocessing_identity: NonEmptyStr
    feature_dimension: Literal[512]
    normalization: Literal["none", "l2"]


class DisabledPairsConfig(StrictBoundaryModel):
    kind: Literal["disabled"]


class OraclePairsConfig(StrictBoundaryModel):
    """Configuration identity only; this does not implement pair construction."""

    kind: Literal["oracle"]
    construction_id: Literal["cmnist-clean-oracle-pairs-v1"]
    source_partition_ids: tuple[NonEmptyStr, ...]
    pair_count: PositiveInt
    pair_seed: StrictInt
    orientation: Literal["red_minus_green"]

    @model_validator(mode="after")
    def _validate_sources(self) -> OraclePairsConfig:
        expected_sources = ("train_e01_sources", "train_e02_sources")
        if self.source_partition_ids != expected_sources:
            raise ValueError(
                "initial CMNIST oracle pairs require the approved training source "
                f"partitions {expected_sources!r}"
            )
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
    relative_singular_value_tolerance: Annotated[
        StrictFloat, Field(gt=0.0)
    ]


ProjectionConfig: TypeAlias = Annotated[
    DisabledProjectionConfig | LinearProjectionConfig,
    Field(discriminator="kind"),
]


class ErmAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["erm"]


class GritAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["grit"]


class GroupDroAlgorithmConfig(StrictBoundaryModel):
    kind: Literal["groupdro"]
    group_definition: Literal["target_color", "target_background"]
    adversarial_step_size: Annotated[StrictFloat, Field(gt=0.0)]
    sampling: Literal["inverse_group_frequency_with_replacement"]
    generalization_adjustment: Annotated[StrictFloat, Field(ge=0.0, le=0.0)]
    normalize_loss: Literal[False]


AlgorithmConfig: TypeAlias = Annotated[
    ErmAlgorithmConfig | GritAlgorithmConfig | GroupDroAlgorithmConfig,
    Field(discriminator="kind"),
]


class LinearProbeTrainingConfig(StrictBoundaryModel):
    optimizer: Literal["adam"]
    batch_size: PositiveInt
    learning_rate: Annotated[StrictFloat, Field(gt=0.0)]
    weight_decay: Annotated[StrictFloat, Field(ge=0.0)]
    max_epochs: PositiveInt


class CpuRuntimeConfig(StrictBoundaryModel):
    device: Literal["cpu"]
    deterministic_algorithms: Literal[True]


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


class CmnistArtifactLineageConfig(StrictBoundaryModel):
    """Prepared-artifact identity required by reportable CMNIST runs."""

    dataset_manifest_digest: NonEmptyStr
    feature_cache_manifest_digest: NonEmptyStr
    pair_manifest_digest: NonEmptyStr | None


class _CommonCmnistExperimentConfig(StrictBoundaryModel):
    schema_version: Literal["grit.experiment/v1"]
    experiment_name: NonEmptyStr
    protocol_id: Literal["cmnist/v1"]
    reportable: StrictBool
    dataset: CmnistDatasetConfig
    representation: FrozenFeatureConfig
    pairs: PairsConfig
    projection: ProjectionConfig
    algorithm: AlgorithmConfig
    training: LinearProbeTrainingConfig
    runtime: CpuRuntimeConfig
    seed_sets: SeedSets
    artifact_lineage: CmnistArtifactLineageConfig | None = None

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
        if isinstance(self.algorithm, ErmAlgorithmConfig | GroupDroAlgorithmConfig):
            if not isinstance(self.pairs, DisabledPairsConfig):
                raise ValueError("ERM and GroupDRO require pairs.kind='disabled'")
            if not isinstance(self.projection, DisabledProjectionConfig):
                raise ValueError("ERM and GroupDRO require projection.kind='disabled'")
            if (
                isinstance(self.algorithm, GroupDroAlgorithmConfig)
                and self.algorithm.group_definition != "target_color"
            ):
                raise ValueError("CMNIST GroupDRO requires target-color groups")
        else:
            if not isinstance(self.pairs, OraclePairsConfig):
                raise ValueError("initial GRIT requires pairs.kind='oracle'")
            if not isinstance(self.projection, LinearProjectionConfig):
                raise ValueError(
                    "initial GRIT requires projection.kind='linear_pair_difference'"
                )
        if self.reportable:
            if self.artifact_lineage is None:
                raise ValueError("reportable CMNIST requires prepared-artifact lineage")
            if isinstance(self.algorithm, ErmAlgorithmConfig | GroupDroAlgorithmConfig):
                if self.artifact_lineage.pair_manifest_digest is not None:
                    raise ValueError(
                        "reportable CMNIST ERM and GroupDRO cannot bind oracle pairs"
                    )
            elif self.artifact_lineage.pair_manifest_digest is None:
                raise ValueError("reportable CMNIST GRIT requires oracle-pair lineage")
            counts = self.dataset.source_counts
            if (counts.train_e01, counts.train_e02, counts.validation, counts.test) != (
                25_000,
                25_000,
                10_000,
                10_000,
            ):
                raise ValueError("reportable CMNIST requires production source counts")
            if float(self.dataset.label_flip_prob) != 0.25:
                raise ValueError("reportable CMNIST requires label_flip_prob=0.25")
            if self.representation.encoder_id != "openai-clip-vit-b32":
                raise ValueError("reportable CMNIST requires official OpenAI CLIP")
            representation_identity = (
                self.representation.encoder_revision,
                self.representation.weights_identity,
                self.representation.preprocessing_identity,
            )
            if representation_identity != (
                OPENAI_CLIP_REVISION,
                OPENAI_CLIP_WEIGHTS_IDENTITY,
                OPENAI_CLIP_PREPROCESSING_ID,
            ):
                raise ValueError(
                    "reportable CMNIST requires the pinned official CLIP identity"
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
