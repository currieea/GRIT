"""Canonical production-search configuration and planning tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
from pydantic import ValidationError

from grit.cmnist import (
    CMNIST_ENVIRONMENT_SPECS,
    CmnistDatasetManifest,
    CmnistOraclePairManifest,
    CmnistPartitionManifest,
    CmnistPartitionTargets,
    DigitCount,
    EnvironmentManifest,
    OraclePairRecord,
    SourcePartitionRecord,
)
from grit.config import SeedSets
from grit.features import (
    ArrayFileManifest,
    CmnistFeatureCacheManifest,
    EncoderIdentity,
    FeatureTableManifest,
)
from grit.production_search import (
    materialize_cmnist_candidate_config,
    plan_production_search,
)
from grit.production_waterbirds_search import (
    materialize_waterbirds_candidate_config,
)
from grit.projection import fit_linear_projection
from grit.results import CodeProvenance, EnvironmentProvenance
from grit.schemas import CmnistSelector, SeedStage, canonical_digest_value
from grit.search import (
    APPROVED_LEARNING_RATES,
    APPROVED_RANKS,
    APPROVED_WEIGHT_DECAYS,
    CmnistProductionSearchConfig,
    ResolvedProductionSearchConfig,
    SearchArtifactPaths,
    SearchLineage,
    SearchPlan,
    SearchRuntimeConfig,
    SearchSeedConfig,
    SearchSpaceConfig,
    VerifiedInputArtifact,
    WaterbirdsProductionSearchConfig,
    build_search_plan,
    load_production_search_config,
)
from grit.search_scheduler import (
    CmnistCompletedStageRun,
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    make_search_task,
)
from grit.selection import (
    ValidationMetricRecord,
    make_tuning_finalists,
    select_checkpoint,
)
from tests.production_artifact_fixtures import (
    write_manifest_only_waterbirds_production,
)


def _seeds() -> SearchSeedConfig:
    return SearchSeedConfig(
        construction=17,
        pairs=23,
        stages=SeedSets(
            tuning=(101, 102, 103),
            confirmation=(201, 202),
            final=(301, 302, 303, 304, 305, 306, 307, 308, 309, 310),
        ),
    )


def _space() -> SearchSpaceConfig:
    return SearchSpaceConfig(
        methods=("erm", "grit"),
        learning_rates=APPROVED_LEARNING_RATES,
        weight_decays=APPROVED_WEIGHT_DECAYS,
        projection_ranks=APPROVED_RANKS,
    )


def _config(dataset: str = "cmnist") -> (
    CmnistProductionSearchConfig | WaterbirdsProductionSearchConfig
):
    common: dict[str, object] = {
        "schema_version": "grit.production-search/v1",
        "experiment_name": f"{dataset}-primary",
        "experiment_variant": "primary_unnormalized",
        "normalization": "none",
        "artifacts": SearchArtifactPaths(
            dataset_manifest="dataset.json",
            feature_cache_manifest="features.json",
            oracle_pair_manifest="pairs.json",
        ),
        "seeds": _seeds(),
        "search_space": _space(),
        "output_root": "output",
        "runtime": SearchRuntimeConfig(
            device="cpu", deterministic_algorithms=True, workers=1
        ),
        "relative_singular_value_tolerance": 1e-12,
    }
    if dataset == "cmnist":
        return CmnistProductionSearchConfig.model_validate(
            {
                **common,
                "dataset": "cmnist",
                "protocol_id": "cmnist/v1",
                "pair_count": 256,
                "selectors": ("primary_robust", "secondary_source"),
                "batch_size": 256,
                "max_epochs": 40,
            }
        )
    return WaterbirdsProductionSearchConfig.model_validate(
        {
            **common,
            "dataset": "waterbirds_cf",
            "protocol_id": "waterbirds_cf/v1",
            "pair_count": 240,
            "selectors": ("waterbirds_validation_worst_group",),
            "batch_size": 256,
            "max_epochs": 100,
        }
    )


def resolved_search_fixture(
    dataset: str = "cmnist",
) -> ResolvedProductionSearchConfig:
    config = _config(dataset)
    schema_versions = (
        (
            "grit.cmnist-dataset/v1",
            "grit.cmnist-features/v1",
            "grit.cmnist-oracle-pairs/v2",
        )
        if dataset == "cmnist"
        else (
            "grit.waterbirds-cf-dataset/v2",
            "grit.waterbirds-features/v1",
            "grit.waterbirds-oracle-pairs/v2",
        )
    )
    inputs = (
        VerifiedInputArtifact(
            kind="dataset_manifest",
            path="/prepared/dataset_manifest.json",
            digest="sha256:dataset",
            schema_version=schema_versions[0],
        ),
        VerifiedInputArtifact(
            kind="feature_manifest",
            path="/prepared/feature_manifest.json",
            digest="sha256:features",
            schema_version=schema_versions[1],
        ),
        VerifiedInputArtifact(
            kind="pair_manifest",
            path="/prepared/pair_manifest.json",
            digest="sha256:pairs",
            schema_version=schema_versions[2],
        ),
    )
    return ResolvedProductionSearchConfig(
        schema_version="grit.resolved-production-search/v1",
        authored_config_digest=config.canonical_digest(),
        authored_config_path=f"/configs/{dataset}.yaml",
        output_root=f"/outputs/{dataset}",
        config=config,
        lineage=SearchLineage(
            dataset_manifest_digest="sha256:dataset",
            feature_cache_manifest_digest="sha256:features",
            pair_manifest_digest="sha256:pairs",
            normalization="none",
            adjusted_weight_spec_digest=(
                "sha256:weights" if dataset == "waterbirds_cf" else None
            ),
        ),
        input_artifacts=inputs,
    )


def _digit_counts(indices: tuple[int, ...]) -> tuple[DigitCount, ...]:
    return tuple(
        DigitCount(
            digit=digit,
            count=sum(index % 10 == digit for index in indices),
        )
        for digit in range(10)
    )


def _partition_record(
    name: str,
    official_split: str,
    indices: tuple[int, ...],
) -> SourcePartitionRecord:
    return SourcePartitionRecord.model_validate(
        {
            "name": name,
            "official_split": official_split,
            "source_indices": indices,
            "digit_counts": _digit_counts(indices),
            "membership_digest": canonical_digest_value(
                {
                    "name": name,
                    "official_split": official_split,
                    "source_indices": indices,
                }
            ),
        }
    )


def _write_manifest_only_cmnist_production(
    root: Path,
    *,
    encoder: EncoderIdentity | None = None,
    invalid_official_train_universe: bool = False,
) -> tuple[Path, Path, Path]:
    train_e01_indices = tuple(range(25_000))
    if invalid_official_train_universe:
        train_e01_indices = (*train_e01_indices[:-1], 60_000)
    partitions = (
        _partition_record("train_e01_sources", "train", train_e01_indices),
        _partition_record(
            "train_e02_sources", "train", tuple(range(25_000, 50_000))
        ),
        _partition_record(
            "validation_sources", "train", tuple(range(50_000, 60_000))
        ),
        _partition_record("test_sources", "test", tuple(range(10_000))),
    )
    partition_manifest = CmnistPartitionManifest(
        schema_version="grit.cmnist-partitions/v1",
        dataset_id="cmnist",
        construction_method_id="cmnist-stratified-hash-v1",
        construction_seed=17,
        targets=CmnistPartitionTargets(
            train_e01=25_000,
            train_e02=25_000,
            validation=10_000,
            test=10_000,
        ),
        partitions=partitions,
    )
    source_ids = {
        "train_e01": tuple(f"mnist:train:{index}" for index in range(25_000)),
        "train_e02": tuple(
            f"mnist:train:{index}" for index in range(25_000, 50_000)
        ),
        "validation": tuple(
            f"mnist:train:{index}" for index in range(50_000, 60_000)
        ),
        "test": tuple(f"mnist:test:{index}" for index in range(10_000)),
    }
    environments = tuple(
        EnvironmentManifest(
            name=spec.name,
            role=spec.role,
            source_partition_id=spec.source_partition_id,
            color_flip_prob=spec.color_flip_prob,
            count=(
                25_000
                if spec.name in {"train_e01", "train_e02"}
                else 10_000
            ),
            source_membership_digest=canonical_digest_value(
                source_ids[
                    "validation"
                    if spec.name.startswith("val_")
                    else "test"
                    if spec.name == "test_ood"
                    else spec.name
                ]
            ),
            record_digest=f"sha256:records:{spec.name}",
        )
        for spec in CMNIST_ENVIRONMENT_SPECS
    )
    dataset = CmnistDatasetManifest(
        schema_version="grit.cmnist-dataset/v1",
        dataset_id="cmnist",
        partition_manifest=partition_manifest,
        partition_manifest_digest=partition_manifest.canonical_digest(),
        official_train_pool_digest="sha256:official-train-pool",
        official_test_pool_digest="sha256:official-test-pool",
        rendering_method_id="cmnist-rgb-render-v1",
        construction_seed=17,
        label_flip_prob=0.25,
        environments=environments,
    )
    dataset_digest = dataset.canonical_digest()
    pair_records = tuple(
        OraclePairRecord(
            pair_id=canonical_digest_value(
                {
                    "dataset_manifest_digest": dataset_digest,
                    "method": "cmnist-clean-oracle-pairs-v1",
                    "pair_seed": 23,
                    "source_id": f"mnist:train:{index}",
                    "orientation": "red_minus_green",
                }
            ),
            source_id=f"mnist:train:{index}",
            official_source_index=index,
            digit=index % 10,
            clean_label=int(index % 10 >= 5),
            noisy_target=int(index % 10 >= 5),
            left_color="red",
            right_color="green",
        )
        for index in range(256)
    )
    pairs = CmnistOraclePairManifest(
        schema_version="grit.cmnist-oracle-pairs/v2",
        dataset_manifest_digest=dataset_digest,
        construction_method_id="cmnist-clean-oracle-pairs-v1",
        pair_seed=23,
        requested_count=256,
        realized_count=256,
        source_partition_ids=("train_e01_sources", "train_e02_sources"),
        orientation="red_minus_green",
        records=pair_records,
        membership_digest=canonical_digest_value(
            tuple(record.source_id for record in pair_records)
        ),
    )
    table_specs: tuple[
        tuple[
            str,
            Literal["training", "validation", "final_test", "pair_projection"],
            tuple[str, ...],
        ],
        ...,
    ] = (
        ("train_e01", "training", source_ids["train_e01"]),
        ("train_e02", "training", source_ids["train_e02"]),
        ("val_e01", "validation", source_ids["validation"]),
        ("val_e02", "validation", source_ids["validation"]),
        ("val_e05", "validation", source_ids["validation"]),
        ("test_ood", "final_test", source_ids["test"]),
        (
            "oracle_pair_red",
            "pair_projection",
            tuple(record.source_id for record in pair_records),
        ),
        (
            "oracle_pair_green",
            "pair_projection",
            tuple(record.source_id for record in pair_records),
        ),
    )
    feature_root = root / "feature-cache"
    tables: list[FeatureTableManifest] = []
    for table_name, role, table_sources in table_specs:
        files: list[ArrayFileManifest] = []
        for field_name, shape, dtype in (
            ("features", (len(table_sources), 512), "float32"),
            ("digits", (len(table_sources),), "int64"),
            ("clean_labels", (len(table_sources),), "int64"),
            ("targets", (len(table_sources),), "int64"),
            ("colors", (len(table_sources),), "int64"),
        ):
            path = feature_root / table_name / f"{field_name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"manifest-only:{table_name}:{field_name}".encode()
            path.write_bytes(payload)
            files.append(
                ArrayFileManifest.model_validate(
                    {
                        "field_name": field_name,
                        "relative_path": path.relative_to(feature_root).as_posix(),
                        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                        "shape": shape,
                        "dtype": dtype,
                    }
                )
            )
        tables.append(
            FeatureTableManifest(
                table_name=table_name,
                role=role,
                source_ids=table_sources,
                source_ids_digest=canonical_digest_value(table_sources),
                row_count=len(table_sources),
                feature_dimension=512,
                feature_dtype="float32",
                files=tuple(files),
            )
        )
    feature = CmnistFeatureCacheManifest(
        schema_version="grit.cmnist-features/v1",
        dataset_id="cmnist",
        source_manifest_digest=dataset_digest,
        pair_manifest_digest=pairs.canonical_digest(),
        encoder=encoder
        or EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=(
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
            ),
            model_name="ViT-B/32",
            weights_identity=(
                "sha256:40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
            ),
            preprocessing_identity=(
                "openai-clip-vit-b32-preprocess@"
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
            ),
            raw_output_dimension=512,
        ),
        normalization="none",
        feature_dimension=512,
        feature_dtype="float32",
        tables=tuple(tables),
    )
    dataset_path = root / "dataset-manifest.json"
    pair_path = root / "pair-manifest.json"
    feature_path = feature_root / "manifest.json"
    dataset_path.write_text(dataset.canonical_json(), encoding="utf-8")
    pair_path.write_text(pairs.canonical_json(), encoding="utf-8")
    feature_path.write_text(feature.canonical_json(), encoding="utf-8")
    return dataset_path, feature_path, pair_path


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds_cf"])
def test_production_plan_has_exact_deterministic_candidate_grid(dataset: str) -> None:
    first = build_search_plan(resolved_search_fixture(dataset))
    second = build_search_plan(resolved_search_fixture(dataset))
    assert first.candidates == second.candidates
    assert sum(item.method_id == "erm" for item in first.candidates) == 16
    assert sum(item.method_id == "grit" for item in first.candidates) == 400
    assert first.expected_run_counts.tuning == 1_248
    assert tuple(item.requested_rank for item in first.candidates[:16]) == (None,) * 16
    assert tuple(item.method_id for item in first.candidates) == (
        *("erm" for _ in range(16)),
        *("grit" for _ in range(400)),
    )


def test_yaml_mapping_order_does_not_change_config_or_candidates(
    tmp_path: Path,
) -> None:
    payload = _config().model_dump(mode="json")
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(dict(reversed(tuple(payload.items())))).replace(
            "1e-05", "0.00001"
        ).replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    first = load_production_search_config(first_path)
    second = load_production_search_config(second_path)
    assert first == second
    assert build_search_plan(resolved_search_fixture()).candidates == build_search_plan(
        resolved_search_fixture()
    ).candidates


def test_search_plan_rejects_duplicate_config_and_candidate_id_collision() -> None:
    plan = build_search_plan(resolved_search_fixture())
    payload = plan.model_dump(mode="json")
    candidates = payload["candidates"]
    duplicate_config = dict(candidates[1])
    duplicate_config["scientific_config_digest"] = candidates[0][
        "scientific_config_digest"
    ]
    candidates[1] = duplicate_config
    with pytest.raises(ValidationError, match="candidate grid"):
        SearchPlan.model_validate_json(json.dumps(payload))

    payload = plan.model_dump(mode="json")
    candidates = payload["candidates"]
    collision = dict(candidates[1])
    collision["candidate_id"] = candidates[0]["candidate_id"]
    candidates[1] = collision
    with pytest.raises(ValidationError, match="candidate grid"):
        SearchPlan.model_validate_json(json.dumps(payload))


def test_production_config_requires_exact_disjoint_seed_sets() -> None:
    payload = _config().model_dump(mode="json")
    payload["seeds"]["stages"]["confirmation"] = [101, 202]
    with pytest.raises(ValidationError, match="disjoint"):
        CmnistProductionSearchConfig.model_validate_json(json.dumps(payload))
    payload = _config().model_dump(mode="json")
    payload["seeds"]["stages"]["tuning"] = [101, 102]
    with pytest.raises(ValidationError, match="contain 3"):
        CmnistProductionSearchConfig.model_validate_json(json.dumps(payload))


def test_plan_round_trip_revalidates_canonical_boundary() -> None:
    plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    reparsed = SearchPlan.model_validate_json(plan.canonical_json())
    assert reparsed == plan
    assert reparsed.canonical_digest() == plan.canonical_digest()
    assert reparsed.code == CodeProvenance.model_validate(plan.code)
    assert reparsed.environment == EnvironmentProvenance.model_validate(
        plan.environment
    )
    payload = plan.model_dump(mode="json")
    payload["expected_run_counts"]["tuning"] = 1
    with pytest.raises(ValidationError, match="run counts"):
        SearchPlan.model_validate_json(json.dumps(payload))
    payload = plan.model_dump(mode="json")
    payload["output_schemas"][0]["schema_version"] = "forged/v1"
    with pytest.raises(ValidationError, match="schema inventory"):
        SearchPlan.model_validate_json(json.dumps(payload))


def test_planned_candidate_digests_materialize_into_dataset_configs() -> None:
    cmnist = build_search_plan(resolved_search_fixture("cmnist"))
    for candidate in cmnist.candidates:
        resolved = materialize_cmnist_candidate_config(
            cmnist, candidate, CmnistSelector.PRIMARY_ROBUST
        )
        assert resolved.scientific_config_digest() == (
            candidate.scientific_config_digest
        )

    waterbirds = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    erm = waterbirds.candidates[0]
    assert materialize_waterbirds_candidate_config(
        waterbirds, erm, None
    ).scientific_config_digest() == erm.scientific_config_digest
    grit = next(
        item
        for item in waterbirds.candidates
        if item.method_id == "grit" and item.requested_rank == 24
    )
    left = torch.zeros((25, 512), dtype=torch.float32)
    left[:, :25] = torch.eye(25, dtype=torch.float32)
    pair_digest = waterbirds.resolved_config.lineage.pair_manifest_digest
    if not pair_digest:
        raise AssertionError("Waterbirds test plan lacks pair lineage")
    projection = fit_linear_projection(
        left,
        torch.zeros_like(left),
        requested_rank=24,
        pair_manifest_digest=pair_digest,
        feature_cache_manifest_digest=(
            waterbirds.resolved_config.lineage.feature_cache_manifest_digest
        ),
    )
    assert materialize_waterbirds_candidate_config(
        waterbirds, grit, projection
    ).scientific_config_digest() == grit.scientific_config_digest


def test_waterbirds_scientific_candidate_identity_excludes_stage_seeds() -> None:
    plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    resolved = materialize_waterbirds_candidate_config(
        plan, plan.candidates[0], None
    )
    payload = resolved.model_dump(mode="json")
    payload["seed_sets"] = {
        "tuning": (901, 902, 903),
        "confirmation": (904, 905),
        "final": (906, 907, 908, 909, 910, 911, 912, 913, 914, 915),
    }
    changed = type(resolved).model_validate(payload)
    assert changed.scientific_config_digest() == resolved.scientific_config_digest()


def test_plan_only_verifies_official_manifest_lineage_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan-only operation crossed a runtime/final boundary")

    monkeypatch.setattr("grit.production_search._load_cmnist_cache", forbidden)
    monkeypatch.setattr("grit.production_search.train_linear_probe", forbidden)
    monkeypatch.setattr("grit.production_search.open_final_test", forbidden)
    dataset_path, feature_path, pair_path = (
        _write_manifest_only_cmnist_production(tmp_path / "prepared")
    )
    config = _config()
    payload = config.model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "planned").as_posix()
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    plan = plan_production_search(config_path)
    assert len(plan.candidates) == 416
    assert plan_production_search(config_path) == plan
    assert not (tmp_path / "planned" / "runs").exists()
    assert not tuple((tmp_path / "planned").rglob("*checkpoint*"))

    feature_payload = cast(
        dict[str, object],
        json.loads(feature_path.read_text(encoding="utf-8")),
    )
    encoder_payload = cast(dict[str, object], feature_payload["encoder"])
    encoder_payload["implementation"] = "grit.synthetic"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    rejected = dict(payload)
    rejected["output_root"] = (tmp_path / "rejected").as_posix()
    rejected_path = tmp_path / "fake-encoder.yaml"
    rejected_path.write_text(
        json.dumps(rejected)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official OpenAI CLIP"):
        plan_production_search(rejected_path)
    assert not (tmp_path / "rejected").exists()


def test_cmnist_plan_rejects_nonofficial_source_universe(tmp_path: Path) -> None:
    dataset_path, feature_path, pair_path = _write_manifest_only_cmnist_production(
        tmp_path / "prepared", invalid_official_train_universe=True
    )
    payload = _config().model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "planned").as_posix()
    config_path = tmp_path / "production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official source universes"):
        plan_production_search(config_path)


def test_plan_rejects_nonignored_output_inside_repository(tmp_path: Path) -> None:
    dataset_path, feature_path, pair_path = _write_manifest_only_cmnist_production(
        tmp_path / "prepared"
    )
    payload = _config().model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (
        Path.cwd() / "milestone6a-unignored-test-output"
    ).as_posix()
    config_path = tmp_path / "unsafe-output.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be Git-ignored"):
        plan_production_search(config_path)
    assert not (Path.cwd() / "milestone6a-unignored-test-output").exists()


def test_waterbirds_plan_only_accepts_verified_production_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plan-only operation crossed a runtime/final boundary")

    monkeypatch.setattr(
        "grit.production_waterbirds_search._load_cache", forbidden
    )
    monkeypatch.setattr(
        "grit.production_waterbirds_search.train_waterbirds_linear_probe",
        forbidden,
    )
    dataset_path, feature_path, pair_path = (
        write_manifest_only_waterbirds_production(tmp_path / "prepared-waterbirds")
    )
    config = _config("waterbirds_cf")
    payload = config.model_dump(mode="json")
    payload["artifacts"] = {
        "dataset_manifest": dataset_path.as_posix(),
        "feature_cache_manifest": feature_path.as_posix(),
        "oracle_pair_manifest": pair_path.as_posix(),
    }
    payload["output_root"] = (tmp_path / "waterbirds-plan").as_posix()
    config_path = tmp_path / "waterbirds-production.yaml"
    config_path.write_text(
        json.dumps(payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    plan = plan_production_search(config_path)
    assert plan.dataset == "waterbirds_cf"
    assert len(plan.candidates) == 416
    assert plan.resolved_config.lineage.adjusted_weight_spec_digest is not None
    assert not (tmp_path / "waterbirds-plan" / "runs").exists()

    original_feature = feature_path.read_text(encoding="utf-8")
    feature_payload = cast(dict[str, object], json.loads(original_feature))
    feature_payload["normalization"] = "l2"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload = dict(payload)
    mixed_payload["output_root"] = (tmp_path / "mixed-normalization").as_posix()
    mixed_path = tmp_path / "mixed-normalization.yaml"
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalization"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    feature_payload = cast(dict[str, object], json.loads(original_feature))
    feature_payload["dataset_manifest_digest"] = "sha256:another-dataset"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "mixed-dataset").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    feature_payload = cast(dict[str, object], json.loads(original_feature))
    encoder_payload = cast(dict[str, object], feature_payload["encoder"])
    encoder_payload["implementation"] = "grit.synthetic"
    feature_path.write_text(json.dumps(feature_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "fake-encoder").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official OpenAI CLIP"):
        plan_production_search(mixed_path)
    feature_path.write_text(original_feature, encoding="utf-8")

    original_pair = pair_path.read_text(encoding="utf-8")
    pair_payload = cast(dict[str, object], json.loads(original_pair))
    pair_payload["dataset_manifest_digest"] = "sha256:another-dataset"
    pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "mixed-pairs").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage"):
        plan_production_search(mixed_path)
    pair_path.write_text(original_pair, encoding="utf-8")

    dataset_payload = cast(
        dict[str, object], json.loads(dataset_path.read_text(encoding="utf-8"))
    )
    dataset_payload["non_reportable"] = True
    dataset_path.write_text(json.dumps(dataset_payload), encoding="utf-8")
    mixed_payload["output_root"] = (tmp_path / "non-reportable").as_posix()
    mixed_path.write_text(
        json.dumps(mixed_payload)
        .replace("1e-05", "0.00001")
        .replace("1e-12", "0.000000000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="production Waterbirds manifest"):
        plan_production_search(mixed_path)


def _cmnist_executor(
    plan: SearchPlan,
    calls: list[str],
    *,
    fail_once: list[bool] | None = None,
):
    def execute(task: SearchRunTask, _run_root: Path) -> CompletedStageRun:
        calls.append(task.task_id)
        if fail_once is not None and fail_once:
            fail_once.pop()
            raise RuntimeError("synthetic interruption")
        records: list[ValidationMetricRecord] = []
        for epoch, base in ((0, 0.60), (1, 0.70)):
            checkpoint_id = f"checkpoint:{task.task_id}:{epoch}"
            for split_name, offset in zip(
                ("val_e01", "val_e02", "val_e05"),
                (0.02, 0.01, 0.0),
                strict=True,
            ):
                records.append(
                    ValidationMetricRecord(
                        record_id=f"metric:{checkpoint_id}:{split_name}",
                        run_id=f"run:{task.task_id}",
                        candidate_id=task.candidate.candidate_id,
                        method_id=task.candidate.method_id,
                        scientific_config_digest=(
                            task.candidate.scientific_config_digest
                        ),
                        checkpoint_id=checkpoint_id,
                        epoch=epoch,
                        seed=task.seed,
                        value=base + offset,
                        sample_count=10,
                        metric_kind="validation",
                        seed_stage=task.stage,
                        split_name=split_name,
                        metric_name="accuracy",
                        projection_rank=task.candidate.requested_rank,
                    )
                )
        materialized = tuple(records)
        selectors = (
            CmnistSelector.PRIMARY_ROBUST,
            CmnistSelector.SECONDARY_SOURCE,
        )
        return CmnistCompletedStageRun(
            schema_version="grit.cmnist-search-stage-run/v1",
            dataset="cmnist",
            status="complete",
            task=task,
            lineage=plan.resolved_config.lineage,
            validation_metrics=materialized,
            checkpoint_decisions=tuple(
                select_checkpoint(materialized, selector) for selector in selectors
            ),
        )

    return execute


def test_completed_run_reuse_and_missing_run_continuation(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    tasks = tuple(
        make_search_task(plan, plan.candidates[0], SeedStage.TUNING, seed)
        for seed in plan.seeds.stages.tuning[:2]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    calls: list[str] = []
    first = scheduler.run_tasks(tasks[:1], _cmnist_executor(plan, calls))
    resumed = scheduler.run_tasks(tasks, _cmnist_executor(plan, calls))
    assert resumed[0] == first[0]
    assert calls == [tasks[0].task_id, tasks[1].task_id]
    status = scheduler.status(tasks)
    assert status.complete_task_ids == tuple(task.task_id for task in tasks)
    assert not status.missing_task_ids


def test_corrupt_or_incompatible_prior_result_is_rejected(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    _ = scheduler.run_tasks((task,), _cmnist_executor(plan, []))
    result_path = tmp_path / task.relative_directory / "result.json"
    result_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        scheduler.run_tasks((task,), _cmnist_executor(plan, []))

    other_plan = build_search_plan(resolved_search_fixture("waterbirds_cf"))
    with pytest.raises(ValueError, match="another plan"):
        LocalRunScheduler(tmp_path / "other", other_plan).run_tasks(
            (task,), _cmnist_executor(plan, [])
        )

    forged_payload = task.model_dump(mode="json")
    forged_payload["lineage"]["dataset_manifest_digest"] = "sha256:forged"
    forged_lineage = SearchLineage.model_validate(forged_payload["lineage"])
    identity_payload = {
        "plan_digest": task.plan_digest,
        "dataset": task.dataset,
        "lineage_digest": forged_lineage.canonical_digest(),
        "candidate_id": task.candidate.candidate_id,
        "scientific_config_digest": task.candidate.scientific_config_digest,
        "method_id": task.candidate.method_id,
        "stage": task.stage.value,
        "seed": task.seed,
        "selector": task.selector,
        "frozen_winner_digest": None,
    }
    forged_payload["task_id"] = (
        "task:" + SearchRunTask.canonical_identity_digest(identity_payload)
    )
    forged = SearchRunTask.model_validate(forged_payload)
    calls: list[str] = []
    with pytest.raises(ValueError, match="input lineage"):
        LocalRunScheduler(tmp_path / "forged", plan).run_tasks(
            (forged,), _cmnist_executor(plan, calls)
        )
    assert not calls


def test_interrupted_run_is_archived_then_reproduced(tmp_path: Path) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    resumed_scheduler = LocalRunScheduler(tmp_path / "resumed", plan)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        resumed_scheduler.run_tasks(
            (task,), _cmnist_executor(plan, [], fail_once=[True])
        )
    interrupted = resumed_scheduler.status((task,))
    assert interrupted.interrupted_task_ids == (task.task_id,)
    resumed = resumed_scheduler.run_tasks((task,), _cmnist_executor(plan, []))

    uninterrupted = LocalRunScheduler(tmp_path / "uninterrupted", plan).run_tasks(
        (task,), _cmnist_executor(plan, [])
    )
    assert resumed == uninterrupted
    assert not (
        tmp_path / "resumed" / task.relative_directory / "run-state.json"
    ).exists()
    staging_parent = (tmp_path / "resumed" / task.relative_directory).parent
    assert any("interrupted-1" in path.name for path in staging_parent.iterdir())


def test_interrupted_continuation_preserves_finalist_selection_summary(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    candidates = tuple(
        candidate for candidate in plan.candidates if candidate.method_id == "erm"
    )[:3]
    tasks = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in candidates
        for seed in plan.seeds.stages.tuning
    )
    resumed_scheduler = LocalRunScheduler(tmp_path / "resumed-selection", plan)
    _ = resumed_scheduler.run_tasks(
        tasks[:3], _cmnist_executor(plan, [])
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        resumed_scheduler.run_tasks(
            tasks[3:5], _cmnist_executor(plan, [], fail_once=[True])
        )
    resumed = resumed_scheduler.run_tasks(tasks, _cmnist_executor(plan, []))
    uninterrupted = LocalRunScheduler(
        tmp_path / "uninterrupted-selection", plan
    ).run_tasks(tasks, _cmnist_executor(plan, []))

    def selection_summary(
        runs: tuple[CompletedStageRun, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        records = tuple(
            metric
            for run in runs
            if isinstance(run, CmnistCompletedStageRun)
            for metric in run.validation_metrics
        )
        primary = make_tuning_finalists(
            records, CmnistSelector.PRIMARY_ROBUST, plan.seeds.stages
        )
        secondary = make_tuning_finalists(
            records, CmnistSelector.SECONDARY_SOURCE, plan.seeds.stages
        )
        return (
            tuple(item.candidate_id for item in primary.ordered_candidates),
            tuple(item.candidate_id for item in secondary.ordered_candidates),
        )

    assert resumed == uninterrupted
    assert selection_summary(resumed) == selection_summary(uninterrupted)


def test_completed_staging_result_is_promoted_without_reexecution(
    tmp_path: Path,
) -> None:
    plan = build_search_plan(resolved_search_fixture())
    task = make_search_task(
        plan, plan.candidates[0], SeedStage.TUNING, plan.seeds.stages.tuning[0]
    )
    scheduler = LocalRunScheduler(tmp_path, plan)
    base_executor = _cmnist_executor(plan, [])

    def finish_then_interrupt(
        staged_task: SearchRunTask,
        run_root: Path,
    ) -> CompletedStageRun:
        completed = base_executor(staged_task, run_root)
        (run_root / "result.json").write_text(
            completed.canonical_json() + "\n", encoding="utf-8"
        )
        raise RuntimeError("interrupted after durable completion")

    with pytest.raises(RuntimeError, match="durable completion"):
        scheduler.run_tasks((task,), finish_then_interrupt)

    calls: list[str] = []
    recovered = scheduler.run_tasks((task,), _cmnist_executor(plan, calls))
    assert len(recovered) == 1
    assert not calls
