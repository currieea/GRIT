"""Scientific construction, cache, training, and planning checks for RotatedMNIST."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from grit.config import (
    OPENAI_CLIP_PREPROCESSING_ID,
    OPENAI_CLIP_REVISION,
    OPENAI_CLIP_WEIGHTS_IDENTITY,
    LinearProbeTrainingConfig,
)
from grit.data.cmnist import MnistPool
from grit.data.rotated_mnist import (
    ROTATED_MNIST_ENVIRONMENT_SPECS,
    DigitCount,
    RotatedMnistConstruction,
    RotatedMnistDatasetManifest,
    RotatedMnistEnvironmentManifest,
    RotatedMnistOraclePairManifest,
    RotatedMnistOraclePairRecord,
    RotatedMnistPartitionManifest,
    RotatedMnistPartitionRecord,
    RotatedMnistPartitionTargets,
    build_rotated_mnist_oracle_pairs,
    construct_rotated_mnist,
)
from grit.features.cmnist import (
    DeterministicFakeEncoder,
    EncoderIdentity,
    FeatureExtractionRuntime,
)
from grit.features.rotated_mnist import (
    ArrayField,
    RotatedMnistArrayFileManifest,
    RotatedMnistFeatureCacheManifest,
    RotatedMnistFeatureTableManifest,
    RotatedMnistTuningFeatureCache,
    TableRole,
    load_rotated_mnist_feature_cache,
    load_rotated_mnist_tuning_feature_cache,
    prepare_rotated_mnist_feature_cache,
)
from grit.methods.training import OrdinaryLinearProbeMethod, train_linear_probe
from grit.schemas import CmnistSelector, SeedStage, canonical_digest_value
from grit.search.outputs import RotatedMnistProductionSummary
from grit.search.plan import (
    RotatedMnistProductionSearchConfig,
    SearchPlan,
    SearchSpaceConfig,
    build_search_plan,
    load_production_search_config,
    resolve_production_search_config,
    write_search_plan,
)
from grit.search.rotated_mnist import (
    materialize_rotated_mnist_candidate_config,
    rotated_mnist_status_from_plan,
    run_rotated_mnist_search,
)
from grit.search.run import complete_outputs_valid
from grit.selection.cmnist import select_checkpoint


def _pool(split: str, count: int, image_size: int = 8) -> MnistPool:
    generator = torch.Generator().manual_seed(41 if split == "train" else 43)
    return MnistPool(
        official_split="train" if split == "train" else "test",
        source_indices=torch.arange(count, dtype=torch.int64),
        images=torch.rand((count, image_size, image_size), generator=generator),
        digits=torch.arange(count, dtype=torch.int64) % 10,
    )


def _small_construction() -> tuple[MnistPool, MnistPool, RotatedMnistConstruction]:
    train = _pool("train", 60)
    test = _pool("test", 20)
    construction = construct_rotated_mnist(
        train,
        test,
        construction_seed=17,
        targets=RotatedMnistPartitionTargets(
            train_r0=25, train_r45=25, validation=10, test=20
        ),
    )
    return train, test, construction


def test_rotated_mnist_partitions_before_rendering_and_exact_pairs() -> None:
    train, _, construction = _small_construction()
    train_r0 = set(construction.train_r0.source_ids)
    train_r45 = set(construction.train_r45.source_ids)
    validation = set(construction.val_r0.source_ids)
    final = set(construction.preparation_tables()[5].source_ids)
    assert not train_r0 & train_r45
    assert not (train_r0 | train_r45) & validation
    assert all(source_id.startswith("mnist:train:") for source_id in validation)
    assert all(source_id.startswith("mnist:test:") for source_id in final)
    assert construction.val_r0.source_ids == construction.val_r45.source_ids
    assert construction.val_r0.source_ids == construction.val_r60.source_ids

    pairs = build_rotated_mnist_oracle_pairs(
        construction, train, pair_seed=23, pair_count=16
    )
    pair_sources = tuple(record.source_id for record in pairs.records)
    assert pairs.left_r0.source_ids == pair_sources
    assert pairs.right_r45.source_ids == pair_sources
    assert set(pair_sources) <= train_r0 | train_r45
    assert not set(pair_sources) & (validation | final)
    assert torch.equal(pairs.left_r0.targets, pairs.right_r45.targets)
    assert all(
        record.left_angle_degrees == 0 and record.right_angle_degrees == 45
        for record in pairs.records
    )


def test_rotated_mnist_construction_is_order_independent_and_deterministic() -> None:
    train = _pool("train", 60)
    test = _pool("test", 20)
    permutation = torch.randperm(60, generator=torch.Generator().manual_seed(5))
    shuffled = MnistPool(
        official_split="train",
        source_indices=train.source_indices[permutation],
        images=train.images[permutation],
        digits=train.digits[permutation],
    )
    targets = RotatedMnistPartitionTargets(
        train_r0=25, train_r45=25, validation=10, test=20
    )
    first = construct_rotated_mnist(train, test, construction_seed=17, targets=targets)
    second = construct_rotated_mnist(
        shuffled, test, construction_seed=17, targets=targets
    )
    assert first.manifest == second.manifest
    assert torch.equal(first.train_r0.images, second.train_r0.images)
    assert torch.equal(first.val_r60.images, second.val_r60.images)


def test_rotated_mnist_cache_hides_final_table_and_trains_ten_classes(
    tmp_path: Path,
) -> None:
    train, _, construction = _small_construction()
    pairs = build_rotated_mnist_oracle_pairs(
        construction, train, pair_seed=23, pair_count=16
    )
    root = tmp_path / "feature-cache"
    feature_manifest = prepare_rotated_mnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=31),
        root,
        normalization="none",
    )
    tuning = load_rotated_mnist_tuning_feature_cache(
        root,
        expected_source_manifest_digest=construction.manifest.canonical_digest(),
        expected_pair_manifest_digest=pairs.manifest.canonical_digest(),
        expected_normalization="none",
    )
    assert not hasattr(tuning, "_test_r90")
    full = load_rotated_mnist_feature_cache(root)
    assert full.manifest == feature_manifest
    trained = train_linear_probe(
        tuning.training_tables(),
        tuning.validation_tables(),
        LinearProbeTrainingConfig(
            optimizer="adam",
            batch_size=16,
            learning_rate=0.001,
            weight_decay=0.0,
            max_epochs=1,
        ),
        run_id="run:rotated-test",
        candidate_id="candidate:rotated-test",
        scientific_config_digest="sha256:rotated-test",
        seed_stage=SeedStage.TUNING,
        seed=101,
        method=OrdinaryLinearProbeMethod(
            method_id="erm", projection=None, projection_rank=None
        ),
        num_classes=10,
    )
    assert trained.algorithm.capture_inference_state().weight.shape == (10, 512)
    decision = select_checkpoint(
        trained.validation_metrics, CmnistSelector.PRIMARY_ROBUST
    )
    assert len(decision.contributing_record_ids) == 3
    assert {item.split_name for item in trained.validation_metrics} == {
        "val_r0",
        "val_r45",
        "val_r60",
    }


def test_rotated_mnist_checked_config_and_production_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "rotated-mnist-none"
    _write_manifest_only_production(artifact_root)
    monkeypatch.setenv("PROJECT_SCRATCH", tmp_path.as_posix())
    config_path = Path(__file__).resolve().parents[1] / (
        "configs/rotated_mnist/production-search.yaml"
    )
    config = load_production_search_config(config_path)
    assert isinstance(config, RotatedMnistProductionSearchConfig)
    assert config.search_space.methods == ("erm", "grit")
    assert config.search_space.projection_ranks == tuple(range(2, 25))
    resolved = resolve_production_search_config(config, config_path=config_path)
    plan = build_search_plan(resolved)
    assert plan.dataset == "rotated_mnist"
    assert len(plan.candidates) == 384
    assert plan.expected_run_counts.tuning == 1_152
    assert plan.expected_run_counts.final == 40
    for candidate in (plan.candidates[0], plan.candidates[-1]):
        materialized = materialize_rotated_mnist_candidate_config(
            plan, candidate, CmnistSelector.PRIMARY_ROBUST
        )
        assert materialized.scientific_config_digest() == (
            candidate.scientific_config_digest
        )


def test_rotated_mnist_compact_production_lifecycle_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "rotated-mnist-none"
    _write_manifest_only_production(artifact_root)
    monkeypatch.setenv("PROJECT_SCRATCH", tmp_path.as_posix())
    checked_config_path = Path(__file__).resolve().parents[1] / (
        "configs/rotated_mnist/production-search.yaml"
    )
    checked = load_production_search_config(checked_config_path)
    assert isinstance(checked, RotatedMnistProductionSearchConfig)
    payload = checked.model_dump(mode="python")
    payload.update(
        {
            "search_space": SearchSpaceConfig(
                methods=("erm", "grit"),
                learning_rates=(0.0001, 0.001, 0.003),
                weight_decays=(0.0,),
                projection_ranks=(2,),
            ),
            "max_epochs": 1,
            "output_root": (tmp_path / "outputs" / "compact-rotated").as_posix(),
        }
    )
    config = RotatedMnistProductionSearchConfig.model_validate(payload)
    config_path = tmp_path / "compact-rotated.yaml"
    config_path.write_text(config.canonical_json() + "\n", encoding="utf-8")
    plan = write_search_plan(config, config_path=config_path)
    assert len(plan.candidates) == 6

    train, _, construction = _small_construction()
    pairs = build_rotated_mnist_oracle_pairs(
        construction, train, pair_seed=23, pair_count=16
    )
    cache_root = tmp_path / "small-feature-cache"
    prepare_rotated_mnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=31),
        cache_root,
        normalization="none",
    )
    cache = load_rotated_mnist_feature_cache(cache_root)

    def load_small_cache(
        _plan: SearchPlan, *, tuning_only: bool
    ) -> RotatedMnistTuningFeatureCache:
        del _plan, tuning_only
        return cache

    monkeypatch.setattr(
        "grit.search.rotated_mnist._load_cache",
        load_small_cache,
    )

    summary = run_rotated_mnist_search(plan)
    assert isinstance(summary, RotatedMnistProductionSummary)
    assert tuple(item.method_id for item in summary.methods) == (
        "erm",
        "erm",
        "grit",
        "grit",
    )
    assert len(summary.paired_selectors) == 2
    assert complete_outputs_valid(plan)
    status = rotated_mnist_status_from_plan(plan)
    assert status.phase == "complete"
    assert status.tuning_complete == 18
    assert status.final_complete == 40
    assert (
        Path(plan.resolved_config.output_root) / "projections/grit-rank-2.json"
    ).is_file()


def _write_manifest_only_production(root: Path) -> None:
    root.mkdir(parents=True)
    memberships = (
        tuple(range(0, 25_000)),
        tuple(range(25_000, 50_000)),
        tuple(range(50_000, 60_000)),
        tuple(range(10_000)),
    )
    names = (
        "train_r0_sources",
        "train_r45_sources",
        "validation_sources",
        "test_sources",
    )
    splits = ("train", "train", "train", "test")
    partition_records = tuple(
        RotatedMnistPartitionRecord(
            name=name,
            official_split=split,
            source_indices=membership,
            digit_counts=tuple(
                DigitCount(
                    digit=digit,
                    count=sum(index % 10 == digit for index in membership),
                )
                for digit in range(10)
            ),
            membership_digest=canonical_digest_value(
                {
                    "name": name,
                    "official_split": split,
                    "source_indices": membership,
                }
            ),
        )
        for name, split, membership in zip(names, splits, memberships, strict=True)
    )
    partitions = RotatedMnistPartitionManifest(
        schema_version="grit.rotated-mnist-partitions/v1",
        dataset_id="rotated_mnist",
        construction_method_id="rotated-mnist-stratified-hash-v1",
        construction_seed=1729,
        targets=RotatedMnistPartitionTargets(
            train_r0=25_000,
            train_r45=25_000,
            validation=10_000,
            test=10_000,
        ),
        partitions=partition_records,
    )
    partition_by_name = {item.name: item for item in partition_records}
    environment_manifests = tuple(
        RotatedMnistEnvironmentManifest(
            name=spec.name,
            role=spec.role,
            source_partition_id=spec.source_partition_id,
            angle_degrees=spec.angle_degrees,
            count=len(partition_by_name[spec.source_partition_id].source_indices),
            source_membership_digest=canonical_digest_value(
                tuple(
                    "mnist:"
                    f"{partition_by_name[spec.source_partition_id].official_split}:"
                    f"{index}"
                    for index in partition_by_name[
                        spec.source_partition_id
                    ].source_indices
                )
            ),
            record_digest=f"sha256:environment:{spec.name}",
        )
        for spec in ROTATED_MNIST_ENVIRONMENT_SPECS
    )
    dataset = RotatedMnistDatasetManifest(
        schema_version="grit.rotated-mnist-dataset/v1",
        dataset_id="rotated_mnist",
        partition_manifest=partitions,
        partition_manifest_digest=partitions.canonical_digest(),
        official_train_pool_digest="sha256:official-train",
        official_test_pool_digest="sha256:official-test",
        rendering_method_id="rotated-mnist-bilinear-rgb-v1",
        construction_seed=1729,
        environments=environment_manifests,
    )
    dataset_digest = dataset.canonical_digest()
    pair_records = tuple(
        RotatedMnistOraclePairRecord(
            pair_id=canonical_digest_value(
                {
                    "dataset_manifest_digest": dataset_digest,
                    "method": "rotated-mnist-exact-source-oracle-pairs-v1",
                    "pair_seed": 2718,
                    "source_id": f"mnist:train:{index}",
                    "orientation": "rotation_0_minus_45",
                }
            ),
            source_id=f"mnist:train:{index}",
            source_partition_id="train_r0_sources",
            official_source_index=index,
            digit=index % 10,
            left_angle_degrees=0,
            right_angle_degrees=45,
        )
        for index in range(256)
    )
    pairs = RotatedMnistOraclePairManifest(
        schema_version="grit.rotated-mnist-oracle-pairs/v1",
        dataset_manifest_digest=dataset_digest,
        construction_method_id="rotated-mnist-exact-source-oracle-pairs-v1",
        pair_seed=2718,
        requested_count=256,
        realized_count=256,
        source_partition_ids=("train_r0_sources", "train_r45_sources"),
        orientation="rotation_0_minus_45",
        records=pair_records,
        membership_digest=canonical_digest_value(
            tuple(item.source_id for item in pair_records)
        ),
    )
    table_names = tuple(spec.name for spec in ROTATED_MNIST_ENVIRONMENT_SPECS) + (
        "oracle_pair_r0",
        "oracle_pair_r45",
    )
    roles: tuple[TableRole, ...] = (
        "training",
        "training",
        "validation",
        "validation",
        "validation",
        "final_test",
        "pair_projection",
        "pair_projection",
    )
    sources = tuple(
        tuple(
            f"mnist:{partition_by_name[spec.source_partition_id].official_split}:{index}"
            for index in partition_by_name[spec.source_partition_id].source_indices
        )
        for spec in ROTATED_MNIST_ENVIRONMENT_SPECS
    ) + (
        tuple(item.source_id for item in pair_records),
        tuple(item.source_id for item in pair_records),
    )
    feature_root = root / "feature-cache"
    tables: list[RotatedMnistFeatureTableManifest] = []
    for table_name, role, source_ids in zip(table_names, roles, sources, strict=True):
        table_root = feature_root / table_name
        table_root.mkdir(parents=True)
        files: list[RotatedMnistArrayFileManifest] = []
        arrays: tuple[tuple[ArrayField, tuple[int, ...], str], ...] = (
            ("features", (len(source_ids), 512), "float32"),
            ("targets", (len(source_ids),), "int64"),
            ("angles", (len(source_ids),), "int64"),
        )
        for field_name, shape, dtype in arrays:
            path = table_root / f"{field_name}.npy"
            payload = f"manifest-only:{table_name}:{field_name}".encode()
            path.write_bytes(payload)
            files.append(
                RotatedMnistArrayFileManifest(
                    field_name=field_name,
                    relative_path=path.relative_to(feature_root).as_posix(),
                    sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    shape=shape,
                    dtype=dtype,
                )
            )
        tables.append(
            RotatedMnistFeatureTableManifest(
                table_name=table_name,
                role=role,
                source_ids=source_ids,
                source_ids_digest=canonical_digest_value(source_ids),
                row_count=len(source_ids),
                feature_dimension=512,
                feature_dtype="float32",
                files=tuple(files),
            )
        )
    feature = RotatedMnistFeatureCacheManifest(
        schema_version="grit.rotated-mnist-features/v1",
        dataset_id="rotated_mnist",
        source_manifest_digest=dataset_digest,
        pair_manifest_digest=pairs.canonical_digest(),
        encoder=EncoderIdentity(
            implementation="openai/CLIP",
            implementation_revision=OPENAI_CLIP_REVISION,
            model_name="ViT-B/32",
            weights_identity=OPENAI_CLIP_WEIGHTS_IDENTITY,
            preprocessing_identity=OPENAI_CLIP_PREPROCESSING_ID,
            raw_output_dimension=512,
        ),
        extraction_runtime=FeatureExtractionRuntime(
            requested_device="cpu",
            resolved_device="cpu",
            computation_dtype="torch.float32",
            deterministic_algorithms=True,
            tf32_enabled=False,
            mixed_precision=False,
            batch_size=256,
            torch_version=str(torch.__version__),
            cuda_runtime_version=None,
            device_name="cpu",
            compute_capability=None,
        ),
        normalization="none",
        feature_dimension=512,
        feature_dtype="float32",
        tables=tuple(tables),
    )
    (root / "dataset-manifest.json").write_text(
        dataset.canonical_json(), encoding="utf-8"
    )
    (root / "pair-manifest.json").write_text(pairs.canonical_json(), encoding="utf-8")
    (feature_root / "manifest.json").write_text(
        feature.canonical_json(), encoding="utf-8"
    )
