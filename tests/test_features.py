"""Offline feature-cache and encoder-boundary tests for CMNIST."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from grit.cmnist import (
    CmnistConstruction,
    CmnistOraclePairSet,
    CmnistPartitionTargets,
    MnistPool,
    build_clean_oracle_pairs,
    construct_cmnist,
    pair_source_view,
)
from grit.data import ExampleIdentity, FinalTestSplitDescriptor, FinalTestView
from grit.features import (
    CmnistFeatureCacheManifest,
    DeterministicFakeEncoder,
    FeatureCacheValidationError,
    OfficialOpenAiClipEncoder,
    load_cmnist_feature_cache,
    prepare_cmnist_feature_cache,
)


def _construction_and_pairs(
    *,
    construction_seed: int = 4,
) -> tuple[CmnistConstruction, CmnistOraclePairSet]:
    train_count = 50
    test_count = 10
    train_indices = torch.arange(train_count, dtype=torch.int64)
    test_indices = torch.arange(test_count, dtype=torch.int64)
    train = MnistPool(
        official_split="train",
        source_indices=train_indices,
        images=(train_indices % 17).to(torch.float32).reshape(-1, 1, 1).div(16),
        digits=train_indices.remainder(10),
    )
    test = MnistPool(
        official_split="test",
        source_indices=test_indices,
        images=(test_indices % 9).to(torch.float32).reshape(-1, 1, 1).div(8),
        digits=test_indices.remainder(10),
    )
    targets = CmnistPartitionTargets(
        train_e01=20,
        train_e02=20,
        validation=10,
        test=10,
    )
    construction = construct_cmnist(
        train,
        test,
        construction_seed=construction_seed,
        targets=targets,
    )
    pairs = build_clean_oracle_pairs(
        pair_source_view(construction, train),
        pair_seed=8,
        pair_count=16,
    )
    return construction, pairs


def test_feature_preparation_rejects_pairs_from_another_construction(
    tmp_path: Path,
) -> None:
    construction, _ = _construction_and_pairs(construction_seed=4)
    _, other_pairs = _construction_and_pairs(construction_seed=5)
    output = tmp_path / "mixed"
    with pytest.raises(FeatureCacheValidationError, match="dataset manifest"):
        prepare_cmnist_feature_cache(
            construction,
            other_pairs,
            DeterministicFakeEncoder(),
            output,
            normalization="none",
        )
    assert not output.exists()


def test_feature_preparation_rejects_malformed_pairs_before_writing(
    tmp_path: Path,
) -> None:
    construction, pairs = _construction_and_pairs()
    first = pairs.records[0]
    changed_record = first.model_copy(update={"digit": (int(first.digit) + 1) % 10})
    malformed_records = replace(
        pairs,
        records=(changed_record, *pairs.records[1:]),
    )
    records_output = tmp_path / "malformed-records"
    with pytest.raises(FeatureCacheValidationError, match="records do not match"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_records,
            DeterministicFakeEncoder(),
            records_output,
            normalization="none",
        )
    assert not records_output.exists()

    malformed_metadata = replace(
        pairs,
        records=(changed_record, *pairs.records[1:]),
        manifest=pairs.manifest.model_copy(
            update={"records": (changed_record, *pairs.records[1:])}
        ),
    )
    metadata_output = tmp_path / "malformed-metadata"
    with pytest.raises(FeatureCacheValidationError, match="metadata does not match"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_metadata,
            DeterministicFakeEncoder(),
            metadata_output,
            normalization="none",
        )
    assert not metadata_output.exists()

    changed_red = pairs.left_red.clone()
    changed_red[0, 0, 0, 0] += 0.25
    malformed_endpoints = replace(pairs, left_red=changed_red)
    endpoints_output = tmp_path / "malformed-endpoints"
    with pytest.raises(FeatureCacheValidationError, match="recoloring invariant"):
        prepare_cmnist_feature_cache(
            construction,
            malformed_endpoints,
            DeterministicFakeEncoder(),
            endpoints_output,
            normalization="none",
        )
    assert not endpoints_output.exists()


def test_fake_feature_cache_is_canonical_validated_and_offline(tmp_path: Path) -> None:
    construction_object, pairs_object = _construction_and_pairs()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=13),
        first_root,
        normalization="none",
    )
    second = prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=13),
        second_root,
        normalization="none",
    )
    assert first == second
    assert (
        CmnistFeatureCacheManifest.model_validate_json(first.canonical_json()) == first
    )
    cache = load_cmnist_feature_cache(
        first_root,
        expected_source_manifest_digest=construction_object.manifest.canonical_digest(),
        expected_pair_manifest_digest=pairs_object.manifest.canonical_digest(),
        expected_normalization="none",
    )
    assert tuple(table.name for table in cache.training_tables()) == (
        "train_e01",
        "train_e02",
    )
    assert tuple(table.name for table in cache.validation_tables()) == (
        "val_e01",
        "val_e02",
        "val_e05",
    )
    assert cache.train_e01.features.shape == (20, 512)
    red, green = cache.pair_tables()
    assert red.source_ids == green.source_ids
    assert not torch.equal(red.features, green.features)


def test_l2_cache_is_explicit_and_cannot_mix_with_primary(tmp_path: Path) -> None:
    construction_object, pairs_object = _construction_and_pairs()
    root = tmp_path / "normalized"
    prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(seed=2),
        root,
        normalization="l2",
    )
    cache = load_cmnist_feature_cache(root, expected_normalization="l2")
    norms = torch.sqrt(cache.train_e01.features.square().sum(dim=1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0)
    with pytest.raises(FeatureCacheValidationError, match="normalization"):
        load_cmnist_feature_cache(root, expected_normalization="none")


def test_final_view_is_bound_to_exact_feature_cache_manifest(tmp_path: Path) -> None:
    construction, pairs = _construction_and_pairs()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=1),
        first_root,
        normalization="none",
    )
    prepare_cmnist_feature_cache(
        construction,
        pairs,
        DeterministicFakeEncoder(seed=2),
        second_root,
        normalization="none",
    )
    first = load_cmnist_feature_cache(first_root)
    second = load_cmnist_feature_cache(second_root)
    source_ids = first.manifest.tables[5].source_ids
    assert source_ids == second.manifest.tables[5].source_ids
    assert first.manifest.canonical_digest() != second.manifest.canonical_digest()
    view = FinalTestView(
        descriptor=FinalTestSplitDescriptor(
            dataset_id="cmnist",
            manifest_id=first.manifest.canonical_digest(),
            name="test_ood",
            role="final_test",
            source_partition_id="test_sources",
            view_id="test_ood",
        ),
        examples=tuple(
            ExampleIdentity(
                example_id=f"{source_id}:view:test_ood",
                source_id=source_id,
                view_id="test_ood",
            )
            for source_id in source_ids
        ),
        authorization_id="authorization:test",
        run_id="run:test",
        candidate_id="candidate:test",
        scientific_config_digest="sha256:test-config",
        checkpoint_id="checkpoint:test",
        epoch=0,
        method_id="erm",
        seed=301,
    )
    assert first.open_final_table(view).source_ids == source_ids
    with pytest.raises(FeatureCacheValidationError, match="feature cache manifest"):
        second.open_final_table(view)


def test_feature_cache_fails_helpfully_for_missing_or_mismatched_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(FeatureCacheValidationError, match="manifest is missing"):
        load_cmnist_feature_cache(tmp_path / "absent")

    construction_object, pairs_object = _construction_and_pairs()
    root = tmp_path / "cache"
    prepare_cmnist_feature_cache(
        construction_object,
        pairs_object,
        DeterministicFakeEncoder(),
        root,
        normalization="none",
    )
    with pytest.raises(FeatureCacheValidationError, match="source manifest"):
        load_cmnist_feature_cache(
            root,
            expected_source_manifest_digest="sha256:not-the-source",
        )
    (root / "train_e01" / "features.npy").unlink()
    with pytest.raises(FeatureCacheValidationError, match="file is missing"):
        load_cmnist_feature_cache(root)


def test_official_clip_adapter_requires_explicit_download_permission(
    tmp_path: Path,
) -> None:
    encoder = OfficialOpenAiClipEncoder(
        weights_root=tmp_path / "weights",
        allow_download=False,
        batch_size=1,
    )
    with pytest.raises(FileNotFoundError, match="--allow-download"):
        encoder.encode(torch.zeros((1, 3, 4, 4), dtype=torch.float32))
