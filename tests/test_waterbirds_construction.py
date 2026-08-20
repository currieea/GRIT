"""Hermetic tests for deterministic Waterbirds-CF construction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from grit.waterbirds import (
    BASE_ARTIFACT_NAME,
    ParsedWaterbirdsAssets,
    ProductionWaterbirdsProfile,
    WaterbirdsDatasetManifest,
    composite_groupdro,
    construct_waterbirds_cf,
    parse_places_backgrounds,
)
from tests.waterbirds_fixtures import make_waterbirds_fixture


def test_fixture_construction_replaces_minority_and_preserves_counts(
    tmp_path: Path,
) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )

    assert construction.manifest.non_reportable is True
    assert construction.manifest.counts.supervised_training == 10
    assert construction.manifest.counts.unpaired_majority == 4
    assert construction.manifest.counts.retained_majority_endpoints == 3
    assert construction.manifest.counts.generated_minority_endpoints == 3
    assert construction.manifest.counts.oracle_relationships == 3
    assert construction.manifest.counts.training_groups.as_tuple() == (4, 2, 1, 3)
    assert set(construction.manifest.replaced_released_record_ids) == {
        "waterbirds:released:5",
        "waterbirds:released:6",
        "waterbirds:released:7",
    }
    training_ids = {record.record_id for record in construction.training_records()}
    assert not training_ids.intersection(
        construction.manifest.replaced_released_record_ids
    )
    assert len(construction.manifest.relationships) == 3
    assert {
        relationship.orientation for relationship in construction.manifest.relationships
    } == {"land_minus_water"}


def test_construction_is_order_independent_and_seeded(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    first = construct_waterbirds_cf(fixture.assets, fixture.profile, tmp_path / "first")
    reordered_assets = replace(
        fixture.assets,
        released=fixture.assets.released.model_copy(
            update={"records": tuple(reversed(fixture.assets.released.records))}
        ),
        cub=fixture.assets.cub.model_copy(
            update={"records": tuple(reversed(fixture.assets.cub.records))}
        ),
        places=fixture.assets.places.model_copy(
            update={"records": tuple(reversed(fixture.assets.places.records))}
        ),
    )
    second = construct_waterbirds_cf(
        reordered_assets, fixture.profile, tmp_path / "second"
    )
    different_seed = construct_waterbirds_cf(
        fixture.assets,
        fixture.profile.model_copy(update={"construction_seed": 23}),
        tmp_path / "different-seed",
    )

    assert first.manifest.canonical_digest() == second.manifest.canonical_digest()
    assert (
        first.manifest.canonical_digest() != different_seed.manifest.canonical_digest()
    )
    assert tuple(
        item.majority_record_id for item in first.manifest.relationships
    ) != tuple(
        item.majority_record_id for item in different_seed.manifest.relationships
    )


def test_released_validation_and_test_bytes_are_unchanged(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    released_by_id = {
        record.record_id: record for record in fixture.assets.released.records
    }

    for record in (
        *construction.validation_records(),
        *construction.final_test_records(),
    ):
        source = fixture.assets.released_root / record.image_relative_path
        assert (
            construction.path_for(record.record_id).read_bytes() == source.read_bytes()
        )
        assert record.image_sha256 == released_by_id[record.record_id].image_sha256


def test_groupdro_compositor_preserves_foreground_and_recolors_background() -> None:
    source = np.zeros((4, 6, 3), dtype=np.uint8)
    source[:, :, :] = (200, 10, 20)
    mask = np.zeros((4, 6, 3), dtype=np.uint8)
    mask[:, :3, :] = 255
    background = np.zeros((8, 12, 3), dtype=np.uint8)
    background[:, :, :] = (1, 120, 240)

    result, geometry, foreground_digest = composite_groupdro(
        Image.fromarray(source),
        Image.fromarray(mask),
        Image.fromarray(background),
    )
    pixels = np.asarray(result)

    assert result.size == (6, 4)
    assert np.array_equal(pixels[:, :3, :], source[:, :3, :])
    assert np.array_equal(pixels[:, 3:, :], background[:4, :3, :])
    assert geometry.width == 6
    assert geometry.height == 4
    assert foreground_digest.startswith("sha256:")


def test_manifest_round_trip_revalidates_relationship_identity(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    construction = construct_waterbirds_cf(
        fixture.assets, fixture.profile, tmp_path / "output"
    )
    restored = WaterbirdsDatasetManifest.model_validate_json(
        construction.manifest.canonical_json()
    )
    assert restored == construction.manifest

    payload = json.loads(construction.manifest.canonical_json())
    first_relationship = payload["relationships"][0]
    first_relationship["land_record_id"] = "waterbirds:released:missing"
    with pytest.raises(ValidationError, match="outside training"):
        WaterbirdsDatasetManifest.model_validate_json(json.dumps(payload))


def test_production_profile_rejects_fixture_inventory(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    released = fixture.assets.released.model_copy(
        update={"artifact_name": BASE_ARTIFACT_NAME}
    )
    production_assets = ParsedWaterbirdsAssets(
        released_root=fixture.assets.released_root,
        cub_root=fixture.assets.cub_root,
        masks_root=fixture.assets.masks_root,
        places_root=fixture.assets.places_root,
        released=released,
        cub=fixture.assets.cub,
        places=fixture.assets.places,
    )
    with pytest.raises(ValueError, match="group counts"):
        construct_waterbirds_cf(
            production_assets,
            ProductionWaterbirdsProfile(
                kind="production",
                base_artifact_name=BASE_ARTIFACT_NAME,
                construction_seed=0,
            ),
            tmp_path / "production",
        )


def test_places_parser_rejects_missing_approved_category(tmp_path: Path) -> None:
    fixture = make_waterbirds_fixture(tmp_path / "assets")
    ocean = fixture.assets.places_root / "data_large" / "o" / "ocean"
    for path in ocean.iterdir():
        path.unlink()
    ocean.rmdir()

    with pytest.raises(FileNotFoundError, match="ocean"):
        parse_places_backgrounds(fixture.assets.places_root)
