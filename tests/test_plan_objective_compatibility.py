"""Historical objectives remain inspectable but cannot resume as corrected runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from grit.results import CodeProvenance
from grit.schemas import CmnistSelector
from grit.search.cmnist import materialize_cmnist_candidate_config
from grit.search.plan import (
    ResolvedProductionSearchConfig,
    SearchPlan,
    SearchSpaceConfig,
    _candidate_grid,  # pyright: ignore[reportPrivateUsage]
    build_search_plan,
    write_search_plan,
)
from grit.search.run import production_search_status
from grit.search.waterbirds import materialize_waterbirds_candidate_config
from tests.test_search_plan import resolved_search_fixture


@pytest.mark.parametrize("dataset", ["cmnist", "waterbirds_cf"])
def test_historical_matchdg_plan_is_readable_but_cannot_resume(
    dataset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    space = SearchSpaceConfig(
        methods=("matchdg",),
        learning_rates=(0.001, 0.002, 0.003),
        weight_decays=(0.0,),
        matchdg_latent_dims=(8,),
        matchdg_penalty_weights=(1.0,),
    )
    resolved = resolved_search_fixture(dataset)
    root = tmp_path / "old-output"
    authored = tmp_path / "old-config.yaml"
    config = resolved.config.model_copy(
        update={
            "search_space": space,
            "output_root": str(root),
        }
    )
    authored.write_text(config.canonical_json(), encoding="utf-8")
    resolved = resolved.model_copy(
        update={
            "config": config,
            "authored_config_digest": config.canonical_digest(),
            "authored_config_path": str(authored),
            "output_root": str(root),
        }
    )
    monkeypatch.setattr(
        "grit.search.plan._code_provenance",
        lambda: CodeProvenance(
            git_revision="1" * 40,
            git_dirty=False,
        ),
    )
    current = build_search_plan(resolved)
    legacy_candidates = _candidate_grid(resolved, legacy_matchdg=True)
    assert {item.scientific_config_digest for item in legacy_candidates}.isdisjoint(
        item.scientific_config_digest for item in current.candidates
    )
    historical = SearchPlan.model_validate(
        current.model_copy(
            update={
                "candidates": legacy_candidates,
            }
        )
    )
    root.mkdir()
    (root / "authored-config.yaml").write_text(
        config.canonical_json(), encoding="utf-8"
    )
    (root / "resolved-config.json").write_text(
        resolved.canonical_json(), encoding="utf-8"
    )
    (root / "search-plan.json").write_text(
        historical.canonical_json(), encoding="utf-8"
    )
    assert (
        SearchPlan.model_validate_json(
            (root / "search-plan.json").read_text(encoding="utf-8")
        )
        == historical
    )
    status = production_search_status(authored)
    assert status.phase == "tuning"
    assert status.tuning_complete == 0
    assert status.plan_digest == historical.canonical_digest()

    def resolve_fixture(
        *_args: object, **_kwargs: object
    ) -> ResolvedProductionSearchConfig:
        return resolved

    monkeypatch.setattr(
        "grit.search.plan.resolve_production_search_config",
        resolve_fixture,
    )
    with pytest.raises(ValueError, match="existing search plan is incompatible"):
        write_search_plan(config, config_path=authored)
    assert (root / "search-plan.json").read_text(encoding="utf-8") == (
        historical.canonical_json()
    )

    # Even callers bypassing the CLI cannot materialize an old objective as a new
    # training task under its historical scientific identity.
    with pytest.raises(ValueError, match="candidate digest cannot be materialized"):
        if dataset == "cmnist":
            materialize_cmnist_candidate_config(
                historical, historical.candidates[0], CmnistSelector.PRIMARY_ROBUST
            )
        else:
            materialize_waterbirds_candidate_config(
                historical, historical.candidates[0], None
            )
