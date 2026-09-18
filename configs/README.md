# Experiment configurations

- `cmnist/production-search.yaml`, `rotated_mnist/production-search.yaml`, and
  `waterbirds/production-search.yaml`: the primary unnormalized studies. Paths use
  `${PROJECT_SCRATCH}` so they run unmodified after
  `scratch-project`. Copy and edit for other machines or for the L2 sensitivity (which
  needs `--normalization l2` at prepare time, a separate artifact root, and
  `experiment_variant: l2_normalized_sensitivity`).
- `cmnist/{groupdro,rex,irm,fish,lisa,swad,matchdg}-search.yaml` and
  `waterbirds/{groupdro,rex,irm,fish,lisa,swad,matchdg}-search.yaml`: the baselines,
  each in its own output tree so adding one never reruns another. The Waterbirds files
  need the prepared `waterbirds-none` artifacts from `scripts/prepare_waterbirds.py`.
- `cmnist/{sd,fishr,rdm}-search.yaml` and `waterbirds/{sd,fishr,rdm}-search.yaml`: the
  additional baselines, with 80, 64, and 64 candidates respectively and ordinary
  validation selection. They reuse the already prepared `cmnist-none` and
  `waterbirds-none` features; no additional data preparation is needed. SD uses ERM
  sampling and no background labels; Fishr and RDM use the balanced sampler over the
  two training environments (`train_e01`/`train_e02` for CMNIST, land/water backgrounds
  for Waterbirds). Fishr and RDM use a 1,500-update warm-up; a shortened run must leave
  at least one update after warm-up or it is rejected when the method binds to the
  cache. The experimenter reports that real-data Waterbirds runs behaved as expected;
  verify Fishr/RDM pilot warm-up coverage before freezing the initial grids. CMNIST
  real-data validation pilots remain outstanding. Use each config's canonical outputs
  to establish full-search completion and final metrics.
- `cmnist/*-test-oracle.yaml` and `waterbirds/*-test-oracle.yaml`: the same grids and
  seeds selected on the test split (`selectors: [test_oracle]`), the paper's "oracle
  validation" columns. Each writes a separate `*-test-oracle` output tree; every result
  and summary in it is labeled `test_oracle` and must be reported under that heading,
  never beside the ordinary numbers as if validation-selected. SD, Fishr, and RDM have
  no checked-in test-oracle configurations on either dataset.
- `cmnist/smoke.yaml` and `waterbirds/smoke.yaml`: hermetic end-to-end checks with a
  fake encoder, run by `scripts/smoke.py`. Never report their numbers.

`reportable: true` in a result means real data and the pinned CLIP encoder were used, as
opposed to the fake-encoder smoke path. It does not mean the run matched the paper grid;
judge that from the plan file next to the results.

The checked-in grids, seeds, epochs, and pair counts are the primary protocol from
`docs/experiments/`. Any of them can be changed in a copied config for a sensitivity or a
quick check; the plan and every result record the values that actually ran. A CMNIST or RotatedMNIST config's
`pair_count` may be any prefix of the prepared pair bank (pairs are stored in seeded
order), so bank 512 pairs once with `--pair-count 512` and run 32/64/128/256/512 from
five configs. Waterbirds requires its full canonical 240-pair bank.

## Objective/intervention experiments

The original three-variant configs below remain unchanged. For the expanded matrix
use the representation configs described next.

`{cmnist,waterbirds}/{erm,rex,irm,fishr}-interventions-search.yaml` each run one
objective's vanilla, GRIT and prediction-consistency variants, with independent winners.
`rex` is V-REx; `grit` retains its original meaning of ERM plus GRIT. New method IDs
are `erm_consistency`, `rex_grit`, `rex_consistency`, `irm_grit`, `irm_consistency`,
`fishr_grit`, and `fishr_consistency`. Resolved candidates store a composed algorithm
with `base_objective`, `pair_intervention`, and a separate `consistency_weight`.
Existing authored configs and commands still work.

The provisional matrix keeps the existing 16 optimizer settings and base-penalty grids,
but searches ranks `[2, 8, 16, 24]` and consistency strengths `[0.01, 0.1, 1, 10]`.
These ranges require pilot review before a full search; they are not a frozen production
grid. Do not infer equal tuning compute across interventions.

| Objective row, per dataset | Vanilla | GRIT | Consistency | Tuning tasks (3 seeds) |
| --- | ---: | ---: | ---: | ---: |
| ERM | 16 | 64 | 64 | 432 |
| V-REx | 64 | 256 | 256 | 1,728 |
| IRMv1 | 64 | 256 | 256 | 1,728 |
| Fishr | 64 | 256 | 256 | 1,728 |

Each row additionally confirms top three per combination with two seeds: 18–36 tasks
for CMNIST's two-selector union, 18 for Waterbirds. Final counts are 60 per CMNIST row
and 30 per Waterbirds row, using ten explicit shared seeds. Per dataset the full
provisional matrix is 1,872 candidates and 5,616 tuning tasks. Using all 23 legacy
ranks instead would yield 5,824 candidates, so rank coverage is deliberately bounded
until pilot review. Pair counts remain 256 CMNIST / 240 Waterbirds; ranks and strengths
of zero may be used in copied sanity-control configs but are absent here.

Run training/validation pilots only, in dependency order (each `--pilot` runs one full
training schedule for each of the three variants and never opens ordinary test):

```bash
for dataset in cmnist waterbirds; do
  for objective in erm rex irm fishr; do
    uv run scripts/run_search.py configs/$dataset/$objective-interventions-search.yaml --pilot
    uv run scripts/search_status.py configs/$dataset/$objective-interventions-search.yaml
  done
done
```

Use `scratch-project` on the servers or set `PROJECT_SCRATCH` to the prepared artifact
root's parent. Defaults expect `artifacts/cmnist-none` and `artifacts/waterbirds-none`.
Fishr pilots retain 1,500 warm-up updates and the full 40/100 epochs; both datasets have
updates after warm-up: the canonical environment sizes imply 7,840 total updates
for CMNIST and 2,800 for Waterbirds (6,340 and 1,300 active Fishr-penalty updates).
Do not shorten these pilots below activation. Review validation
scores, finite objectives, selected epochs, and runtime before expanding the grid.
To preview a verified plan without training, add `--dry-run`. Full searches use the
same command without `--pilot`, only after review. Summaries include absolute metrics
and all three within-objective contrasts, preserving CMNIST selector and diagnostic
track separation. W&B remains optional and mirrors only already-computed measurements.

### Expanded representation matrix

Use `{cmnist,waterbirds}/{erm,rex,irm,fishr}-representation-interventions-search.yaml`.
These retain the previous three variants and add `<objective>_representation_consistency`
and auxiliary `<objective>_two_layer`. `erm`, `rex`, `irm`, and `fishr` select the base;
`grit` remains ERM plus projection. Historical `*_consistency` IDs and
`consistency_weights` still mean prediction consistency.

The independent `representation_consistency_weights: [0.01, 0.1, 1.0, 10.0]` grid
is crossed with optimizer and base-objective settings. `representation_latent_dims: [32]`
fixes the initial width provisionally for both representation and control. The control
always sets representation strength to zero and receives no pair differences, pair
access or candidate pair lineage. A representation-consistency candidate with strength
zero still identifies its pair bank; its updates match the control at identical settings.
Representation candidates use `representation_consistency_weight` and
`representation_latent_dim`; resolved composed algorithms record `latent_dim`, separate
base/representation coefficients, and `objective_version: factorized-pairs/v1`.
Prediction/GRIT/standalone candidate identities are preserved. MatchDG keeps its separate
`[8, 16, 32]` width search and current corrected objective.

| Objective row, per dataset | Vanilla | GRIT | Prediction | Representation | Two-layer control | Tuning tasks (3 seeds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ERM | 16 | 64 | 64 | 64 | 16 | 672 |
| V-REx | 64 | 256 | 256 | 256 | 64 | 2,688 |
| IRMv1 | 64 | 256 | 256 | 256 | 64 | 2,688 |
| Fishr | 64 | 256 | 256 | 256 | 64 | 2,688 |

Per dataset this is 2,912 candidates and 8,736 tuning tasks. Each row additionally
confirms top three per variant with two seeds: 30–60 tasks for CMNIST's two-selector
union, 30 for Waterbirds. Final counts are 100 per CMNIST row and 50 per Waterbirds row.
Every variant has its own winner. Summaries retain all previous comparisons and add
representation minus vanilla, GRIT minus representation, representation minus prediction,
and representation minus two-layer control, with explicit seed joins and compatible
lineage/pair budgets.

All expanded configs use fresh `*-interventions-representation-v1` output roots. Do not
point an expanded config at an existing three-variant or standalone output directory.
Historical results and existing configs require no migration. New factorized selected
checkpoints store both layers plus a collapsed inference map. A fresh direct
`LinearProbeAlgorithm` can restore their predictions; factorized optimization requires
both layers. These are selected checkpoints, not exact optimizer/EMA resume snapshots.
Search task restart behavior is unchanged.

On a prepared server, run these bounded training/validation pilots in order:

```bash
scratch-project
for objective in erm rex irm fishr; do
  for dataset in cmnist waterbirds; do
    config=configs/$dataset/$objective-representation-interventions-search.yaml
    uv run scripts/run_search.py "$config" --pilot
    uv run scripts/search_status.py "$config"
  done
done
```

Each pilot executes one full schedule for each of five variants on the first tuning
seed; it never evaluates ordinary test. Keep the full 40 CMNIST / 100 Waterbirds epochs:
Fishr runs 7,840 / 2,800 updates and crosses its 1,500-update warm-up. If artifacts are
absent, prepare them first using the README commands; Waterbirds preparation needs the
released images, CUB masks and Places sources. No additional feature extraction is needed
for this extension. `--only fishr_representation_consistency --pilot`, for example,
restricts execution to the selected variant through the existing CLI.

Every factorized task saves `representation-diagnostics.json` with epoch-keyed layer
weight norms and (for pair-consuming variants) unweighted representation and logit pair
discrepancies. These values also appear in W&B when enabled; the selected checkpoint
manifest retains its diagnostics. Inspect validation scores, finite objectives, runtime,
and trajectories after warm-up. Shrinking A with growing B can reduce the representation
penalty without improving logit invariance. Coefficients with equal numbers need not have
equal effects across prediction, representation, or rescaled base objectives.

If the execution pilots leave scale unclear, this exact bounded follow-up evaluates all
four representation strengths at the pilot's optimizer/base settings and first tuning
seed, reusing the already completed pilot. It remains validation-only:

```bash
uv run python - <<'PY'
from pathlib import Path
from grit.search.run import (
    ProductionExecutionLimits, plan_production_search, run_production_search,
)
for objective in ("erm", "rex", "irm", "fishr"):
    for dataset in ("cmnist", "waterbirds"):
        path = Path(f"configs/{dataset}/{objective}-representation-interventions-search.yaml")
        plan = plan_production_search(path)
        method = f"{objective}_representation_consistency"
        candidates = [c for c in plan.candidates if c.method_id == method]
        first = candidates[0]
        fields = ("learning_rate", "weight_decay", "penalty_weight", "representation_latent_dim")
        chosen = tuple(c.candidate_id for c in candidates
                       if all(getattr(c, f) == getattr(first, f) for f in fields))
        run_production_search(path, ProductionExecutionLimits(
            stop_after="tuning", candidate_ids=chosen,
            tuning_seed=plan.seeds.stages.tuning[0], max_new_runs=len(chosen),
        ))
PY
```

Do not launch the full matrix until these pilots are reviewed. Width and strength ranges
remain provisional; this extension introduces no representation normalization. The shared
search artifact inventory still verifies prepared manifests even for pair-free objectives;
the two-layer control itself requires no pair data and control-only CMNIST execution does
not load pair feature arrays.

### MatchDG migration

The featurizer-bias correction was already present in code. Corrected searches now
record `objective_version: bias-free-pair-difference/v2`; historical candidate digests
cannot be reused. Checked-in MatchDG configs (including held-out and test-oracle
tracks) now add `-bias-free-v2` to experiment names and output roots, before the final
`-test-oracle` suffix on diagnostic tracks. Commands are
unchanged; copied old configs must similarly choose a fresh output root before rerunning.
Historical files are not modified. Other standalone search identities remain compatible.

Selected checkpoint manifests now include the projection basis needed for inference.
Restore their state into a fresh `LinearProbeAlgorithm` through
`PersistedLinearCheckpointStore`; pair data are not needed for inference. Historical
head-only GRIT checkpoints still require their original fitted projection.
