# GRIT: Domain Generalization via Invariant Feature Projection

GRIT removes spurious correlations by projecting input features onto the null space of
nuisance directions, the directions that differ across domains for the same class. The
nuisance directions are estimated from counterfactual pairs (oracle, conditional
matching, or nearest-neighbor matching).

> The method is called **ECMP** in the inherited code under `legacy/`. It was renamed to
> **GRIT** in the paper. The new implementation lives in `src/grit/`, driven by the
> scripts in `scripts/`.

## Current state (September 2026)

`main` now contains the complete linear development stack that previously ended at
`waterbirds-paper-table`. The old branch names remain useful as historical checkpoints,
but they are all ancestors of `main`, not competing implementations:

- `rewrite` is the tested GRIT rewrite and the first CMNIST/RotatedMNIST slice;
- `cmnist-paper-table` adds the full frozen-feature CMNIST baseline and selection
  matrix; and
- `waterbirds-paper-table` adds Waterbirds-CF construction, all table baselines, and
  validation/test-oracle search tracks.

The implementation is integrated and tested, but the experimental tables are still
being finalized. Ten-seed CMNIST runs have been synthesized; the existing GroupDRO
numbers should remain excluded until the identified implementation issue is corrected
and rerun. A reportable Waterbirds artifact and several complete validation-selected
searches exist on Jujube. The Waterbirds numbers are still under a paper-comparability
audit, particularly the unusually strong LISA result and the gap between the rewritten
GRIT result and the inherited paper result. The separately labeled Waterbirds
test-oracle configs had not been initialized at the last server check. Do not treat
those diagnostic configs as ordinary validation-selected results.

Large artifacts and results are intentionally not committed. Their canonical records
live at the `artifacts` and `output_root` paths in each config, normally under
`$PROJECT_SCRATCH/{artifacts,outputs}`. A complete search writes its reportable
aggregate to `output_root/summaries/`; use `scripts/search_status.py` against the exact
config to check whether that output tree is complete.

## Files that matter

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Repository conventions and the scientific leakage rules that all changes must preserve. |
| `docs/experiments/*.md` | Authoritative dataset construction, information-access, selection, and reporting protocols. Change the relevant protocol before changing experimental behavior. |
| `configs/README.md` and `configs/<dataset>/*.yaml` | Map of reportable grids and their resolved artifact/output locations. Ordinary and test-oracle tracks use separate configs and output trees. |
| `scripts/prepare_*.py` | Build datasets, oracle pair banks, and frozen CLIP feature caches. |
| `scripts/run_search.py` and `scripts/search_status.py` | Run/resume a search and inspect its canonical phase/completion counts. |
| `src/grit/` | Current implementation: data construction, features, methods, validation-only selection, search orchestration, and result schemas. |
| `tests/` and `scripts/smoke.py` | Unit/integration coverage and hermetic end-to-end checks. |
| `legacy/` | Inherited ECMP code and original W&B launchers, retained only to audit or port paper behavior. It is not the active implementation. |
| `docs/history/` | Archived rewrite plans and contracts. These provide context, not current instructions. |

## Setup

```bash
uv sync --group dev
```

That installs the default PyTorch build (CUDA on Linux). On the ECN servers, run
`scratch-project` inside the repo first so the virtualenv, data, and outputs land on
node-local disk. Every path below defaults to `$PROJECT_SCRATCH/{data,artifacts,outputs}`;
without that variable, `GRIT_SCRATCH` or a git-ignored `scratch/` in the repo is used.

## ColoredMNIST

```bash
uv run scripts/prepare_cmnist.py
uv run scripts/run_search.py configs/cmnist/production-search.yaml --pilot
uv run scripts/run_search.py configs/cmnist/production-search.yaml
uv run scripts/search_status.py configs/cmnist/production-search.yaml

# Independent GroupDRO baseline (after the ERM/GRIT pilot is healthy)
uv run scripts/run_search.py configs/cmnist/groupdro-search.yaml --pilot
uv run scripts/run_search.py configs/cmnist/groupdro-search.yaml
uv run scripts/search_status.py configs/cmnist/groupdro-search.yaml
```

`prepare_cmnist.py` downloads MNIST and the pinned CLIP weights if needed, builds the
partitions and 256 oracle pairs, and caches CLIP features under
`$PROJECT_SCRATCH/artifacts/cmnist-none/`. It uses `cuda:0` when available; pass
`--device cuda:1` or `--device cpu` to override.

`run_search.py` validates those artifacts, writes the plan under
`$PROJECT_SCRATCH/outputs/cmnist-primary-r2-24/`, and runs every task not already there. It is
safe to interrupt and rerun. `--pilot` runs one ERM and one GRIT tuning task and stops.
`--dry-run` writes the plan and prints status without training. Run the full grid inside
tmux.

The primary grid is 16 ERM and 368 GRIT (16 by ranks 2 through 24) candidates. The
independent GroupDRO grid has 48 candidates (16 by three adversarial step sizes). Both
use 3 tuning seeds, top-3 confirmation with 2 more seeds, and 10 final seeds for the
winner. The remaining paper-table baselines and the explicitly diagnostic test-oracle
tracks have their own configs; see `configs/README.md`. Selection uses validation only
unless the config and output are explicitly labeled `test-oracle`; see
`docs/experiments/cmnist.md`.

## RotatedMNIST

The first RotatedMNIST slice uses disjoint MNIST source partitions, 0- and 45-degree
training environments, validation at 0, 45, and 60 degrees, and the official MNIST test
sources only for final 90-degree evaluation. Oracle pairs are the exact same training
image rendered at 0 and 45 degrees.

```bash
uv run scripts/prepare_rotated_mnist.py
uv run scripts/run_search.py configs/rotated_mnist/production-search.yaml --pilot
uv run scripts/run_search.py configs/rotated_mnist/production-search.yaml
uv run scripts/search_status.py configs/rotated_mnist/production-search.yaml
```

Prepared artifacts default to
`$PROJECT_SCRATCH/artifacts/rotated-mnist-none/`; production outputs default to
`$PROJECT_SCRATCH/outputs/rotated-mnist-primary-r2-24/`. The grid has the same 16 ERM
and 368 oracle-GRIT candidates as the primary CMNIST comparison. See
`docs/experiments/rotated_mnist.md` for the protocol and leakage boundary.

## Waterbirds

Waterbirds-CF is rebuilt from released Waterbirds-95, CUB images and masks, and four
Places365 categories. Put them under `$PROJECT_SCRATCH/data/{waterbirds,cub,cub-masks,places}`
or pass `--released-root`, `--cub-root`, `--masks-root`, `--places-root`. Then:

```bash
uv run scripts/prepare_waterbirds.py
uv run scripts/run_search.py configs/waterbirds/production-search.yaml --pilot
uv run scripts/run_search.py configs/waterbirds/production-search.yaml
uv run scripts/search_status.py configs/waterbirds/production-search.yaml
```

The baseline and diagnostic tracks are separate configs under `configs/waterbirds/`.
Run or inspect each with the same `run_search.py`/`search_status.py` commands. A status
error saying that no complete production plan exists means that the exact config's
output tree has not been initialized; it does not invalidate results stored under a
different config. See `docs/experiments/waterbirds.md` for the construction, protocol,
and reporting boundary.

## Custom paths and grids

Copy a config, edit it, and pass the copy to `run_search.py`. Paths accept absolute
values and `${ANY_ENV_VAR}`. The learning-rate, weight-decay, and rank grids, the epoch
count, and `pair_count` are all ordinary settings; a quick check might use two learning
rates, three ranks, and five epochs. `pair_count` may be any prefix of the prepared pair
bank, so prepare once with `--pair-count 512` for the pair-budget sensitivity.
Preparation seeds must match the config; the defaults (`--construction-seed 1729`,
`--pair-seed 2718`) match the checked-in configs. Use a separate prepared root and
config for the L2-normalized sensitivity.

## Development

```bash
uv run ruff check .
uv run basedpyright
uv run pytest
uv run scripts/smoke.py cmnist        # hermetic end-to-end with a fake encoder
uv run scripts/smoke.py waterbirds
```

Start with `AGENTS.md` and the relevant file under `docs/experiments/` before changing
experimental behavior.
