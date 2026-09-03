# GRIT: Domain Generalization via Invariant Feature Projection

GRIT removes spurious correlations by projecting input features onto the null space of
nuisance directions, the directions that differ across domains for the same class. The
nuisance directions are estimated from counterfactual pairs (oracle, conditional
matching, or nearest-neighbor matching).

> The method is called **ECMP** in the inherited code under `solver/`. It was renamed to
> **GRIT** in the paper. The new implementation lives in `src/grit/`.

## Setup

```bash
uv sync --frozen --extra cu128 --group dev   # GPU machines
uv sync --frozen --extra cpu --group dev     # WSL / laptops
```

On the ECN servers, run `scratch-project` inside the repo first. It exports
`PROJECT_SCRATCH` and puts the virtualenv on node-local disk. Every path below defaults
to `$PROJECT_SCRATCH/{data,artifacts,outputs}`, so the checked-in configs run unmodified.

## ColoredMNIST

```bash
scratch-project
uv run --frozen --extra cu128 grit prepare cmnist        # downloads MNIST + CLIP, ~minutes on GPU
uv run --frozen --extra cu128 grit run configs/cmnist/production-search.yaml --pilot
uv run --frozen --extra cu128 grit run configs/cmnist/production-search.yaml   # full grid, use tmux
uv run --frozen --extra cu128 grit status configs/cmnist/production-search.yaml
```

`prepare` writes `$PROJECT_SCRATCH/artifacts/cmnist-none/` (dataset manifest, 256 oracle
pairs, CLIP feature cache). `run` validates those artifacts, writes the plan under
`$PROJECT_SCRATCH/outputs/cmnist-primary/`, and runs every task it does not already find
there. It is safe to interrupt and rerun; completed tasks are reused. `--pilot` runs one
ERM and one GRIT tuning task and stops. `--dry-run` writes the plan and prints status
without training.

The grid is 16 ERM and 400 GRIT (16 by 25 ranks) candidates, 3 tuning seeds, top-3
confirmation with 2 more seeds, and 10 final seeds for the winner. Selection uses
validation only; see `docs/experiments/cmnist.md`.

## Waterbirds

Waterbirds-CF is rebuilt from released Waterbirds-95, CUB images and masks, and four
Places365 categories. Put them under `$PROJECT_SCRATCH/data/{waterbirds,cub,cub-masks,places}`
(or pass `--released-root`, `--cub-root`, `--masks-root`, `--places-root`), then:

```bash
uv run --frozen --extra cu128 grit prepare waterbirds
uv run --frozen --extra cu128 grit run configs/waterbirds/production-search.yaml --pilot
uv run --frozen --extra cu128 grit run configs/waterbirds/production-search.yaml
```

See `docs/experiments/waterbirds.md` for the construction and protocol.

## Custom paths

Copy a config, edit the paths, and pass the copy to `grit run`. Absolute paths and
`${ANY_ENV_VAR}` both work. Preparation seeds must match the config
(`--construction-seed 1729`, `--pair-seed 2718` are the defaults and match the checked-in
configs). Use a separate prepared root and config for the L2-normalized sensitivity.

## Development

```bash
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu basedpyright
uv run --frozen --extra cpu pytest
uv run --frozen --extra cpu grit smoke cmnist       # hermetic end-to-end with fake encoder
uv run --frozen --extra cpu grit smoke waterbirds
```

Guidance for contributors and agents is in `AGENTS.md`. The original rewrite plan,
contracts, and progress log are archived under `docs/history/`.
