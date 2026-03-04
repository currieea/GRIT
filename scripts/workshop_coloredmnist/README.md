# Workshop ColoredMNIST Results

This folder provides an executable workflow for the workshop comparison on standard `ColoredMNIST`:

- `ERM`
- `ECMP`
- `KernelGRIT`

All runs use frozen CLIP features (`--pretrained true`) and select best config per seed by `in_val.acc_avg`.

## Profiles

Two profiles are supported:

- `reduced` (default): quick workshop sweep
- `extended`: multi-seed + wider tuning sweep

## 1) Dry-run and manifest generation

Reduced profile (default):

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py
```

Extended profile in isolated output dir:

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py \
  --profile extended \
  --results-dir results/cmnist_extended
```

## 2) Execute experiments

Reduced:

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py --execute
```

Extended:

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py \
  --profile extended \
  --results-dir results/cmnist_extended \
  --execute
```

Optional smoke test:

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py --execute --max-runs 3
```

Logs are written to:

- `results/workshop_coloredmnist/logs/<METHOD>/*.log`

## 3) Aggregate results

Reduced:

```bash
uv run python scripts/workshop_coloredmnist/aggregate_results.py
```

Extended:

```bash
uv run python scripts/workshop_coloredmnist/aggregate_results.py \
  --results-dir results/cmnist_extended
```

This writes:

- `results/workshop_coloredmnist/summary/all_runs.csv`
- `results/workshop_coloredmnist/summary/selected_by_seed.csv`
- `results/workshop_coloredmnist/summary/summary.csv`
- `results/workshop_coloredmnist/summary/summary.md`

## Defaults used

Shared args:
  - `--dataset ColoredMNIST`
  - `--pretrained true`
  - `--root_dir ./data`
  - `--batch_size 256`
  - `--lr 0.001`
  - `--weight_decay 1e-4`
  - `--epochs 40`
  - `--featurizer linear`
  - `--split_scheme official`
  - `--no_wandb`

Reduced profile defaults:
- Seeds: `1001`
- `ERM`: 1 run
- `ECMP`: projection in `{conditional, oracle}`, `param1` in `{6,14,22}`
- `KernelGRIT`: projection `{oracle}`, `param1=2000`, `param2=10000`, `param3` in `{0.01,0.03}`
- Total planned runs: `9`

Extended profile defaults:
- Seeds: `1001,1002,1003,1004,1005`
- `ERM`: 5 runs
- `ECMP`: projection in `{conditional, oracle}`, `param1` in `{2,6,10,14,18,22}`
- `KernelGRIT`: projection `{oracle}`, `param1=2000`, `param2` in `{1000,3000,10000,30000}`, `param3` in `{0.003,0.01,0.03,0.1}`
- Total planned runs: `145`

## Optional overrides

You can override profile defaults using:

- `--seed-list "1001,1002,1003"`
- `--ecmp-projections "conditional,oracle"`
- `--ecmp-param1 "2,6,10"`
- `--kgrit-param2 "1000,10000"`
- `--kgrit-param3 "0.01,0.03"`
