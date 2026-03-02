# Workshop ColoredMNIST Results

This folder provides an executable workflow for the workshop comparison on standard `ColoredMNIST`:

- `ERM`
- `ECMP`
- `KernelGRIT`

All runs use frozen CLIP features (`--pretrained true`) and select best config per seed by `in_val.acc_avg`.

## 1) Dry-run and manifest generation

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py
```

This writes:

- `results/workshop_coloredmnist/manifest.csv`

## 2) Execute experiments

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py --execute
```

Optional smoke test:

```bash
uv run python scripts/workshop_coloredmnist/run_experiments.py --execute --max-runs 3
```

Logs are written to:

- `results/workshop_coloredmnist/logs/<METHOD>/*.log`

## 3) Aggregate results

```bash
uv run python scripts/workshop_coloredmnist/aggregate_results.py
```

This writes:

- `results/workshop_coloredmnist/summary/all_runs.csv`
- `results/workshop_coloredmnist/summary/selected_by_seed.csv`
- `results/workshop_coloredmnist/summary/summary.csv`
- `results/workshop_coloredmnist/summary/summary.md`

## Defaults used

- Seeds: `1001`
- Shared args:
  - `--dataset ColoredMNIST`
  - `--pretrained true`
  - `--root_dir ./data`
  - `--batch_size 256`
  - `--lr 0.001`
  - `--weight_decay 1e-4`
  - `--epochs 40`
  - `--featurizer linear`
  - `--no_wandb`
- Method grids:
  - `ERM`: 1 run
  - `ECMP`: projection in `{conditional, oracle}`, `param1` in `{6,14,22}`
  - `KernelGRIT`: projection `{oracle}`, `param1=2000`, `param2=10000`, `param3` in `{0.01,0.03}`

Total planned runs: `9`.
