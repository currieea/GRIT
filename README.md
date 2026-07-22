# GRIT: Domain Generalization via Invariant Feature Projection

GRIT removes spurious correlations by projecting input features onto the null space of nuisance directions — directions that differ across domains for the same class. The nuisance directions are estimated from counterfactual pairs (oracle, conditional matching, or nearest-neighbor matching).

> **Note:** The method is called **ECMP** throughout the codebase. It was renamed to **GRIT** in the most recent version of the paper.

## Setup

**Requirements:** Python 3.8, PyTorch, CUDA recommended.

```bash
pip install torch torchvision wilds wandb tqdm numpy pandas pillow
pip install git+https://github.com/openai/CLIP.git
```

For the pinned environment used by the Table 1 reproduction workflow:

```bash
pip install -r requirements-reproduction.txt
```

You will also need a [WandB](https://wandb.ai) account (or pass `--no_wandb` to skip logging).

## Datasets

The codebase supports: **ColoredMNIST**, **RotatedMNIST**, **PACS**, **Waterbirds**, **CelebA**, **Camelyon**.

Set your data root via `--root_dir`. The default path in the codebase is `/local/scratch/a/bai116/datasets/`.

### Using CLIP features (recommended)

Most experiments use CLIP-preprocessed features rather than raw pixels. Run the appropriate preprocessing script once before training:

```bash
python scripts/coloredMNIST_preprocess.py
python scripts/waterbirds_preprocess.py
python scripts/celeba_preprocess.py
# etc.
```

Each script saves `x_array.pth`, `y_array.pth`, `split_array.pth`, `metadata_array.pth`, and (for GRIT oracle) `diff.pth` into a versioned subdirectory (e.g., `ColoredMNIST-cf-clip_v1.0/`). You'll need to update the hardcoded `root_dir` and `new_dir` paths in each script before running.

## Running Experiments

### Reproducing Table 1: ColoredMNIST

The reproduction workflow uses `LISAColoredMNIST` for every executed row so
that methods are compared on one dataset construction. It prepares deterministic
frozen CLIP ViT-B/32 features, runs the audited hyperparameter grids, and writes
local JSON, CSV, Markdown, and per-run logs. W&B is optional.

The data root is chosen in this order: `--data-root`, `GRIT_DATA_ROOT`, then
`data/` inside this repository. On the Inouye Lab cluster, for example:

```bash
export GRIT_DATA_ROOT=/local/scratch/a/currie15/datasets/GRIT

python scripts/reproduce_table1.py prepare
python scripts/reproduce_table1.py run --quick
python scripts/reproduce_table1.py run
```

An explicit CLI path overrides the environment variable:

```bash
python scripts/reproduce_table1.py prepare --data-root /path/to/grit-data
python scripts/reproduce_table1.py run --data-root /path/to/grit-data
```

`run` executes the complete 164-run grid at seed 1001. `--quick` executes one
historically selected configuration for each of the 13 supported rows. Runs are
stored under `reproduction_results/cmnist/runs/` and resume automatically. Use
`--dry-run` to inspect commands, `--rows ROW_ID ...` to select rows,
`--epochs 1` for smoke tests, or `--force` to rerun completed configurations.
Prepared tensor hashes are checked once when a run starts; use
`--skip-data-hash-check` only when that validation cost is undesirable.

To log the reproduction to your own W&B project:

```bash
python scripts/reproduce_table1.py run --quick \
  --wandb-project grit-reproduction \
  --wandb-entity YOUR_ENTITY
```

The final comparison is written to:

```text
reproduction_results/cmnist/results.json
reproduction_results/cmnist/table1_cmnist.csv
reproduction_results/cmnist/table1_cmnist.md
```

Models are selected only by `in_test.acc_avg`; the reported OOD result is taken
from the same epoch. MatchDG CNN, MatchDG CLIP finetuning, and ERM oracle remain
explicitly unresolved because the exported runs do not establish their
configurations. Random guess and theory oracle are retained as reference rows.

The strongest historical ERM candidate used the repository's distinct
`ColoredMNIST` construction. This workflow intentionally uses
`LISAColoredMNIST` for ERM as well, prioritizing a controlled comparison over a
mixed-dataset numerical match. Consequently, its ERM value may differ from the
paper more than the other rows.

The audited GRIT sweeps record `param2=512` or `1024` even though the paper
appendix describes 256 counterfactual pairs. The checked-in ECMP implementation
does not apply its commented-out `param2` subsampling, so the manifest preserves
the logged values and reports this discrepancy rather than assigning them new
semantics.

### Option 1: WandB sweeps (used for paper results)

Each file under `experiments/<dataset>/` defines a grid sweep and launches an agent immediately:

```bash
python experiments/cmnist/ecmp_oracle.py    # GRIT with oracle counterfactuals
python experiments/cmnist/ecmp_condition.py # GRIT with conditional matching
python experiments/cmnist/ecmp_nearest.py   # GRIT with nearest-neighbor matching
python experiments/cmnist/erm.py            # ERM baseline
python experiments/cmnist/irm.py            # IRM baseline
# etc.
```

### Option 2: Direct execution

```bash
python main.py \
  --solver ECMP \
  --dataset LISAColoredMNIST \
  --pretrained true \
  --projection oracle \
  --param1 10 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --batch_size 256 \
  --epochs 40 \
  --seed 1001 \
  --root_dir /path/to/datasets/ \
  --no_wandb
```

## Key Arguments

| Argument | Description |
|---|---|
| `--solver` | `ERM`, `ECMP` (GRIT), `IRM`, `REx`, `Fish`, `GroupDRO`, `MatchDG`, `LISA`, `SWAD` |
| `--dataset` | `ColoredMNIST`, `LISAColoredMNIST`, `RotatedMNIST`, `PACS`, `CounterfactualWaterbirds`, `CelebA`, `Camelyon` |
| `--pretrained` | `true` = use CLIP features, `false` = use raw pixels/images |
| `--projection` | `oracle`, `conditional`, or `nearest` (GRIT/ECMP and MatchDG only) |
| `--param1` | Number of SVD components to remove (GRIT/ECMP); penalty weight for IRM/REx |
| `--featurizer` | `linear` (default, operates on CLIP features), `cnn` (MNIST), `resnet` (images) |
| `--no_wandb` | Disable WandB, print metrics to stdout |

## Baselines

| Solver | Description |
|---|---|
| `ERM` | Empirical Risk Minimization |
| `IRM` | Invariant Risk Minimization |
| `REx` | Risk Extrapolation |
| `Fish` | Gradient matching across domains |
| `GroupDRO` | Group Distributionally Robust Optimization |
| `MatchDG` | Domain generalization via contrastive matching |
| `LISA` | Learning Invariant Predictors with Selective Augmentation |
| `SWAD` | Stochastic Weight Averaging Densely |
| `ECMP` | **GRIT** (this paper's method) |
