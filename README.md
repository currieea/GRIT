# GRIT: Domain Generalization via Invariant Feature Projection

GRIT removes spurious correlations by projecting input features onto the null space of nuisance directions — directions that differ across domains for the same class. The nuisance directions are estimated from counterfactual pairs (oracle, conditional matching, or nearest-neighbor matching).

> **Note:** The method is called **ECMP** throughout the codebase. It was renamed to **GRIT** in the most recent version of the paper.

## Rewrite development

The rigor-first rewrite is being built in `src/grit/` alongside the inherited
implementation. Its experiment settings belong in `configs/`; the top-level `main.py`,
`datasets/`, `models/`, `solver/`, `experiments/`, and `scripts/` paths remain
compatibility and historical references during migration.

Create the locked development environment and run its checks with:

```bash
uv sync --frozen --extra cpu --group dev
uv run --extra cpu python --version
uv run --extra cpu ruff check .
uv run --extra cpu basedpyright
uv run --extra cpu pytest
```

The tracked `.python-version` selects Python 3.10.20 as the current reproducible rewrite
development interpreter. `pyproject.toml` retains a Python 3.10 minimum and mutually
exclusive CPU and CUDA 12.8 profiles using PyTorch 2.11.0 and torchvision 0.26.0. Official
OpenAI CLIP remains pinned at revision
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. CUDA is currently available only for
single-GPU float32 feature preparation; training and projection fitting remain CPU-only.

Run the explicitly non-reportable, offline CMNIST ERM/oracle-GRIT smoke profile with:

```bash
uv run --frozen --extra cpu grit-cmnist-run configs/cmnist/smoke.yaml
```

The command uses deterministic synthetic MNIST-like sources and a fake 512-dimensional
encoder, then exercises construction, training-only oracle pairs, projection, real Adam
training, validation selection, checkpoint restoration, final-test gating, and canonical
local results. Generated output is written under the ignored `artifacts/` directory.

Prepare real official MNIST and unnormalized OpenAI CLIP ViT-B/32 features explicitly:

```bash
uv run --frozen --extra cpu grit-cmnist-prepare \
  --data-root /path/to/mnist \
  --clip-weights-root /path/to/clip-weights \
  --output-root /path/to/cmnist-cache \
  --construction-seed 1729 \
  --pair-seed 2718 \
  --normalization none \
  --feature-device cpu \
  --clip-batch-size 256 \
  --allow-download
```

Omit `--allow-download` to require that both source data and weights already exist.
These seeds match the checked production-search example; preparation seeds must match the
configuration that consumes the artifacts. For GPU extraction, create a separate locked
environment with `UV_PROJECT_ENVIRONMENT=.venv-cu128 uv sync --frozen --extra cu128`, then
run the same command from that environment with `--extra cu128 --feature-device cuda` and
an explicit `CUDA_VISIBLE_DEVICES`. Preparation records the resolved device, precision,
batch size, PyTorch/CUDA versions, GPU identity, and compute capability in the feature
manifest. It does not run the full scientific hyperparameter sweep.

### Production-capable local search

Milestone 6A adds a local, configuration-driven search for the approved CMNIST and
Waterbirds-CF ERM/oracle-GRIT studies over already prepared official OpenAI CLIP feature
artifacts. The checked-in examples contain conspicuous placeholder paths and therefore
cannot accidentally fall back to smoke data:

```bash
uv run --frozen --extra cpu grit-search plan configs/cmnist/production-search.yaml
uv run --frozen --extra cpu grit-search plan configs/waterbirds/production-search.yaml
uv run --frozen --extra cpu grit-search pilot-candidates /path/to/production-search.yaml
uv run --frozen --extra cpu grit-search status /path/to/production-search.yaml
uv run --frozen --extra cpu grit-search run /path/to/production-search.yaml
```

`plan` parses and cross-validates the dataset, feature-cache, and oracle-pair manifests,
checks every referenced feature artifact digest, and writes the complete 416-candidate
plan without loading feature arrays, training, creating checkpoints, or issuing final-test
access. `run` executes or continues the exact 3-seed tuning, top-three confirmation, and
10-seed final lifecycle. Both `plan` and `run` require a real Git commit and a clean
worktree before they write or train. `pilot-candidates` and `status` are deliberately
different: both require the existing authored/resolved/plan triplet and are read-only, so
they remain usable from a dirty development tree. The former presents two canonical pilot
tasks; the latter verifies and reports canonical run and selection state.

All four operations take the same authored production YAML. The output root must be a
dedicated nonexistent, empty, or exactly compatible prior-search directory, disjoint from
the repository root and prepared inputs. If it is inside the source repository, it must be
Git-ignored so generated output cannot change the recorded code dirty-state.

Local JSON is authoritative. A production-capable command is not evidence that the real
scientific grid was executed: this repository has run only hermetic smoke workflows and
manifest-only plan tests. Real reportable execution still requires verified server assets
and an approved server environment. W&B mirroring, conditional/nearest pairing, and later
algorithms remain deferred.

For first execution on a real server, follow the
[server-execution runbook](docs/server-execution.md). Bounded `run` controls can execute
canonical tuning tasks from the full plan before the unrestricted lifecycle; they do not
alter scientific identities or make an incomplete pilot reportable.

## Inherited setup

The commands below describe the preserved implementation and have not been reproduced as
part of the rewrite scaffold. See the
[inherited baseline record](docs/legacy-baseline.md) for static findings and runtime
limitations.

**Requirements:** Python 3.8, PyTorch, CUDA recommended.

```bash
pip install torch torchvision wilds wandb tqdm numpy pandas pillow
pip install git+https://github.com/openai/CLIP.git
```

You will also need a [WandB](https://wandb.ai) account (or pass `--no_wandb` to skip logging).

## Datasets

The codebase supports: **ColoredMNIST**, **RotatedMNIST**, **PACS**, **Waterbirds**, **CelebA**, **Camelyon**.

The inherited code intends `--root_dir` to select a data root and embeds
`/local/scratch/a/bai116/datasets/` as its default. Its current parser mistakenly defines
that option as a boolean flag, so the documented path-valued form below is not valid
without a compatibility fix.

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

### Option 1: Historical W&B sweep launchers

Each file under `experiments/<dataset>/` defines a grid sweep and launches an agent immediately:

```bash
python experiments/cmnist/ecmp_oracle.py    # GRIT with oracle counterfactuals
python experiments/cmnist/ecmp_condition.py # GRIT with conditional matching
python experiments/cmnist/ecmp_nearest.py   # GRIT with nearest-neighbor matching
python experiments/cmnist/erm.py            # ERM baseline
python experiments/cmnist/irm.py            # IRM baseline
# etc.
```

### Option 2: Historical direct command

This command records the inherited intended interface; it is not currently valid as
written because of the `--root_dir` parser issue above.

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
