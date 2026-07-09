# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the official repo for the **GRIT** method (called **ECMP** in the codebase — the name was changed to GRIT in the most recent version of the paper; they refer to the same method) — a domain generalization research codebase. It benchmarks multiple domain generalization algorithms (ERM, IRM, REx, Fish, GroupDRO, MatchDG, LISA, SWAD, and the novel ECMP method) across several datasets (ColoredMNIST, RotatedMNIST, PACS, Waterbirds, CelebA, Camelyon).

## Running Experiments

Experiments are run via WandB sweeps. Each experiment script in `experiments/` defines a sweep configuration:

```bash
# Run a sweep configuration (e.g., ECMP with oracle projection on ColoredMNIST)
python experiments/cmnist/ecmp_oracle.py
# This creates a WandB sweep and launches an agent

# Run main.py directly (bypassing sweep)
python main.py --solver ECMP --dataset ColoredMNIST --projection oracle \
  --param1 10 --pretrained true --lr 0.001 --epochs 40 --no_wandb
```

Key `main.py` arguments:
- `--solver`: `ERM`, `IRM`, `REx`, `Fish`, `GroupDRO`, `ECMP`, `MatchDG`, `LISA`, `SWAD`
- `--dataset`: `ColoredMNIST`, `LISAColoredMNIST`, `RotatedMNIST`, `PACS`, `Waterbirds`, `CelebA`, `Camelyon`
- `--pretrained true/false`: use CLIP-preprocessed features (`true`) or raw pixels/images (`false`)
- `--projection`: `oracle`, `conditional`, or `nearest` (ECMP only)
- `--param1`: primary method hyperparameter (e.g., number of SVD components for ECMP)
- `--param2`, `--param3`: secondary method hyperparameters
- `--featurizer`: `linear` (default, no featurizer), `cnn`, `resnet`
- `--no_wandb`: disable WandB logging, print metrics to stdout instead

## Dataset Preprocessing

Before running experiments with `--pretrained true`, datasets must be preprocessed with CLIP embeddings. Scripts are in `scripts/`:

```bash
python scripts/coloredMNIST_preprocess.py   # generates CLIP features for ColoredMNIST
python scripts/waterbirds_preprocess.py
python scripts/celeba_preprocess.py
# etc.
```

Preprocessed data is saved as `.pth` files (`x_array.pth`, `y_array.pth`, `split_array.pth`, `metadata_array.pth`, `diff.pth`) in a versioned subdirectory (e.g., `ColoredMNIST-cf-clip_v1.0/`). The default data root is `/local/scratch/a/bai116/datasets/`.

## Architecture

### Entry Point
`main.py` parses args, initializes WandB, instantiates the solver via `eval(hparam['solver'])(hparam)`, and calls `solver.fit()`.

### Solver Hierarchy (`solver/`)
All solvers inherit from `ERM` (`solver/erm.py`):
- `ERM.__init__` loads the dataset, builds train/eval data loaders, initializes model and optimizer.
- `ERM.fit()` runs the training loop with optional WandB logging and calls `self.evaluate()` each epoch.
- `ERM.report()` tracks best model by in-domain test, OOD val, and OOD test accuracy.
- Subclasses override `fit()` and may override loader properties (`loader_type`, `uniform_over_groups`, `domain_fields`, `n_groups_per_batch`).

**ECMP** (`solver/ecmp.py`) is the novel method: it projects input features onto the null space of nuisance directions. The `diff` tensor (counterfactual differences) is computed via `oracle` (from precomputed `dataset.diff`), `conditional` (label-matched cross-domain pairs), or `nearest` (nearest-neighbor cross-domain pairs). SVD of `diff.T` gives nuisance directions; projection removes them.

### Dataset Layer (`datasets/`)
All datasets extend `WILDSCFDataset` (in `datasets/wilds_cf_dataset.py`), which itself extends WILDS's `WILDSDataset`. Each dataset class (e.g., `ColoredMNISTDataset`) and its CLIP variant (`ColoredMNISTClipDataset`) live in the same file. The dataset class is selected dynamically in `ERM.__init__` via `eval(dataset_name + 'Dataset')` or `eval(dataset_name + 'ClipDataset')`.

Key dataset attributes used by solvers:
- `_x_array`, `_y_array`, `_split_array`, `_metadata_array`: raw tensors
- `default_domain_fields`: field name used for domain grouping
- `diff`: precomputed oracle counterfactual differences (CLIP variant only)
- `default_optimizer`: `'sgd'` or `'adam'`
- `key_metric`: e.g., `'acc_avg'` or `'acc_wg'`

### Model Layer (`models/`)
- `Classifier`: linear head
- `MNIST_CNN`, `ResNet50`: featurizers for raw-pixel inputs
- `Clip`: CLIP ViT-B/32 encoder used for preprocessing
- `Linear`, `Identity`: pass-through wrappers

Models are always wrapped in `nn.DataParallel` after initialization.

### Experiment Scripts (`experiments/`)
Each file under `experiments/<dataset>/` defines a WandB sweep config and immediately creates + launches a sweep agent. The `experiments/` root also contains older monolithic scripts that pre-date the per-dataset subdirectory organization.

## Dependencies

Core: PyTorch, `wilds`, `wandb`, `tqdm`, `numpy`, `torchvision`, `clip` (OpenAI).

Python 3.8 is assumed (based on `__pycache__` artifacts).
