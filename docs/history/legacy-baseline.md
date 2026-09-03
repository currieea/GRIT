# Inherited implementation baseline

Status: **Preserved and statically characterized; runtime reproduction deferred**

This document records the inherited implementation as a historical and behavioral
reference. It does not claim that any published experiment was reproduced.

## Baseline identity and preservation

The inherited baseline is local branch `main` at commit
`66c282b6846e8564d165b73d643670be727a2cab` (`66c282b`, "added agents.md file. Copy of
claude.md"). On 2026-08-20, both `git rev-parse main` and
`git merge-base main rewrite` returned that exact commit.

The baseline was inspected from `rewrite` without checking out `main`. A path-restricted
diff from `66c282b` through this checkpoint contains no changes under `main.py`,
`datasets/`, `models/`, `solver/`, `experiments/`, `scripts/`, or `utils.py`. Those paths
remain the preserved implementation; the new package is developed alongside them.

## Representative inherited commands

These commands are transcribed from the inherited [README](../README.md). They are
historical entry points, not commands verified by this audit.

Setup:

```bash
pip install torch torchvision wilds wandb tqdm numpy pandas pillow
pip install git+https://github.com/openai/CLIP.git
```

Preprocessing examples:

```bash
python scripts/coloredMNIST_preprocess.py
python scripts/waterbirds_preprocess.py
python scripts/celeba_preprocess.py
```

Representative direct run:

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

Representative sweep launchers:

```bash
python experiments/cmnist/ecmp_oracle.py
python experiments/cmnist/ecmp_condition.py
python experiments/cmnist/ecmp_nearest.py
python experiments/cmnist/erm.py
python experiments/waterbirds/ecmp_oracle.py
```

Each sweep module calls `wandb.sweep(...)` and `wandb.agent(...)` at module scope; for
example, see [the CMNIST oracle launcher](../experiments/cmnist/ecmp_oracle.py#L4-L31)
and [the Waterbirds oracle launcher](../experiments/waterbirds/ecmp_oracle.py#L4-L29).
Importing or executing one therefore creates external W&B state and starts an agent.
None was launched during this audit.

The direct command above also exposes a verified CLI defect: `--root_dir` is declared
with `action="store_true"`, so it cannot accept the documented path value
([`main.py`](../main.py#L48-L68)). The historical command is retained here as evidence,
not presented as currently working.

## Inherited responsibility map

| Responsibility | Inherited location and observed behavior |
|---|---|
| CLI, seeding, tracking, dispatch | [`main.py`](../main.py#L20-L45) initializes W&B, selects a device, seeds libraries, dynamically resolves a solver with `eval(...)`, and starts training. |
| Dataset, loaders, model, optimization, training, evaluation, and reporting | [`solver/erm.py`](../solver/erm.py#L25-L60) constructs datasets and loaders, models and optimizers; its `fit`, `evaluate`, and `report` methods own the rest of the lifecycle. |
| Method-specific updates | Classes under [`solver/`](../solver/) inherit the combined ERM lifecycle. ECMP overrides projected training and constructs pairs/projection in [`solver/ecmp.py`](../solver/ecmp.py#L9-L47). |
| Dataset construction and artifact loading | Modules under [`datasets/`](../datasets/) define split arrays, metadata, transforms, metrics, and direct or cached inputs. Dataset classes are selected by string concatenation and `eval(...)` in [`ERM.__init__`](../solver/erm.py#L25-L41). |
| Models and feature encoders | Modules under [`models/`](../models/) provide the linear classifier, CNN, ResNet, CLIP wrapper, and related helpers. ERM decides which to construct in [`_initialize_model`](../solver/erm.py#L160-L174). |
| Hyperparameter search | Files under [`experiments/`](../experiments/) combine W&B configuration, search spaces, external project/entity names, and immediate sweep execution. |
| Feature and pair preprocessing | Top-level [`scripts/`](../scripts/) perform dataset access, CLIP inference, pair-difference construction, and direct artifact writes. These are inherited executable programs, not a reusable package layer. |

This concentration of responsibilities is why the rewrite separates dataset bundles,
pair builders, projection, algorithms, selection, evaluation, tracking, and orchestration.

## Observed evaluation and model-selection behavior

The following statements are verified directly from code:

- Every epoch, `ERM.evaluate()` evaluates every non-training, non-counterfactual split,
  including `val`, `in_test` where present, and `test`
  ([`solver/erm.py`](../solver/erm.py#L90-L114)). Test data is therefore accessed before
  training and selection are frozen.
- `ERM.report()` tracks three in-memory log dictionaries: the epoch with the best
  `in_test` key metric, the epoch with the best `val` key metric, and the epoch with the
  best `test` key metric ([`solver/erm.py`](../solver/erm.py#L134-L154)). The validation
  record is updated but is not used in the final W&B summary or to restore a checkpoint.
- The dataset key metric is `acc_avg` for ColoredMNIST
  ([`datasets/colored_mnist.py`](../datasets/colored_mnist.py#L188-L200)) and `acc_wg` for
  both Waterbirds variants ([`datasets/waterbirds.py`](../datasets/waterbirds.py#L109-L111),
  [`datasets/waterbirds.py`](../datasets/waterbirds.py#L154-L156)). Thus the report's
  test-oracle epoch uses average accuracy for ColoredMNIST and worst-group accuracy for
  Waterbirds.
- No ordinary ERM checkpoint is saved or restored. The final summary includes metrics
  from the best `in_test` epoch and the best `test` epoch, while the best validation log
  remains unused ([`solver/erm.py`](../solver/erm.py#L147-L158)).
- All 60 checked-in Python files that call `wandb.sweep(...)` declare the optimization
  metric as `test.acc_avg`; representative definitions appear in
  [`experiments/cmnist/erm.py`](../experiments/cmnist/erm.py#L4-L25) and
  [`experiments/waterbirds/ecmp_oracle.py`](../experiments/waterbirds/ecmp_oracle.py#L4-L24).
  Consequently, the inherited sweep configuration uses test average accuracy for
  hyperparameter selection, including Waterbirds where `ERM.report()` separately tracks
  `acc_wg` as its dataset key metric.

These are static control-flow observations. Their numerical effects were not measured.

## Oracle projection information versus test-oracle selection

The word *oracle* refers to two independent information flows:

1. With `projection=oracle`, ECMP reads a precomputed pair-difference tensor from
   `dataset.diff` and estimates a nuisance projector
   ([`solver/ecmp.py`](../solver/ecmp.py#L9-L26),
   [`solver/ecmp.py`](../solver/ecmp.py#L152-L154)). For cached datasets this tensor is
   loaded from `diff.pth`, such as in
   [`ColoredMNISTClipDataset`](../datasets/colored_mnist.py#L203-L211) and
   [`CounterfactualWaterbirdsClipDataset`](../datasets/waterbirds.py#L122-L135). This is
   pair/projection information supplied during training.
2. Independently, `ERM.report()` tracks the epoch maximizing a test metric and every
   sweep optimizes `test.acc_avg`. That is test-oracle model or hyperparameter selection.

Oracle pairs do not inherently require test-oracle selection. The rewrite permits an
explicitly labeled training-side projection oracle while requiring ordinary checkpoint
and configuration selection to use validation data only.

## Dataset and artifact assumptions

- The inherited default data root is `/local/scratch/a/bai116/datasets/`, embedded in
  [`main.py`](../main.py#L48-L52) and preprocessing programs such as
  [`scripts/coloredMNIST_preprocess.py`](../scripts/coloredMNIST_preprocess.py#L48-L53)
  and [`scripts/waterbirds_preprocess.py`](../scripts/waterbirds_preprocess.py#L56-L63).
- Preprocessing writes directly to hard-coded `new_dir` directories and assumes those
  directories and external model/data dependencies are available. Cached loaders expect
  files such as `x_array.pth`, split/label/metadata arrays, and `diff.pth` without a
  checked manifest or artifact hash.
- Raw ColoredMNIST construction requests MNIST downloads at dataset construction time
  ([`datasets/colored_mnist.py`](../datasets/colored_mnist.py#L54-L68)). The Waterbirds
  preprocessing path instead passes `download=False` and assumes a prebuilt
  Counterfactual Waterbirds dataset and metadata are already present
  ([`scripts/waterbirds_preprocess.py`](../scripts/waterbirds_preprocess.py#L56-L64)).
- The repository contains no tracked dataset or feature artifacts. At audit time, neither
  the inherited default data root nor a repository-local `data/` directory existed.

## Known ambiguities and correctness concerns

Verified implementation mismatches or leakage paths:

- The documented `--root_dir PATH` form conflicts with the parser's boolean flag
  declaration, as noted above.
- ECMP projects training features in `fit()`
  ([`solver/ecmp.py`](../solver/ecmp.py#L28-L47)) but inherits `ERM.evaluate()`, which
  passes unprojected `x` to the model ([`solver/erm.py`](../solver/erm.py#L90-L114)).
- Conditional matching shuffles global training indices but stores positions in that
  shuffled metadata array, then indexes the full feature array with those positions
  rather than mapping through `train_idx`
  ([`solver/ecmp.py`](../solver/ecmp.py#L49-L100)).
- The ColoredMNIST preprocessing program comments out creation of `diff.pth`
  ([`scripts/coloredMNIST_preprocess.py`](../scripts/coloredMNIST_preprocess.py#L53-L71)),
  although `ColoredMNISTClipDataset` requires that file.
- Waterbirds preprocessing infers alternating counterfactual endpoints from one grouped
  loader batch rather than reading explicit pair identities
  ([`scripts/waterbirds_preprocess.py`](../scripts/waterbirds_preprocess.py#L61-L72)).
- Waterbirds metadata admits land, water, snow, and desert backgrounds
  ([`datasets/waterbirds.py`](../datasets/waterbirds.py#L12-L31)); the canonical rewrite
  protocol uses only the four binary bird/land-water groups.
- Test metrics drive the inherited sweep and test-oracle epoch behavior described above.
  The ordinary ERM lifecycle has no explicit resolved-configuration, code-revision,
  dependency-environment, or artifact-manifest record, nor aggregated-seed selection or
  selected-checkpoint restoration.

Suspected runtime impact, not established by this audit:

- Applying ECMP to unprojected evaluation inputs may change predictions because the
  classifier is initialized before projected training; the size of that effect is
  unmeasured.
- The conditional-matching index mismatch may select unintended endpoints. Pair counts,
  label/domain constraints after indexing, and downstream numerical impact were not
  checked against data.
- The Waterbirds grouped-loader ordering may or may not reproduce the intended 240
  controlled pairs for a particular external artifact. No artifact was available to
  inspect.
- The inherited configuration sets Waterbirds' default optimizer to SGD
  ([`datasets/waterbirds.py`](../datasets/waterbirds.py#L12-L31)); rewrite planning flags
  this as differing from the paper's stated frozen-feature optimizer, but this audit did
  not reproduce either result.

## Runtime characterization deferred

No required dataset or cached-feature artifact was present at the inherited default or
repository-local data roots, and no historical run artifact is tracked. Model weights
were not fetched, and the external W&B service was intentionally not invoked. Per scope,
no data was downloaded, no W&B sweep was created, and no costly legacy experiment was
run. The baseline is therefore preserved and statically characterized, not
runtime-reproduced. Future compatibility work may run narrowly scoped commands once
versioned artifacts and external-service requirements are available; it must record
those inputs and must not reinterpret test-selected historical results as ordinary
validation-selected results.
