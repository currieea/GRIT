# Target architecture for the GRIT rewrite

Status: **Approved direction; concrete interfaces remain milestone-scoped**

## Architectural intent

The rewrite separates experiment orchestration from scientific components. The current
`ERM` class is simultaneously a baseline algorithm, dataset factory, loader factory,
model factory, training loop, evaluator, selector, and logger. The target design assigns
those responsibilities to explicit components.

The architecture should remain practical for PyTorch and WILDS-style datasets. It should
adopt the clean boundaries of Kernel-GRIT without assuming that this broader repository
has only one training loop or only frozen representations.

## Proposed layout

```text
pyproject.toml
uv.lock

src/grit/
  __init__.py
  config.py
  results.py

  data/
    base.py
    registry.py
    colored_mnist.py
    waterbirds.py
    loaders.py

  features/
    base.py
    clip.py
    image.py
    cache.py

  pairing/
    base.py
    oracle.py
    conditional.py
    nearest.py

  projection/
    linear.py

  models/
    classifier.py
    featurizers.py

  algorithms/
    base.py
    erm.py
    ecmp.py
    groupdro.py
    irm.py
    rex.py
    matchdg.py
    fish.py
    lisa.py
    swad.py

  training/
    runner.py
    checkpoints.py
    selection.py

  evaluation/
    evaluator.py
    metrics.py
    groups.py

  tracking/
    base.py
    local.py
    wandb.py

  cli/  # add command modules only when their commands are implemented

configs/
  cmnist/
  waterbirds/

tests/
  data/
  pairing/
  projection/
  algorithms/
  training/
  evaluation/
```

This is a target map, not a requirement to create every file before it is needed.
Prefer adding the smallest coherent component required by the current vertical slice.
Command modules and their console entry points are added together; an empty `cli/`
package or speculative entry points are not scaffold requirements.

## Component boundaries

### Dataset bundle

A dataset adapter is responsible for constructing named splits, metadata, groups, and
artifact provenance. It must not decide the model-selection policy or instantiate an
algorithm.

Conceptual output:

```python
DatasetBundle(
    train=...,
    validation=...,
    test=...,
    counterfactual=...,
    metadata=...,
    group_spec=...,
    manifest=...,
)
```

The final concrete type remains to be designed.

### Pair builders

Pair builders select aligned source examples and return explicit indices, metadata, and
provenance. They do not calculate classifier loss or own a training loop.

```text
DatasetBundle + PairBuilderConfig -> PairSet
```

Oracle, conditional, and nearest are implementations of the same conceptual contract.

### Projection

Projection estimates nuisance directions from a `PairSet` and transforms feature rows.
It is classifier-independent and independently testable.

```text
PairSet -> fitted Projection
features + fitted Projection -> transformed features
```

### Algorithms

Algorithms own method-specific optimization state and updates. They receive prepared
models, batches, and context; they do not discover datasets or decide which split selects
the final checkpoint.

The common algorithm interface must allow both ordinary single-batch updates and methods
with specialized epoch/update behavior. Avoid forcing every method through an abstraction
that only fits ERM.

### Experiment runner

The runner owns the lifecycle:

```text
resolve config
  -> load dataset bundle
  -> construct optional pairs/projection
  -> initialize model and algorithm
  -> train
  -> evaluate permitted validation metrics
  -> select and restore checkpoint
  -> evaluate final test
  -> write result and provenance
```

The runner must not implement algorithm-specific mathematics.

### Selection policy

Selection is an explicit configuration and result object. Ordinary selection receives
only approved validation metrics. A separately labeled diagnostic may calculate a
test-oracle envelope, but it must not silently replace the ordinary result.

### Tracking

Local structured output is canonical. W&B mirrors configurations, histories, and
artifacts but does not define selection semantics. A run must remain possible without
W&B.

### Command-line interfaces

New executable logic lives in package modules under `src/grit/cli/`. A command module
parses user input and delegates scientific work to typed package components; it does not
become a second orchestration or algorithm layer. Once a real command is implemented, its
user-facing command is exposed through a `[project.scripts]` entry point in
`pyproject.toml`.

The existing top-level `scripts/` directory remains inherited preprocessing and
compatibility evidence. Its files are not templates for rewrite commands and are not
brought wholesale under strict lint or type checking. All new Python implementation,
including command implementations, must live under the checked `src/grit/` package;
tests remain under `tests/`.

## Configuration principles

- Experiment settings live in data files rather than executable sweep modules.
- Configurations are validated into typed objects.
- Dataset, algorithm, pair builder, projection, selector, and tracker are selected through
  explicit registries.
- Resolved configurations are saved with results.
- Machine-specific paths are CLI/environment overrides, not committed defaults.
- Method-specific parameters have descriptive names; avoid generic `param1`, `param2`,
  and `param3` in the new path.

## Result and provenance principles

Every completed run should record at least:

- Schema version and status
- Resolved configuration
- Seed and deterministic settings
- Git revision and dirty state
- Dependency versions and device information
- Dataset and feature artifact manifests
- Pair source indices or an auditable pair manifest
- Projection rank and numerical diagnostics
- Training and validation history
- Selected configuration, epoch, metric, and checkpoint
- Final test metrics
- Elapsed time and failure information

## Legacy compatibility

During migration, the old and new paths coexist. `main.py` may eventually become a
compatibility shim, but it must not be redirected until a new vertical slice is verified.
Legacy numerical behavior and recommended rigorous behavior may be exposed as distinct
configurations where comparison is useful. Existing top-level preprocessing scripts may
remain compatibility references, but new commands use package modules and installed
entry points.

## Unresolved architectural decisions

- Exact configuration library and schema mechanism
- Concrete dataset and batch types
- Whether projection occurs in dataset adapters, model wrappers, or runner preparation
- Minimum algorithm interface needed by Fish and SWAD
- Checkpoint storage format and retention policy
- Local search scheduler and W&B integration boundary
- Upper supported Python version and the compatible PyTorch/CLIP/CUDA matrix

Resolve these through the first two vertical slices rather than speculative abstraction.
