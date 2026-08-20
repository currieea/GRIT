# Target architecture for the GRIT rewrite

Status: **Minimal Milestone 3 spine implemented and verified; awaiting user review**

## Architectural intent

The rewrite separates experiment orchestration from scientific components. The current
`ERM` class is simultaneously a baseline algorithm, dataset factory, loader factory,
model factory, training loop, evaluator, selector, and logger. The target design assigns
those responsibilities to explicit components.

The architecture should remain practical for PyTorch and WILDS-style datasets. It should
adopt the clean boundaries of Kernel-GRIT without assuming that this broader repository
has only one training loop or only frozen representations.

[`contracts.md`](contracts.md) now distinguishes approved core behavior from provisional
CMNIST guidance and deferred extensions. Its scientific/leakage invariants are binding;
concrete type names, complete field sets, and module boundaries remain internal and
revisable until both CMNIST and Waterbirds have exercised them.

## Approved core boundary

The initial contract spine is deliberately small:

- strict Pydantic v2 configuration/result boundaries for CMNIST ERM and oracle GRIT;
- role-scoped training, validation, final-test, and diagnostic views;
- validation-only ordinary checkpoint/candidate selectors with deterministic ties;
- separate candidate and per-run checkpoint freezes plus restoration before final test;
- distinct ordinary and CMNIST test-oracle configuration/result types;
- bounded algorithm updates inside trainer-owned iteration;
- canonical local JSON and a null event sink; and
- one in-memory lifecycle test proving final metrics cannot flow backward.

CPU-float64 SVD fitting, earlier-epoch checkpoint ties, and union confirmation of CMNIST's
two selector finalist sets are approved directions. Only the tie/selection behavior belongs
in the contract spine; production SVD code waits for the CMNIST vertical slice.

The current internal spine is intentionally flat and small:
`schemas.py`, `config.py`, `data.py`, `selection.py`, `checkpoints.py`, `lifecycle.py`,
`results.py`, and `tracking.py` under `src/grit/`. These are working module names, not a
public layout commitment. They contain strict boundaries, selectors, a fake-state-capable
restoration interface, the final-test gate, and the null sink; they do not contain real
dataset, feature, pair, projection, algorithm, trainer, or tracking integrations.

## Provisional component map

The map below is design inventory, not a request to scaffold it. Milestone 3 should add
only the few internal modules required by the tested spine, with names chosen for current
clarity rather than promised stability. Dataset/feature/pair/projection implementations,
W&B, and later algorithms appear here only as deferred placement guidance.

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
    grit.py  # ECMP remains an inherited/historical name, not the new owner hierarchy
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

Status: **Provisional through vertical slices; production manifests/caches deferred.**

A dataset adapter is responsible for constructing named splits, metadata, groups, and
artifact provenance. It must not decide the model-selection policy or instantiate an
algorithm.

Conceptual output:

```text
source artifacts + typed dataset config
  -> internal DatasetBundle + manifests
  -> role-scoped training, pair-source, validation, final-test, and diagnostic capabilities
```

The full bundle is internal to dataset construction and the runner's capability broker.
Trainers, pair builders, evaluators, and selectors do not receive a dictionary containing
every split. Repeated views of one source partition carry stable source identities as well
as distinct example/view identities.

### Pair builders

Status: **Scientific access rules approved; production interfaces and builders deferred to
the CMNIST/later slices.**

Pair builders select aligned source examples and return explicit indices, metadata, and
provenance. They do not calculate classifier loss or own a training loop.

```text
training-only PairSourceView + PairBuilderConfig -> PairSet
training-only PairSourceView + OraclePairRelationView + OraclePairBuilderConfig -> PairSet
```

Oracle, conditional, and nearest produce the same conceptual pair-set contract. The
oracle relation is a separate capability that estimated builders cannot receive.

### Projection

Status: **Mathematical separation and CPU-float64 initial fitting approved; production API,
mathematics, persistence, and artifact format deferred to CMNIST.**

Projection estimates nuisance directions from a `PairSet` and transforms feature rows.
It is classifier-independent and independently testable.

```text
PairSet -> fitted Projection
features + fitted Projection -> transformed features
```

The runner's representation pipeline applies the same fitted transform to training and
every permitted evaluation role. Projection does not live in a dataset adapter, model,
algorithm, or trainer.

### Algorithms

Status: **Ownership split approved; exact protocol provisional; future-method hooks
deferred.**

Algorithms own method-specific optimization state and updates. They receive prepared
models, batches, and context; they do not discover datasets or decide which split selects
the final checkpoint.

The initial interface needs only a bounded fake update for the synthetic lifecycle. CMNIST
will define the smallest real ERM/GRIT protocol. Verified future needs such as multiple
optimizer steps, parameter replacement, non-model state, and step-level validation remain
documented in `contracts.md`, but must not add hooks before the corresponding method is in
scope.

### Experiment runner

Status: **Lifecycle ordering approved; production orchestration deferred to CMNIST.**

The eventual runner owns the lifecycle:

```text
resolve config
  -> load dataset bundle
  -> construct optional pairs/projection
  -> run each candidate on tuning seeds with validation-selected checkpoints
  -> freeze per-method/per-selector top-three finalist artifacts
  -> run finalists on confirmation seeds
  -> freeze one five-seed candidate per method and selector
  -> run each fresh final seed
  -> freeze that run's validation-selected checkpoint
  -> restore the matching checkpoint
  -> evaluate final test for that run
  -> write result and provenance
```

Milestone 3 exercises this ordering with in-memory views and fake checkpoints rather than a
production runner. The eventual runner must not implement algorithm-specific mathematics.
Its distinct candidate and per-run checkpoint tokens ensure final-seed validation can
choose an epoch but cannot change frozen hyperparameters. Final-test access becomes legal
only after both decisions are frozen and the matching checkpoint is restored.

### Selection policy

Status: **Approved core; exact record/type names provisional.**

Selection is an explicit configuration and result object. Ordinary selection receives
only a validation-record type; final-test records are structurally excluded rather than
hidden behind metric names. A separately labeled diagnostic may calculate a permitted
test-oracle envelope, but it uses distinct configuration/result types and cannot replace
the ordinary result.

### Tracking

Status: **Null sink approved for Milestone 3; production W&B integration deferred.**

Canonical local JSON is authoritative. Milestone 3 supplies only a null sink. A future W&B
adapter may mirror configurations, histories, and artifacts but never defines selection
semantics. A run must remain possible without W&B.

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

- Boundary schemas use Pydantic v2 with strict unknown-field rejection.
- Experiment settings live in data files rather than executable sweep modules.
- Configurations are validated into typed objects.
- Dataset, algorithm, pair builder, projection, selector, and tracker are selected through
  explicit registries.
- Resolved configurations are saved with results.
- Machine-specific paths are CLI/environment overrides, not committed defaults.
- Method-specific parameters have descriptive names; avoid generic `param1`, `param2`,
  and `param3` in the new path.

## Result and provenance principles

Milestone 3 results record only the fields needed to round-trip the synthetic lifecycle and
CMNIST ERM/oracle-GRIT boundary. The following fuller inventory is provisional guidance for
vertical slices rather than an initial schema checklist:

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

- Exact internal type names, field sets, and module boundaries
- Which provisional boundaries survive both CMNIST and Waterbirds unchanged
- Exact safe numerical artifact format and checkpoint retention policy
- Minimal artifact-reference interface needed by the CMNIST slice
- Upper supported Python version and the compatible PyTorch/CLIP/CUDA matrix

Scientific construction and estimated-pair choices remain in their protocol documents.
The smallest contract spine is implemented and synthetically verified. After user approval,
exercise and revise it directly in the CMNIST vertical slice. Waterbirds is the second
proving ground; only after both slices should internal names or module boundaries be treated
as stable shared interfaces. Generalized artifacts, full resume, production W&B,
later-method hooks, and raw-image/accelerator abstractions remain explicitly deferred.
