# Target architecture for the GRIT rewrite

Status: **Milestone 4 CMNIST reviewed and complete; Milestone 5 Waterbirds vertical slice
active**

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

Status: **CMNIST construction and the Waterbirds-CF construction/manifest boundary are
implemented; the shared shape remains provisional until the complete Waterbirds slice is
reviewed.**

A dataset adapter is responsible for constructing named splits, metadata, groups, and
artifact provenance. It must not decide the model-selection policy or instantiate an
algorithm.

Conceptual output:

```text
source artifacts + typed dataset config
  -> internal DatasetBundle + manifests
  -> role-scoped training, pair-source, validation, final-test, and diagnostic capabilities
```

Milestone 4 implements this concretely for injected MNIST-like pools and the explicit
torchvision adapter. `cmnist-stratified-hash-v1` produces fixed source partitions and a
canonical manifest; deterministic rendered tables issue existing role-scoped views. The
full bundle is internal to dataset construction and the runner's capability broker.
Trainers, pair builders, evaluators, and selectors do not receive a dictionary containing
every split. Repeated views of one source partition carry stable source identities as well
as distinct example/view identities.

Milestone 5 adds a dataset-specific Waterbirds-CF parser and construction boundary. It
validates the released Waterbirds, CUB image/mask/annotation, and four approved Places
inventories; deterministically selects the 184/56 controlled sources and backgrounds by
SHA-256 ranking; reproduces the GroupDRO center-crop/mask/composite geometry; preserves
released validation/test bytes; and emits a canonical manifest with explicit supervised
records and oracle relationships. A strict non-reportable fixture profile exercises the
same path without relaxing the production 4,795-record and four-group profile. No source
acquisition or generalized artifact store is part of this boundary.

### Pair builders

Status: **CMNIST and Waterbirds-CF clean-oracle builders implemented; estimated builders
deferred.**

Pair builders select aligned source examples and return explicit indices, metadata, and
provenance. They do not calculate classifier loss or own a training loop.

```text
training-only PairSourceView + PairBuilderConfig -> PairSet
training-only PairSourceView + OraclePairRelationView + OraclePairBuilderConfig -> PairSet
```

Oracle, conditional, and nearest produce the same conceptual pair-set contract. The
oracle relation is a separate capability that estimated builders cannot receive.
Milestone 4 implements only the CMNIST training-source capability and 256-source clean
red-minus-green oracle builder with a canonical pair manifest. The capability binds the
validated source-pool content and dataset manifest; stable pair IDs and the pair manifest
derive that dataset identity rather than accepting one from the caller.

Milestone 5 keeps all 4,795 Waterbirds-CF endpoints in one shared supervised view whose
records omit background and pair fields. The separately issued oracle-relation capability
can produce only the construction's exact 184 landbird and 56 waterbird land-minus-water
relationships, bound to the canonical dataset digest. The non-reportable fixture uses its
declared reduced strata, while the production pair manifest rejects any count other than
184/56. ERM cannot be passed to this builder through its public typed interface.

### Frozen features

Status: **Pinned OpenAI CLIP boundary and dataset-specific CMNIST/Waterbirds caches
implemented; production weights and Waterbirds assets remain server inputs.**

Waterbirds feature preparation accepts the validated construction and a path-capable
encoder, processes variable-sized images through the pinned OpenAI CLIP preprocessing,
and writes one referenced float32 array plus a canonical manifest. The manifest binds the
dataset, encoder revision, weights, preprocessing, normalization, row/image identities,
and array digest. Training tables redact background metadata; validation tables expose
the approved group fields. The deterministic fake encoder marks every cache
non-reportable and exists only for offline lifecycle tests.

### Projection

Status: **CMNIST and Waterbirds CPU-float64 linear projection paths implemented; final
artifact format remains provisional.**

Projection estimates nuisance directions from a `PairSet` and transforms feature rows.
It is classifier-independent and independently testable.

```text
PairSet -> fitted Projection
features + fitted Projection -> transformed features
```

The runner's representation pipeline applies the same fitted transform to training and
every permitted evaluation role. Projection does not live in a dataset adapter, model,
algorithm, or trainer.
CMNIST and Waterbirds projection diagnostics are bound to the exact pair-manifest and
feature-cache-manifest digests used by fitting. Waterbirds resolves the canonical land and
water endpoint rows only after checking their dataset, role, label, background, and image
identities against the cache. The final-test feature capability is likewise bound to the
exact cache manifest, not only to dataset or source IDs.

### Algorithms

Status: **Ownership split and concrete CMNIST ERM/GRIT linear-probe update implemented;
future-method hooks deferred.**

Algorithms own method-specific optimization state and updates. They receive prepared
models, batches, and context; they do not discover datasets or decide which split selects
the final checkpoint.

CMNIST validates a concrete two-class linear algorithm that owns its model, Adam optimizer,
optional fitted input projection, bounded batch update, prediction, and inference state.
The trainer owns epoch/batch iteration, validation cadence, and checkpoint capture.
Verified future needs such as multiple optimizer steps, parameter replacement, non-model
state, and step-level validation remain documented in `contracts.md`, but add no hooks
before the corresponding method is in scope.

### Experiment runner

Status: **Concrete local CMNIST orchestration implemented; generalized search scheduling
and production sweeps deferred.**

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

Milestone 3 exercised this ordering with in-memory views and fake checkpoints. Milestone 4
now exercises it with real linear-probe updates, durable finalist/winner JSON, a narrow
selected-linear-checkpoint format, and canonical final results. The runner does not
implement algorithm-specific mathematics.
Its distinct candidate and per-run checkpoint tokens ensure final-seed validation can
choose an epoch but cannot change frozen hyperparameters. Final-test access becomes legal
only after both decisions are frozen and the matching checkpoint is restored.

### Selection policy

Status: **CMNIST and Waterbirds validation-only selectors implemented; exact shared
record/type names remain provisional.**

Selection is an explicit configuration and result object. Ordinary selection receives
only a validation-record type; final-test records are structurally excluded rather than
hidden behind metric names. A separately labeled diagnostic may calculate a permitted
test-oracle envelope, but it uses distinct configuration/result types and cannot replace
the ordinary result.

Waterbirds uses one dataset-specific validation record containing all four group counts
and accuracies. Worst-group accuracy is primary; adjusted average is recomputed with the
validated Waterbirds-CF training group counts, never validation proportions. Checkpoint
ties then use earlier epoch, while candidate ties additionally prefer lower projection
rank and stable candidate identity. Three-seed ordered finalists retain their exact tuning
decisions; confirmation accepts only the two fresh seed records and reuses those stored
decisions for the five-seed winner. ERM and GRIT are ranked independently.

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

Milestone 4 implements `grit-cmnist-prepare` and `grit-cmnist-run` under
`src/grit/cli/`. The first is the explicit real MNIST/official-CLIP preparation boundary;
the second consumes the strict, non-reportable hermetic smoke YAML. Neither command
contains a second training or selection implementation.

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

Milestone 4 reuses the ordinary result boundary for each final seed and adds concrete
dataset, feature, pair, projection-diagnostic, and selected-checkpoint references. The
following fuller inventory remains provisional guidance rather than a requirement to build
a generalized provenance framework:

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
- Upper supported Python version and the compatible PyTorch/CLIP/CUDA matrix

Scientific construction and estimated-pair choices remain in their protocol documents.
The smallest contract spine is implemented and synthetically verified. After user approval,
exercise and revise it directly in the CMNIST vertical slice. Waterbirds is the second
proving ground; only after both slices should internal names or module boundaries be treated
as stable shared interfaces. Generalized artifacts, full resume, production W&B,
later-method hooks, and raw-image/accelerator abstractions remain explicitly deferred.
