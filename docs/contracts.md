# Shared contracts and design guidance

Status: **Minimal Milestone 3 spine approved, implemented, and verified.**
Detailed type names, field sets, and module boundaries remain internal rather than a
supported public API. Both CMNIST and Waterbirds have now exercised the scientific
boundaries; Milestone 6A reuses only their demonstrated lifecycle mechanics for local
production search.

This proposal turns the approved experiment protocols into shared interfaces for the
rewrite. It preserves useful mathematical behavior without preserving the inherited
architecture in which `ERM` owns dataset construction, models, optimization, evaluation,
selection, and reporting (`../solver/erm.py`, especially lines 25-60 and 90-158). The
CMNIST and Waterbirds protocol documents remain authoritative for scientific choices.
The classification below is binding for implementation scope. Later sections retain the
original detailed reasoning, but a detailed sketch is not automatically an exit criterion
or a promise of a stable public API.

## How to read this document

Each design statement belongs to one of three classes:

1. **Approved core contract:** a scientific or lifecycle invariant that the initial
   Milestone 3 contract spine must enforce now.
2. **Provisional guidance:** a useful design hypothesis to implement only as far as the
   CMNIST ERM/oracle-GRIT slice needs it, then revise from evidence. Waterbirds must also
   exercise a name or boundary before it is treated as stable.
3. **Deferred extension:** explicitly outside the initial Milestone 3 implementation. It
   may remain documented to prevent rediscovery, but it must not create code, abstractions,
   dependencies, or exit criteria yet.

Scientific protocol statements remain governed by
[`experiments/cmnist.md`](experiments/cmnist.md) and
[`experiments/waterbirds.md`](experiments/waterbirds.md). Classification changes below do
not weaken their split, pair, selection, or reporting safeguards.

### Approved core contracts

The following direction is approved and must shape the smallest implementation:

- Use Pydantic v2 for strict configuration and result boundary schemas. The implementation
  checkpoint adds the bounded `pydantic>=2.11,<3` dependency.
- Compose datasets, algorithms, training, evaluation, selection, and reporting rather
  than inheriting them from `ERM`.
- Give training, validation, final-test, and diagnostic consumers role-scoped views.
- Ordinary selectors accept validation metric records only; final-test and diagnostic
  records are different types and cannot be ordinary selector inputs.
- Keep training-side oracle pair information distinct from test-oracle model selection.
- Model ordinary experiments and CMNIST test-oracle diagnostics as distinct discriminated
  configuration and result types.
- Freeze a validation-selected hyperparameter/candidate decision separately from each
  final run's validation-selected checkpoint.
- Restore the selected checkpoint before opening final-test evaluation.
- Let an algorithm own a bounded update while the trainer owns iteration and lifecycle
  control. Milestone 3 does not need a universal hook system to enforce this ownership.
- Make canonical local JSON the authoritative serialized boundary. W&B remains an
  optional mirror and is not part of experiment semantics.
- Fit the initial small pair-difference SVD deterministically on CPU in float64 when the
  CMNIST slice implements projection.
- Use earlier epoch as the last semantic checkpoint tie-break before stable identity.
- Confirm the union of CMNIST primary- and secondary-selector finalists once while
  retaining a separate frozen winner for each selector.

### Initial Milestone 3 implementation spine

Milestone 3 implementation is intentionally narrower than the complete design inventory.
It consists only of:

- strict Pydantic boundary models for configuration and results needed by CMNIST ERM and
  oracle GRIT;
- split-role and role-scoped view types;
- distinct validation, final-test, and diagnostic metric record types;
- validation-only checkpoint and candidate selectors, including deterministic ordering;
- minimal checkpoint identity and inference-restoration contracts exercised with fake
  state, without promising training resume;
- ordinary versus CMNIST test-oracle result models;
- canonical JSON serialization and round trips;
- a null event sink; and
- one in-memory synthetic lifecycle using a fake bounded algorithm update inside
  trainer-owned iteration and proving that validation selects a checkpoint, the selected
  checkpoint is restored, and final-test metrics have no path back to training or ordinary
  selection.

This spine should be the smallest coherent implementation. A concrete name from later in
this document should be introduced only when the spine requires it.

The implemented internal spine is the eight focused modules under `../src/grit/`:
`schemas.py`, `config.py`, `data.py`, `selection.py`, `checkpoints.py`, `lifecycle.py`,
`results.py`, and `tracking.py`. It binds CMNIST split names to their approved roles, binds
oracle-pair configuration to the approved training source partitions, keeps full
resolved-config identity separate from selector-independent scientific-candidate identity,
and revalidates selection/checkpoint/seed evidence when authoritative result JSON is
parsed. The focused Milestone 3 tests live under `../tests/`; that checkpoint added no
dataset, feature, pair, projection, PyTorch trainer, or external tracking implementation.
The implemented CMNIST slice adds only its concrete versions of those first four
components and a real linear-probe trainer/runner, while tracking and generalized
infrastructure remain deferred.

### Provisional guidance to validate through CMNIST

The following ideas are useful starting points but remain revisable:

- the exact `ExperimentSpec`, dataset bundle, record, batch, capability, selection, and
  result type names and their complete field lists;
- the exact module layout and registry organization;
- candidate/run identity layering beyond the minimum needed for deterministic selection;
- representation-provider, model-factory, evaluator, runner, and algorithm protocol
  signatures;
- lightweight pair/projection interfaces and diagnostics needed by the CMNIST slice;
- path override mechanics, failure envelopes, histories, timing records, and extended
  provenance fields; and
- any local artifact reference boundary used by that slice.

These designs should be changed when the CMNIST slice exposes a simpler or safer shape.
They remain internal after CMNIST and become stable only after Waterbirds exercises them.

### Deferred extensions

The initial implementation must not expand to cover:

- a generalized content-addressed artifact-storage framework;
- full optimizer, scheduler, trainer, or RNG resumability;
- a universal numerical or checkpoint container;
- Fish-, SWAD-, MatchDG-, LISA-, or GroupDRO-specific hooks;
- raw-image, distributed, mixed-precision, compilation, or multi-device abstractions;
- production W&B integration;
- production dataset manifests or feature caches;
- production pair construction or projection mathematics; or
- final artifact-format selection beyond a small replaceable interface boundary.

Production pair/projection work begins only as part of a real vertical slice. Detailed
requirements below remain valuable acceptance criteria for that later work, not Milestone 3
exit criteria.

Milestone 4 is that real slice. It implements CMNIST-specific canonical construction,
feature, and pair manifests; deterministic `.npy` feature tables; the CPU-float64 linear
projection; real frozen-feature ERM/GRIT updates; and a narrow selected-linear-checkpoint
adapter. These concrete boundaries do not approve a generalized artifact store, resume
system, W&B integration, raw-image path, or later-method hooks.

Milestone 5 exercises the same safeguards with Waterbirds-specific construction,
four-group metrics, adjusted-weight provenance, oracle relationships, selection, final
gating, and results. Milestone 6A then adds a deliberately narrow production-search
boundary over already prepared artifacts:

- a strict dataset-discriminated YAML resolves one normalization and the complete approved
  16-ERM/400-oracle-GRIT candidate grid;
- the canonical plan records verified manifest paths and digests, exact 3+2+10 seeds,
  candidate order, expected stage counts, and code/environment identity without loading
  feature arrays or creating final-test capabilities;
- run tasks retain the exact dataset-specific lineage and may be reused only when their
  canonical result and validation/checkpoint trace parse and match the plan;
- final tasks are a separate typed construction requiring the matching frozen candidate;
  each final seed still selects on validation, persists and restores that checkpoint, and
  only then opens final test; and
- canonical local stage results, selection artifacts, seed-addressed summaries, paired
  differences, and a digest-verified experiment index remain authoritative. W&B is not in
  this execution path.

This is not a generalized search, resume, or artifact framework. Continuation is at the
completed-run boundary; optimizer/RNG state and partial-epoch recovery remain unsupported.
The production examples require explicit real manifest paths and cannot fall back to smoke
artifacts. Implementing and testing the path did not execute the real scientific grid and
does not create a reportable result.

## 1. Approved ownership rules and provisional responsibility map

**Classification:** composition and information-access prohibitions are approved core.
The component names and the full matrix are provisional guidance; artifact/provenance and
tracking machinery beyond local JSON and a null sink is deferred.

The experiment lifecycle should be assembled from small components by a runner.
Algorithms implement update behavior; they do not inherit a dataset, trainer, evaluator,
selector, or tracker. Data access is granted through narrow capability objects instead of
passing a dictionary of all loaders or metrics. A component may receive only the
capabilities needed in its current lifecycle phase.

The initial vertical slices use frozen feature vectors. Their internal input boundary
should avoid making later raw-image work impossible, but no raw-image provider or generic
representation framework is designed now. Raw-image optimization, distributed training,
and mixed precision are deferred until a dedicated protocol and vertical need exist.

| Concern | Owner | Reads | Returns or writes | Must never decide or access |
| --- | --- | --- | --- | --- |
| Configuration | Config loader and validator | One authored document plus explicit overrides | Validated `ExperimentSpec` and canonical `ResolvedRunConfig` | Dataset contents, metrics, or implicit scientific defaults |
| Datasets | Dataset builder | Dataset config and source artifacts | `DatasetBundle`, manifest, role-scoped views | Model choice, pairing, selection, or tracking |
| Features | Representation provider | Approved split view, feature config, feature artifacts | `FeatureTable` or raw `InputBatch`, feature manifest | Split roles, labels beyond declared preprocessing needs, or selection |
| Pairs | Pair builder | Training-only pair-source view; oracle relation only for an oracle builder | `PairSet`, diagnostics, pair manifest | Validation/test records, classifier state, projection rank, or metrics |
| Projection | Projection fitter/transformer | Pair endpoint features and projection config | Fitted transform, spectrum, numerical diagnostics, artifact references | Pair discovery, classifier fitting, training, or selection |
| Models | Model factory | Model config and input/output specifications | Model with named representation/logit interfaces | Dataset construction, optimization policy, metrics, or tracking |
| Algorithms | Algorithm instance | Training batches, injected model/optimizer services, optional validation-feedback capability | Update results, predictions, checkpointable algorithm state | Unscoped loaders, final-test data/metrics, candidate selection, or logging policy |
| Training | Trainer | Training capability, algorithm, cadence, checkpoint policy | Histories and checkpoint candidates | Pair construction, final-test evaluation, cross-candidate selection, or W&B semantics |
| Evaluation | Evaluator | One role-scoped evaluation view and inference interface | Role-tagged metric records | Training updates, checkpoint choice, or relabeling roles |
| Selection | Selector | Structurally validation-only records and checkpoint identities | Frozen epoch/candidate decision with audit trace | Final-test or diagnostic metrics and checkpoint mutation |
| Search | Search coordinator | Search config, resolved candidates, validation summaries | Stage decisions and frozen winning candidate | Final-test results, hidden W&B state, or undeclared seeds |
| Results | Result writer | Resolved config, manifests, histories, frozen selection, gated evaluations | Versioned local result and artifact index | Recomputing metrics or changing experiment semantics |
| Provenance | Builders and runner, aggregated by result writer | Code, environment, config, and input identities | Typed provenance records and content hashes | Best-effort omission of required identities |
| Tracking | Optional event sink | Explicit events and the completed local result | A mirror in W&B or another service | Supplying configuration, selecting models, or becoming authoritative storage |

These boundaries deliberately differ from inherited `ERM`. In the inherited path,
subclasses dynamically supply loaders during `ERM.__init__`, the base class constructs the
dataset/model/optimizer, and one evaluation dictionary contains validation and test
metrics. Those facts are statically verified in `../solver/erm.py`; they are constraints to
remove, not interfaces to copy.

## 2. Typed experiment configuration

**Classification:** strict Pydantic v2 boundary validation, unknown-field rejection,
ordinary/CMNIST-test-oracle discrimination, and canonical JSON round trips are approved
core. The complete hierarchy and field inventory are provisional; Waterbirds-only fields,
production manifests, and generalized artifact handling are not initial Milestone 3 work.

### Provisional hierarchy

The root authored type is a strict discriminated union. Search expansion resolves either
branch into immutable resolved values. Names below are descriptive proposed type names:

```text
ExperimentSpec = OrdinaryExperimentSpec | CmnistTestOracleDiagnosticSpec

CommonExperimentFields
  schema_version: "grit.experiment/v1"
  experiment_name: str
  protocol_id: str
  dataset: DatasetConfig
  representation: RepresentationConfig
  pairs: PairBuilderConfig | DisabledPairsConfig
  projection: ProjectionConfig | DisabledProjectionConfig
  model: ModelConfig
  algorithm: AlgorithmConfig
  optimizer: OptimizerConfig
  training: TrainingConfig
  checkpointing: CheckpointConfig
  search: SearchConfig
  output: LocalOutputConfig
  tracking: DisabledTrackingConfig | WandbMirrorConfig
  runtime: RuntimeConfig

OrdinaryExperimentSpec(CommonExperimentFields)
  run_kind: ordinary
  selection: OrdinarySelectionConfig
  evaluation: OrdinaryEvaluationConfig

CmnistTestOracleDiagnosticSpec(CommonExperimentFields)
  run_kind: cmnist_test_oracle_diagnostic
  diagnostic_selection: CmnistTestOracleSelectionConfig
  diagnostic_evaluation: TestOracleEvaluationConfig
```

The diagnostic branch is deliberately CMNIST-specific and contains no ordinary selection
field. The `test_ood` split retains role `final_test`; explicit diagnostic capability and
metric types allow that branch to use it for an oracle envelope without relabeling the
data as `diagnostic_only`. Waterbirds cannot parse the diagnostic branch.

Important nested fields are:

- `DatasetConfig`: `dataset_id`, `construction_method_id`, `source_root`, `artifact_root`,
  `source_partition_definitions`, `split_definitions`, `metadata_schema`, `group_spec`,
  `manifest_requirement`, and a discriminated dataset-specific payload. CMNIST includes
  the official-source partition
  method identifier, environment color correlations, validation rendering correlations,
  and label-noise policy. Waterbirds includes source/reconstruction manifest references
  and the supervised counterfactual endpoint policy. Unresolved method identifiers are
  required values, not silently defaulted algorithms.
- `SplitDefinition`: `name`, `role`, `source_partition_id`, `view_id`, and optional
  declared `group_spec_id`. Roles are `training`, `validation`, `final_test`, and
  `diagnostic_only`. Consumer permissions are derived from the registered protocol and
  role; an authored configuration cannot grant itself broader access.
- `RepresentationConfig`: discriminant `kind` (`frozen_features` initially,
  `raw_images` reserved), `encoder_id`, `encoder_revision`, `preprocessing_id`,
  `normalization` (`none` or `l2`), `feature_dtype`, `artifact_manifest`, and cache policy.
  The approved primary frozen-CLIP experiments explicitly use `normalization: none`;
  L2-normalized features are separate sensitivity experiments. A `feature_space_id` hashes
  encoder, checkpoint, preprocessing, normalization, dtype, and feature dimension; pair,
  projection, training, and evaluation artifacts must all reference the same value.
- `PairBuilderConfig`: discriminant `method` (`oracle`, `conditional_random`, or
  `nearest`), approved source split names, requested pair count, orientation policy,
  pair-seed assignment name, method-definition identifier, and method-specific parameters.
  Open scientific definitions have no default implementation identifier.
- `ProjectionConfig`: `kind: linear_pair_difference`, `requested_rank`,
  `center_differences: false`, SVD policy, fitting dtype/device, numerical-rank tolerance,
  and diagnostic policy. A sweep axis may provide multiple requested ranks, but a resolved
  run has exactly one.
- `ModelConfig`: architecture identifier, input representation name, output label space,
  initialization policy, and explicit architecture parameters. Dataset-specific model
  construction is not hidden in a base algorithm; seed values have one owner in resolved
  `SeedAssignments`.
- `AlgorithmConfig`: discriminated algorithm identifier and algorithm-specific fields.
  ERM has no placeholder parameters; GRIT refers to an already-fitted projection input
  transform rather than owning pair construction or SVD.
- `OptimizerConfig`: optimizer identifier, learning rate, weight decay, momentum/betas,
  and any scheduler config. There is no dataset-owned default optimizer.
- `TrainingConfig`: maximum epochs/updates, batch request and sampling policy,
  validation cadence, early-stop policy, and data-loader worker settings. A resolved
  batch request states whether ordinary, group-uniform, or environment-structured batches
  are required.
- `CheckpointConfig`: cadence, retained checkpoint classes, local artifact format,
  resumability level, and whether RNG state is required. It never names a test metric.
- `OrdinarySelectionConfig`: validation objective, direction, aggregation, tie-break
  sequence, eligible split names, epoch-selection rule, and candidate-selection rule.
  `final_test` is invalid in eligible splits. `CmnistTestOracleSelectionConfig` instead
  names its explicitly test-bearing diagnostic objective and cannot be converted to the
  ordinary type.
- `SearchConfig`: candidate axes, stable candidate ordering, tuning seed set,
  confirmation seed set, final seed set, finalists count, and aggregation rule. The
  approved CMNIST and Waterbirds workflows use three tuning seeds, two additional
  confirmation seeds for the top three candidates, and ten fresh final seeds.
- `OrdinaryEvaluationConfig`: metric specifications by allowed split role, typed group
  aggregation (including any `AdjustedAverageAggregationSpec`), and prediction persistence
  policy, plus separately labeled reporting-only diagnostic specs that cannot be selector
  inputs. `TestOracleEvaluationConfig` permits only the explicit CMNIST diagnostic metric
  path. Neither can promote a split to a different role.
- `LocalOutputConfig`: output root, run naming template, collision policy, atomic-write
  behavior, and artifact-retention policy.
- `WandbMirrorConfig`: project/entity naming, offline/failure policy, and mirrored event
  classes. It has no sweep-controller or selection fields.
- `RuntimeConfig`: Python/device request, worker counts, deterministic-algorithm flag,
  backend flags, thread settings, and one `SeedAssignments` value containing explicit
  construction, pair, model, training, and loader seeds. Requested versus resolved device
  is recorded separately.

`SeedPlan` is logically part of search resolution. It records named seed stages and emits
fully explicit `SeedAssignments` into each `ResolvedRunConfig`; a hash-derived seed is
acceptable only when both the derivation algorithm and its inputs are serialized. A
`candidate_spec_id` hashes resolved scientific configuration, including fixed dataset-
construction and pair seeds, but excludes per-run model/training/loader seeds and machine/
output fields. After required artifacts are materialized or resolved, `candidate_id` binds
that spec ID to dataset, feature, and pair artifact identities. A run identity includes
candidate identity, stage, and all resolved seeds. This prevents the same candidate on two
seeds or different artifacts from being confused while keeping scheduling possible before
artifact creation and multi-seed aggregation well-defined.

### Validation, defaults, and schema mechanism

The alternatives considered were:

1. Frozen standard-library dataclasses plus a handwritten strict decoder and validator.
   This avoids a dependency but requires custom discriminated-union, unknown-key, error,
   and canonical-serialization code. Kernel-GRIT demonstrates that frozen dataclasses are
   useful, but its parser manually reads recognized keys without a general unknown-key
   rejection pass (`../../Kernel-GRIT/src/grit/config.py`, lines 40-133 and 497-508).
2. Pydantic v2 models at the configuration and result boundaries, converted to small
   immutable internal records where useful. It provides nested validation,
   discriminated unions, strict unknown-field rejection, schema generation, and stable
   dump/load APIs at the cost of one runtime dependency.

Pydantic v2 is now the approved boundary-schema mechanism. The dependency is deliberately
not added by this documentation-only revision; it should be added with the first tested
boundary models. The dataclass option remains recorded as an alternative considered, not
an open Milestone 3 choice. Permissive dictionary access is not acceptable.

Validation rules are approved boundary behavior even though their concrete model names and
field layout remain internal:

- Reject unknown fields at every nesting level and reject implicit lossy coercions.
- Require an exactly supported `schema_version`; migration is an explicit pure function,
  never an implicit best-effort load.
- Supply defaults only for operational behavior that cannot change scientific meaning,
  such as `tracking.kind: disabled`. Dataset construction, representation normalization,
  pair definitions, projection rank, selection objective, seed sets, and final-test policy
  must be explicit.
- Validate cross-field rules after parsing: GRIT requires pairs and projection; ERM
  forbids them; pair-source splits must be training-role; ordinary selectors accept only
  validation-role splits; CMNIST test-oracle diagnostics must use the diagnostic root type;
  Waterbirds test-oracle selection is rejected; and final seeds must not overlap tuning or
  confirmation seeds.
- Treat paths in authored configuration as logical or relative paths. Overrides are
  accepted only through a documented allowlist such as `dataset.source_root`,
  `dataset.artifact_root`, and `output.root`. Every override records source, original
  value, and resolved value. Moving the same verified artifact does not change scientific
  identity; resolving to different content changes the manifest-bound candidate ID.
  Scientific fields cannot be changed through an unrecorded environment variable.

Canonical resolved configuration is UTF-8 JSON with sorted object keys, stable enum
strings, preserved list order, explicit `null`, and no NaN or Infinity. Paths are
serialized as normalized logical paths plus a separately recorded machine resolution;
secrets are never serialized. The canonical byte representation is hashed with an
explicit algorithm identifier. YAML may be an authoring format, but it is not the
canonical identity format.

The implemented boundary revalidates nested model instances and public selector inputs.
This is not a claim that Pydantic makes Python a security sandbox: low-level
`model_copy()` and `model_construct()` can bypass construction-time checks. Normal
serialized parsing and selection/freeze entry points therefore reconstruct their external
records, seed sets, and finalist artifacts through strict validation before using them.

## 3. Dataset and split contracts

**Classification:** split roles, role-scoped views, and the absence of final-test data from
training/ordinary selection are approved core. The `DatasetBundle`, record, batch,
manifest, and capability names and full field sets are provisional. CMNIST now has concrete
source/dataset/feature manifests and fixed role-scoped tables; protected-identity
infrastructure and generalized artifact storage remain deferred.

The conceptual dataset records are:

```text
DatasetBundle
  dataset_id: str
  source_partitions: Mapping[SourcePartitionName, SourcePartition]
  splits: Mapping[SplitName, NamedSplit]
  metadata_schema: MetadataSchema
  group_specs: Mapping[GroupSpecId, GroupSpec]
  dataset_manifest: DatasetManifest
  artifact_manifest: ArtifactManifest
  counterfactual_availability: CounterfactualAvailability

NamedSplit
  descriptor: SplitDescriptor
  records: Sequence[ExampleRecord] | RecordStoreRef

SourcePartition
  name: str
  source_ids: Sequence[str] | SourceIndexRef
  origin_split: str
  membership_digest: str

SplitDescriptor
  name: str
  role: training | validation | final_test | diagnostic_only
  source_partition_id: str
  view_id: str
  allowed_consumers: tuple[ConsumerKind, ...]  # derived from protocol, not authored
  group_spec_id: str | None

ExampleRecord
  example_id: str
  source_id: str
  view_id: str
  label: int
  input_ref: ArtifactRef | SourceInputRef
  metadata: TypedMetadata

Batch
  example_ids: Sequence[str]
  source_ids: Sequence[str] | None  # included only when the capability permits them
  inputs: InputBatch
  labels: LabelBatch
  metadata: BatchMetadata
```

`example_id` identifies a concrete rendered or stored observation. `source_id` identifies
the underlying semantic source. `view_id` distinguishes repeated views. Thus CMNIST
`val_e01`, `val_e02`, and `val_e05` contain different `example_id` values while sharing
the same ordered 10,000 `source_id` values. Validation requires a one-to-one source join
across these views and rejects missing, duplicated, or reordered identities. They remain
three metric-bearing validation splits, not 30,000 independent source examples.

CMNIST source partitions and rendered splits are different objects. The
`train_e01_sources` and `train_e02_sources` partitions supply sources to rendered
training-role splits `train_e01` and `train_e02`, which are what supervised optimization
actually consumes. `validation_sources` similarly supplies the three validation views.
Pair sourcing names an approved training source partition and rendering policy; it cannot
substitute a rendered validation split or treat a source partition as a classifier batch.

`TypedMetadata` is dataset-specific but schema declared. It can contain environment,
domain/background, original source index, counterfactual endpoint role, and group fields.
`GroupSpec` defines an ordered, versioned mapping from typed metadata fields to stable group
IDs and human-readable labels, plus expected per-split group counts and their manifest
digest. Evaluators can receive the group fields required by a
metric; training views filter metadata by algorithm permission. In particular,
Waterbirds-CF ERM receives all 4,795 supervised records but does not receive background
labels as an explicit training signal, while GroupDRO may receive the approved group
identity. ERM's training metadata view also redacts pair ID, matched endpoint ID, shared CUB
source identity, and any relation-bearing endpoint field. It retains distinct example IDs
for attribution. Estimated pair builders receive only the example IDs and label/background
metadata allowed by their approved eligibility rule, through their separate pair-source
capability; a protected source identity that would reveal the oracle matching is absent.

Canonical records and manifests may retain protected source identities for construction,
integrity checks, and approved repeated-view joins. A role-scoped view explicitly projects
records into a consumer-specific metadata type, so the existence of a field internally
does not imply that every batch receives it.

`DatasetManifest` records dataset ID/version, construction method ID and parameters,
source artifact hashes, source licenses or citations, source and example counts, group
counts where applicable, ordered split membership hashes, source-to-view relationships,
and code/config provenance. `ArtifactManifest` is an ordered index of `ArtifactRef`
records containing logical name, content digest and algorithm, byte size, media type,
schema/format version, producer identity, and relative URI. Absolute machine paths are
runtime provenance, not portable artifact identity.

An adjusted-average aggregation is typed as `AdjustedAverageAggregationSpec` with a group
spec ID, required complete group set, weights source, and weights-manifest digest. For
Waterbirds, the only approved weights source is the validated Waterbirds-CF training
manifest: landbird/land `3498`, landbird/water `184`, waterbird/land `56`, and
waterbird/water `1057`, divided by 4,795. Validation- or test-sample proportions are
forbidden substitutes, and a missing group is an integrity error.

`CounterfactualAvailability` records whether supervised endpoints and/or pair relations
exist, for which training split, and which capability may expose them. It does not itself
reveal relation rows. For Waterbirds-CF, all methods receive both endpoints as ordinary
supervised training examples. Only the oracle pair builder receives the separate relation
capability containing the 240 endpoint identities. For CMNIST, clean recolor endpoints are
projection-only pair-source records and are not added as supervised classifier examples.

The full `DatasetBundle` is construction-time internal state. Consumers receive narrower
views:

- `SupervisedTrainingView` exposes approved training examples and only permitted metadata.
- `PairSourceView` exposes protocol-approved training records, stable example IDs, and only
  the source/metadata fields permitted for that builder kind.
- `OraclePairRelationView` exposes pair relations and is minted only for an oracle method.
- `ValidationEvaluationView` exposes one or more validation-role splits and evaluator
  metadata.
- `FinalTestHandle` is opaque to training and selection and can be opened only by the
  final-evaluation gate.
- `DiagnosticEvaluationView` is separate from ordinary validation and test access.
- `CmnistTestOracleCapability` is minted only from the still-final-test-role `test_ood`
  handle for the explicit diagnostic root; it is not a generic diagnostic split view.

Capability constructors are internal to the runner/dataset boundary and validate the
registered protocol, split role, dataset-manifest identity, and lifecycle phase. The types
are defenses against accidental misuse inside one process, not a claim of security against
arbitrary hostile Python code.

The implemented CMNIST feature-backed final handle carries the exact canonical feature-
cache-manifest digest. Opening a final view against another cache is rejected even when its
dataset and ordered test source IDs match; normalization, encoder, or feature-content
changes therefore cannot cross the final-evaluation boundary unnoticed.

CMNIST uses the approved `cmnist-stratified-hash-v1` source-partition algorithm documented
in [`experiments/cmnist.md`](experiments/cmnist.md). The Waterbirds acquisition or
reconstruction implementation remains unresolved. Configurations require registered
`construction_method_id` values and manifests; resolution cannot fall back to an arbitrary
split or download.

## 4. Leakage-resistant lifecycle and access boundaries

**Classification:** the one-way data flow, separate candidate/checkpoint freezes,
checkpoint restoration, and structural exclusion of final-test metrics are approved core.
The exact phase-token and service names are provisional. Milestone 3 implements this
boundary once with in-memory views and fake checkpoints, not a production orchestration
framework.

Metric names such as `val_accuracy` and programmer conventions such as “do not read this
dictionary key” are insufficient. The inherited evaluator creates one log dictionary for
non-training splits and `ERM.report()` reads validation and test values from it
(`../solver/erm.py`, lines 90-154). A typo, fallback, or convenience call can therefore
make final-test information available during training. The rewrite instead uses distinct
types, views, and phase tokens.

The search coordinator and each final run advance through monotonic lifecycles:

```text
SEARCH_CONFIGURED -> TUNING_RECORDED -> FINALISTS_FROZEN
                  -> CONFIRMATION_RECORDED -> CANDIDATE_FROZEN

CANDIDATE_FROZEN -> FINAL_RUN_TRAINED -> CHECKPOINT_FROZEN
                 -> CHECKPOINT_RESTORED -> FINAL_EVALUATED -> RECORDED
```

- The trainer receives `SupervisedTrainingView`, an algorithm, and optionally a
  `ValidationFeedbackService`. That service can evaluate only declared validation-role
  views and returns `ValidationMetricRecord` values. It contains no final-test handle.
- The ordinary selector accepts a `ValidationSelectionTable`, whose constructor accepts
  only validation records. It cannot accept a generic metric mapping or `FinalMetricRecord`.
- Candidate and hyperparameter search consume frozen validation summaries. They cannot
  construct evaluators or open dataset views.
- Tuning plus confirmation produce `FrozenCandidateSelection`, naming the scientific
  candidate. Validation within each fresh final-seed run produces
  `FrozenCheckpointSelection`, naming that run's selected epoch and checkpoint. A
  checkpoint restorer verifies the checkpoint and configuration/manifests and produces
  `RestoredCheckpoint`.
- The final evaluator requires a matching `FrozenCandidateSelection`,
  `FrozenCheckpointSelection`, and `RestoredCheckpoint`. Only then can the runner exchange
  its opaque `FinalTestHandle` for a final-test evaluation view. Final metrics are returned
  directly to result assembly, never back into the selector or search coordinator.
- Test-oracle evaluation is a separate diagnostic run kind with separate configuration,
  test-bearing diagnostic capability, selector, result type, and output namespace. It is
  permitted only where a protocol explicitly allows it and requires a conspicuous opt-in.
  Its selector may use the configured test diagnostic specifically because the result is
  an oracle envelope; it cannot populate ordinary selected-candidate or final-test fields.

The exclusions in the following capability trace are normative. Its concrete capability
and artifact names remain provisional:

| Stage | Input capability | Output | Who may consume output | Enforced exclusion |
| --- | --- | --- | --- | --- |
| Dataset construction | Source/artifact access | Internal `DatasetBundle` plus manifests | Runner capability broker | No model, selector, or tracker access |
| Representation | One approved split view | Inputs/features plus manifest | Trainer, pair/projection pipeline, or evaluator for that same role | No role widening or split relabeling |
| Pair construction | Training-only `PairSourceView`; oracle relation only for oracle builder | `PairSet` and diagnostics | Projection fitter and result provenance | No validation/final-test records or classifier state |
| Projection fit | Pair endpoint feature rows | Fitted transform and diagnostics | Runner input pipeline | No labels/metrics/optimizer |
| Training | Training view; optional validation feedback | Checkpoint candidates and histories | Checkpoint store, validation evaluator, selector | No final-test handle/metrics |
| Validation | Validation view plus checkpoint/model snapshot | Validation-only records | Epoch and candidate selectors | No final-test data or generic split dictionary |
| Candidate selection | Tuning/confirmation validation tables and checkpoint decisions | Finalists artifact, then `FrozenCandidateSelection` | Final-run resolver | No test/diagnostic/final-stage records; methods selected independently |
| Final-run checkpoint selection | Final-stage validation table for the frozen candidate | `FrozenCheckpointSelection` | Checkpoint restorer and final gate | Cannot alter candidate or hyperparameters |
| Restoration | Both frozen decisions plus checkpoint artifact | Verified `RestoredCheckpoint` | Final gate | Cannot silently use live last-epoch weights |
| Final evaluation | Matching frozen/restored tokens plus opaque final handle | Final metric records | Result assembler only | No feedback edge to training/search/selection |
| Diagnostic oracle | Explicit diagnostic config and allowed diagnostic capability | `CmnistTestOracleDiagnosticResult` | Diagnostic result namespace | Cannot be converted to ordinary selection result |
| Result production | Immutable outputs from preceding stages | Canonical local result and artifact index | User; optional tracking mirror | Tracker cannot alter or complete the result |

Pair builders receive only source records permitted by the protocol. Validation and final
test records are not filtered out after construction; they are absent from the capability.
Likewise, checkpoint restoration occurs before the final-test capability can be opened.

## 5. Pair contracts

**Classification:** the scientific pair permissions and oracle/test-oracle distinction are
approved protocol constraints. The following record fields and builder signatures are
provisional design guidance. Milestone 4 implements the CMNIST clean-oracle subset with a
training-source capability and canonical manifest; estimated builders, other datasets, and
generalized pair artifact storage remain deferred.

### Records, sets, and configuration

```text
PairRecord
  pair_id: str
  left_example_id: str
  right_example_id: str
  left_source_id: str | None
  right_source_id: str | None
  left_source_index: int
  right_source_index: int
  left_label: int
  right_label: int
  left_environment_or_domain: str | None
  right_environment_or_domain: str | None
  left_metadata_digest: str
  right_metadata_digest: str
  left_split_name: str
  right_split_name: str
  left_split_role: training
  right_split_role: training
  method: oracle | conditional_random | nearest
  orientation: str
  construction_key: str

PairSet
  pair_set_id: str
  records: Sequence[PairRecord]
  method_config: ResolvedPairBuilderConfig
  dataset_manifest_ref: ArtifactRef
  feature_manifest_ref: ArtifactRef | None
  construction_seed: int | None
  construction_algorithm_id: str
  diagnostics: PairValidationDiagnostics
  artifact_manifest: ArtifactManifest
```

`pair_id` is stable under serialization and derived from the dataset identity, method
identity, ordered endpoint identities, and construction key. It is not a row number alone.
`construction_key` records deterministic construction information when a random seed is
not sufficient, such as a source relation row or algorithm-version key.

The implemented CMNIST pair-source capability carries the validated dataset manifest, and
its constructor verifies the supplied training-pool content digest and partition membership
against that manifest. The clean-oracle builder has no caller-supplied dataset-digest
argument: it revalidates the capability and derives both the pair-manifest dependency and
stable pair IDs from the capability's canonical dataset identity. CMNIST feature-cache
preparation then checks that dependency, the pair-set/manifest record equality, every
record's source metadata, and the aligned clean red/green endpoint intervention before it
creates the output directory or invokes an encoder.

Left/right source indices are stable row indices in the pair-source-view manifest, not
unscoped raw dataset offsets. Protected semantic source IDs are populated only when the
builder capability permits them; endpoint example IDs and indices remain required for
every method.

`PairValidationDiagnostics` includes requested/realized counts, duplicate and self-pair
counts, invalid/missing index counts, label-agreement/disagreement counts, environment or
domain transition counts, source-role counts, distance summaries where applicable, and
manifest/hash validation. Protocol-specific validation may reject rather than merely log
violations, such as a non-training endpoint or a CMNIST clean pair whose label/content
identity differs.

The pair configuration is a strict discriminated union:

- `OraclePairBuilderConfig` names the approved oracle relation manifest, orientation,
  budget policy, and construction determinism.
- `ConditionalRandomPairBuilderConfig` requires an approved `definition_id`, conditioning
  fields, candidate-pool rules, replacement policy, distance/tie policy if any, and pair-
  seed assignment name.
- `NearestPairBuilderConfig` requires an approved `definition_id`, search representation,
  candidate-pool restrictions, metric, neighbor/tie policy, and determinism fields.

The latter two structures support the eventual methods without selecting their still-open
scientific definitions. An unresolved or unregistered `definition_id` is a validation
error, not a default.

The discriminated dataset payload adds already-approved invariants without filling in open
method definitions:

- CMNIST primary oracle config requires exactly 256 unique training sources sampled
  without replacement at the fixed pair seed (or one prespecified 32/64/128/256/512
  sensitivity budget). Every pair has the same source/content, digit, clean label, and
  noisy target, with opposite color endpoints. Pair budget is not validation-selected.
- Waterbirds primary oracle config requires exactly 240 training relations, stratified as
  184 landbird and 56 waterbird sources. Endpoints share source bird, label, foreground,
  mask, crop, scale, and placement and differ only in canonical background; orientation is
  always land minus water. Smaller label-stratified subsets are separately labeled
  pair-budget sensitivities, never validation-selected replacements for the primary.
- Waterbirds conditional/random and nearest configs both require training-only,
  same-bird-label, opposite-background eligibility and exactly 240 primary pairs. Their
  exact sampling, distance, reuse, replacement, and tie definitions remain unresolved
  unless the protocol has already fixed a field.

### Builder interfaces and permissions

```text
PairBuilder.build(
    sources: PairSourceView,
    config: ConditionalRandomPairBuilderConfig | NearestPairBuilderConfig,
    context: PairBuildContext,
) -> PairSet

OraclePairBuilder.build(
    sources: PairSourceView,
    relation: OraclePairRelationView,
    config: OraclePairBuilderConfig,
    context: PairBuildContext,
) -> PairSet
```

`PairBuildContext` contains explicit RNG/determinism information and artifact writers, not
models, loaders, evaluators, or tracking clients. Feature-dependent builders may request a
training-only `FeatureTable` through the source view; the feature manifest then becomes a
required pair-set dependency.

For Waterbirds-CF, the supervised training view contains all 4,795 records for ERM and
every GRIT variant. The separate relation capability contains exactly the 240 approved
oracle identities and is supplied only to `OraclePairBuilder`. Estimated methods may use
the endpoint records because they are ordinary supervised data, but cannot learn which 240
relationships are oracle pairs. For CMNIST, the oracle builder can access clean recolored
training-source endpoint views; the classifier training view cannot.

Waterbirds oracle records orient each pair as land minus water, regardless of which
endpoint was the existing majority record, and record the bird label, land/water metadata,
source CUB identity, and reconstruction-manifest relation. CMNIST records use one declared
color orientation consistently. Reversing either orientation changes the pair-set identity,
even though the fitted nuisance subspace should be invariant to a global sign change.

## 6. Linear nuisance-projection contracts

**Classification:** classifier-independent projection, uncentered differences, rank-zero
identity, deterministic full SVD, and initial CPU-float64 fitting are approved scientific/
numerical constraints. Milestone 4 implements fitting/transformation and canonical
diagnostics for CMNIST. The API remains internal, and final basis persistence/artifact
format selection remains provisional.

The projection component is a fitted input transform independent of classifiers,
algorithms, trainers, and evaluators. The runner fits it before model training and inserts
the same fitted transform in every train, validation, final-test, and allowed diagnostic
inference path. This explicitly corrects the inherited ECMP mismatch, where training
projects inputs but inherited evaluation calls the model on unprojected inputs
(`../solver/ecmp.py`, lines 9-47; `../solver/erm.py`, lines 90-114).

Conceptual operations are:

```text
PairDifferenceInput
  left: array[P, D]
  right: array[P, D]
  pair_set_id: str
  feature_manifest_id: str

LinearProjectionFitter.fit(
    pairs: PairDifferenceInput,
    config: ResolvedProjectionConfig,
) -> FittedLinearProjection

FittedLinearProjection.transform(inputs: array[..., D]) -> array[..., D]
FittedLinearProjection.diagnostics() -> ProjectionDiagnostics
```

The fit contract forms uncentered row differences `left - right` with shape `[P, D]`.
`center_differences` must be explicitly false for the approved protocols. It performs the
non-randomized full `torch.linalg.svd` operation required by both protocols, never
`torch.svd_lowrank`, under a recorded PyTorch version, dtype, device, deterministic mode,
and tolerance policy. Requested rank is the configured removal rank. Numerical rank is the
count defined by the serialized tolerance rule. Effective rank is
`min(requested_rank, numerical_rank)` after validating that requested rank is nonnegative
and no larger than the algebraically permitted dimension.

The nuisance basis is the first `effective_rank` right singular vectors in deterministic
descending singular-value order. Transformation removes their span: for a row feature
matrix `X`, return `X @ (I - V @ V.T)`, where `V` contains those basis vectors. Equivalent
implementations may avoid materializing the square projector but must satisfy the same
contract and diagnostics.

Rank zero is an exact identity behavior: fitting may still emit diagnostics, but transform
must return values equal to the input without a numerical projection multiply. Input and
output shapes match, including leading batch dimensions. The feature dimension must match
the fitted artifact. Inputs must satisfy the serialized dtype/device conversion policy;
implicit device transfer or unrecorded precision narrowing is forbidden. The initial
implementation uses deterministic CPU float64 fitting followed by an explicit recorded
runtime conversion; the CMNIST slice must validate that choice before broader reuse.

`ProjectionDiagnostics` contains requested, numerical, and effective ranks; the complete
singular-value spectrum; absolute and relative thresholds; discarded/retained energy;
pair/feature dimensions; nonfinite-input checks; orthonormality, symmetry, and idempotence
residuals; backend/dtype/device; and pair/feature manifest identities. A zero numerical
rank is valid and yields identity even when requested rank is positive, while retaining the
requested/effective distinction in results.

For the implemented CMNIST fitter, the exact pair-manifest and feature-cache-manifest
digests are required fit inputs and required canonical diagnostic fields. The runner passes
the identities of the validated pair set and loaded cache used to produce the endpoint
feature matrices; diagnostic JSON round trips retain both identities.

Serialization separates lightweight metadata from tensor-heavy artifacts. Canonical JSON
contains config, dimensions, ranks, diagnostics, hashes, and `ArtifactRef` values. The
removed right-singular-vector basis with shape `[D, effective_rank]` is the canonical
numerical payload; the square projector is derived and is not independently authoritative.
The complete singular-value spectrum is required, either inline when demonstrably small or
through its own artifact reference—it is never omitted as “optional.” Numerical payloads
could eventually use a non-pickle artifact such as versioned NPZ. Loading should verify
hash, shape, dtype, feature manifest, and pair-set identity before producing a fitted
transform. No container is selected in Milestone 3; this paragraph is deferred guidance
behind a small replaceable serialization interface.

## 7. Algorithm, model, and training contracts

**Classification:** algorithm-owned bounded updates and trainer-owned iteration/lifecycle
are approved core. The interface sketch is provisional and should be reduced to what the
CMNIST ERM/oracle-GRIT slice proves necessary. Later-method hooks and full training resume
are deferred and must not influence the initial interface.

### Provisional compositional interface inventory

The runner may construct a model through a small factory and optimizer service(s) from
explicit configuration, and injects them into an algorithm factory. Once constructed, the
algorithm owns training-time model mutation, optimizer stepping, and its algorithm-specific
state. The trainer owns iteration and lifecycle control. This division supports ERM and
GRIT immediately without assuming every method has one scalar loss and exactly one
optimizer step.

Milestone 3 needed only a fake bounded update. The CMNIST slice has now chosen the smallest
actual boundary: one concrete linear-probe algorithm owns its two-class model, Adam,
optional fitted projection, update, prediction, and inference state; one trainer owns
epoch/batch iteration, validation after every epoch, and checkpoint capture. The broader
protocol below remains provisional and may still be renamed or reduced.

```text
Algorithm
  batch_request() -> BatchRequest
  begin_epoch(context: EpochContext) -> None
  training_step(input: TrainingStepInput) -> StepResult
  on_validation(feedback: ValidationFeedback) -> HookResult
  end_epoch(context: EpochContext) -> EpochResult
  predict(batch: InferenceBatch) -> Predictions
  checkpoint_state() -> AlgorithmCheckpointState
  load_checkpoint_state(state: AlgorithmCheckpointState) -> None
```

`TrainingStepInput` carries a typed batch, epoch/update counters, and approved runtime
services. The algorithm may perform zero, one, or multiple optimizer steps and may replace
model parameters, but it reports the number of updates, losses, and state changes in
`StepResult`. `BatchRequest` declares ordinary, group-uniform, or environment-structured
sampling plus required metadata; it is validated against the training capability before
iteration starts. ERM and GRIT use the same ordinary classification update, with GRIT's
already-fitted transform in the input pipeline.

If later required, `on_validation` would receive only `ValidationFeedback`, never evaluator or
loader access. It permits step-level behavior such as a future SWAD implementation while
maintaining the data boundary. `predict` cannot mutate training state. The evaluator owns
the transition into evaluation mode and restores the prior mode even on failure. Exact
autocast, distributed, scheduler, and compilation hooks remain provisional.

Model ownership is explicit:

- `ModelFactory` validates input dimension/representation and label space and returns a
  model with named `represent` and `predict_logits` capabilities where supported.
- The algorithm owns the injected live model during training and exposes it only through
  checkpoint and inference protocols.
- Algorithm state is distinct from model state. Optimizer state is distinct again, even
  when serialized through one checkpoint envelope.
- Algorithms needing intermediate representations request a declared model capability;
  they do not reach into dataset-specific subclasses or assume a `.network` attribute.

The earlier proposal used `AlgorithmCheckpointState` as a typed envelope with separately
named model state,
optimizer-state mapping, and opaque method-state payload; it is not one flattened state
dictionary. The algorithm supplies/restores it because the algorithm owns optimizer
stepping and may replace parameters, while the checkpoint coordinator supplies artifact
storage, identity validation, and trainer/RNG state around it.

That full envelope is deferred. Initial Milestone 3 checkpoint work records only stable
checkpoint identity and proves inference-state restoration with a fake model. Optimizer,
scheduler, trainer, algorithm, and RNG resumability must wait for an observed vertical-
slice need.

The trainer owns epoch/update loops, batch-provider coordination, validation cadence,
counter advancement, cancellation/failure handling, checkpoint requests, and training
history. It does not calculate algorithm-specific penalties or assume one optimizer step.
The runner owns component construction, capability issuance, multi-run search stages,
selection, checkpoint restoration, final evaluation, provenance, and result assembly.

### Evidence retained for deferred future needs

The following requirements were verified directly from inherited code:

- Fish performs multiple inner optimizer steps and then interpolates/replaces model state
  (`../solver/fish.py`, lines 26-75), so `loss(batch) -> scalar` is insufficient.
- GroupDRO maintains adversarial group weights and needs stable group metadata
  (`../solver/groupdro.py`, lines 32-93).
- IRM maintains an update counter and annealed penalty state; REx substitutes a different
  penalty (`../solver/irm.py`, lines 15-91).
- SWAD validates after training batches, maintains averaging/counter state, may stop early,
  and replaces live weights (`../solver/swad.py`, lines 28-115), so epoch-only hooks and
  model-only checkpoints are insufficient.
- MatchDG accesses a named feature representation in addition to logits
  (`../solver/matchdg.py`, lines 152-169).
- The inherited path has no persistent checkpoint save/restore API; its best metric logs do
  not restore best-epoch model/optimizer state.

It is not verified that inherited Fish optimizer-state handling, IRM half-batch environment
semantics, or SWAD validation behavior is scientifically correct. Exact resumability state,
scheduler/AMP/distributed behavior, raw-image model APIs, and MatchDG/LISA interfaces are
provisional. They should not expand the ERM/GRIT minimum implementation until their own
protocols are approved. These observations explain why the initial boundary must remain
revisable; they do not require Fish-, SWAD-, MatchDG-, LISA-, GroupDRO-, IRM-, or REx-
specific hooks in Milestone 3.

## 8. Selection and checkpoint contracts

**Classification:** validation-only selector inputs, deterministic ordering, separate
candidate/checkpoint decisions, earlier-epoch tie-breaking, restoration before final test,
and the CMNIST union-of-finalists rule are approved core. Exact record names and audit field
sets are provisional. Persistent resume envelopes and universal checkpoint storage are
deferred.

### Provisional names for approved decisions

```text
ValidationMetricRecord
  run_id: str
  candidate_id: str
  seed_stage: tuning | confirmation | final
  seed: int
  checkpoint_id: str
  epoch: int
  split_name: ValidationSplitName
  metric_id: str
  direction: maximize | minimize
  value: finite float
  group_values: Mapping[GroupId, finite float] | None
  group_sample_counts: Mapping[GroupId, int] | None
  sample_count: int
  aggregation: MetricAggregationRecord
  manifest_ids: MetricInputManifests

MetricAggregationRecord
  reduction_id: str
  required_group_ids: tuple[GroupId, ...]
  group_weights: Mapping[GroupId, finite float] | None
  weights_source_manifest_digest: str | None

CheckpointSelectionDecision
  selector_id: str
  objective_value: float
  tie_break_values: tuple[TypedTieValue, ...]
  run_id: str
  candidate_id: str
  epoch: int
  checkpoint_id: str
  contributing_record_ids: tuple[str, ...]
  audit_trace: SelectionAuditTrace

CandidateSelectionDecision
  selector_id: str
  method_id: str
  objective_value: float
  tie_break_values: tuple[TypedTieValue, ...]
  candidate_id: str
  contributing_checkpoint_decision_ids: tuple[str, ...]
  stage_seed_sets: Mapping[SeedStage, tuple[int, ...]]
  audit_trace: SelectionAuditTrace

TuningFinalistsArtifact
  method_id: str
  selector_id: str
  ordered_candidate_ids: tuple[str, str, str]
  contributing_checkpoint_decision_ids: tuple[str, ...]
  tuning_seed_set: tuple[int, ...]
  aggregation_order: str

FrozenCandidateSelection
  method_id: str
  candidate_id: str
  finalists_artifact_id: str
  tuning_and_confirmation_decision: CandidateSelectionDecision
  frozen_scientific_config_digest: str
  frozen_at: timestamp

FrozenCheckpointSelection
  candidate_selection_id: str
  run_id: str
  seed_stage: final
  epoch_decision: CheckpointSelectionDecision
  checkpoint_id: str
```

Directions are explicit; values are finite; group coverage is validated against the
declared `GroupSpec`. Average and worst-group metrics are separate typed reductions, not
ambiguous field names. A selector defines a total deterministic order: primary objective,
ordered tie breakers, then a stable candidate/checkpoint identity as the final fallback.
Floating comparisons use a serialized exact or tolerance policy.

Group metrics include per-group sample counts. Waterbirds adjusted-average records require
all four groups and embed the normalized weights plus the validated Waterbirds-CF training-
manifest digest from `AdjustedAverageAggregationSpec`; they cannot derive weights from the
evaluation split.

A checkpoint decision reduces split/group records at one epoch for one run. A candidate
decision first takes the configured checkpoint decision independently for every seed, then
sets the candidate score to the arithmetic mean of those already-computed seed-level
selector scores. It never pools examples across seeds, applies the minimum/worst reducer
after averaging component metrics, or chooses the best individual seed. Its audit trace
stores both reduction layers.

Epoch selection operates within one candidate/seed and returns an actual checkpoint ID.
Candidate selection aggregates the selected validation outcomes across a declared seed
stage. The search coordinator cannot aggregate a tuning checkpoint that was never
persisted. Tuning writes a durable `TuningFinalistsArtifact` containing the ordered top
three for one method and selector. Confirmation accepts only the declared fresh-seed
records and reuses the tuning checkpoint decisions stored in that artifact; callers cannot
replace the tuning evidence. It does not revisit test results. The combined five-seed
decision produces a durable `FrozenCandidateSelection`. Final runs use fresh final seeds
and that frozen scientific configuration. Each final-seed run then creates a
`FrozenCheckpointSelection` using validation only. The reported final metrics come from
those restored selected checkpoints, never implicitly from the last live weights.

Validation records from tuning and confirmation stages may contribute to candidate
selection according to the configured stage: tuning records contribute only through the
persisted finalist artifact, while the confirmation boundary accepts only confirmation
records. A final-stage validation record may select the epoch/checkpoint within that
already-frozen candidate and final seed, but is rejected by every candidate or
hyperparameter selector. Stage-specific table constructors enforce that distinction.

A future resumable checkpoint envelope may contain:

- checkpoint/schema identity, run/candidate/config identities, epoch/update counters, and
  creation reason;
- model parameters and buffers, optimizer/scheduler state, algorithm-specific tensors and
  counters, trainer state, and the configured RNG-state coverage;
- dataset, feature, pair, and projection manifest identities; code/environment provenance;
- content hashes and `ArtifactRef` values for each tensor-heavy payload.

Initial restoration validates checkpoint identity and proves faithful inference-state
restoration. The remaining envelope fields and resume guarantees above are deferred. If a
later checkpoint is advertised as resumable, it must then round-trip every declared mutable
state and RNG source.

### Protocol mappings

For CMNIST, the primary selector maximizes the minimum accuracy across `val_e01`,
`val_e02`, and `val_e05`; ties prefer higher mean accuracy, then lower projection rank,
then stable configuration order. The secondary source selector uses the minimum of
`val_e01` and `val_e02`, with the same higher-mean, lower-rank, stable-order ties. Earlier
epoch is the final semantic checkpoint tie-break for otherwise identical checkpoint
records before stable checkpoint identity. This makes epoch selection total without
changing the approved candidate tie sequence.

Both CMNIST selector branches consume the same three-seed validation records, write their
own ordered top-three artifact, and freeze their own five-seed winner. Confirmation runs
the union of those finalist candidate IDs, so a candidate appearing in both branches is not
duplicated; its fresh records may contribute to both prespecified selectors.

The implemented Milestone 3 boundary enforces this as connected typed artifacts. Tuning
ranking accepts exactly the configured three tuning seeds and no other stage.
`TuningFinalistsArtifact` contains one method/selector's deterministic ordered top three.
The confirmation union embeds the primary and secondary artifacts and emits their ordered
candidate-ID union without duplication. Each five-seed comparison accepts exactly its own
artifact's three candidates, reuses its stored decisions for the configured three tuning
seeds, and accepts new records only for the two confirmation seeds; the artifact is then
embedded in `FrozenCandidateSelection`. Missing, duplicate, extra, or
mis-staged seeds, cross-method records, and candidates outside that selector's artifact are
rejected without adding a scheduler.

For Waterbirds-CF, selection maximizes official-validation worst-group accuracy; ties
prefer higher adjusted average accuracy using validated Waterbirds-CF training-manifest
weights, then lower projection rank when rank is actually a candidate axis, earlier epoch
for checkpoint selection, and stable configuration order.

Both protocols apply the search independently for each method and selector: use three
tuning seeds, retain that method's top three candidates, add two distinct confirmation
seeds, choose its combined five-seed validation winner, and evaluate it on ten fresh final
seeds. Methods are never candidates in one another's selector. Shared final seeds support
paired comparisons after selection; they do not select a method. Exact seed values remain
protocol/config data; the contract enforces disjointness and stage labels.

CMNIST test-oracle envelopes use `CmnistTestOracleSelectionConfig`,
`DiagnosticMetricRecord`, and `CmnistTestOracleDiagnosticResult`, with an output namespace
such as `diagnostics/test_oracle/`. They are not accepted by
`ValidationSelectionTable` and cannot become a frozen candidate or checkpoint selection.
The Waterbirds initial study rejects test-oracle selection configuration entirely.

The implemented diagnostic metric is one eligible oracle trial and retains its record,
run, candidate, method, scientific-configuration digest, projection rank, checkpoint,
epoch, seed, and `test_ood` accuracy identity. A diagnostic-only selector chooses the
maximum accuracy over all eligible records. Exact ties use the documented
`stable_trial_identity` lexical fallback over method, candidate, scientific-config digest,
projection rank (`None` before integer ranks), run, checkpoint, epoch, seed, and record ID;
they do not consult an ordinary selector. Its decision records every eligible contributing
record ID and the complete selected-trial identity. The diagnostic result's `run_id`
identifies the envelope computation, not every eligible trial: trial
run/configuration/checkpoint identities are intentionally allowed to differ. Result
validation recomputes the decision from the supplied envelope and rejects inconsistent
candidate, run, checkpoint, contributor, or winner identities.

## 9. Results, provenance, and tracking

**Classification:** separate ordinary/test-oracle Pydantic result types, canonical local
JSON, round trips, and a null event sink are approved core. The complete provenance field
inventory is provisional. Content-addressed storage, universal tensor containers,
production histories, generalized artifact transactions, and W&B integration are
deferred. Milestone 6A's narrow atomic per-run staging/publication boundary is implemented
without implying any broader storage framework.

Canonical local JSON is authoritative. Results form a strict discriminated union; the
initial implementation includes only fields needed by the synthetic spine and CMNIST
ERM/oracle-GRIT configuration, selection, restoration, metrics, and status. The longer
inventory below is provisional:

```text
RunResult = OrdinaryRunResult | CmnistTestOracleDiagnosticResult

CommonRunResultFields
  schema_version: "grit.run-result/v1"
  run_id, candidate_spec_id, experiment_name, protocol_id
  candidate_id: str | None
  status: planned | running | succeeded | failed | interrupted
  failure: FailureRecord | None
  resolved_configuration: canonical object plus digest
  code_provenance: git revision, branch, dirty state/diff digest
  environment_provenance: Python, dependency lock, OS, device, accelerator/driver
  determinism_provenance: all seeds and backend flags
  dataset_manifest: DatasetManifestRef | None
  feature_manifest: FeatureManifestRef | None
  pair_manifest: PairManifestRef | None
  projection_manifest: ProjectionManifestRef | None
  training_history: HistoryRef | None

OrdinaryRunResult(CommonRunResultFields)
  result_kind: ordinary
  run_stage: tuning | confirmation | final
  validation_history: validation-only history artifact plus summary
  epoch_decision: CheckpointSelectionDecision | None
  candidate_selection: FrozenCandidateSelection | None
  checkpoint_selection: FrozenCheckpointSelection | None
  selected_checkpoint: CheckpointRef | None
  restoration_receipt: RestorationReceipt | None
  final_test_metrics: tuple[FinalMetricRecord, ...] | None
  diagnostic_results: tuple[DiagnosticResultRef, ...]

CmnistTestOracleDiagnosticResult(CommonRunResultFields)
  result_kind: cmnist_test_oracle_diagnostic
  diagnostic_metrics: tuple[DiagnosticMetricRecord, ...]
  oracle_decision: selected trial identity, maximum accuracy, stable tie policy,
                   and every eligible contributing record ID

Both variants
  timing: phase timestamps, durations, and clock policy
  artifacts: ArtifactManifest
```

Failure records include phase, stable exception category, sanitized message, traceback
artifact reference where retained, and the last completed lifecycle state. Partial results
must never present missing selection or final metrics as zero. An ordinary final-stage
success requires the frozen-candidate reference, per-run frozen checkpoint, selected
checkpoint, matching restoration receipt, and final metrics. Tuning/confirmation results
cannot populate final metrics and require their epoch decision when training succeeds.
Manifest and history references may be absent only when failure occurred before their
phase; the status/phase validator enforces this. ERM requires `pair_manifest` and
`projection_manifest` to be `null`; successful GRIT runs require both. Ordinary
`diagnostic_results` are reporting-only records supplied after the relevant gate and never
selector input. The test-oracle variant has no ordinary selection or final-test field, so
oracle data cannot populate one accidentally.

Future local output may use an atomic run directory, an incomplete marker, referenced
artifacts, and a manifest before committing canonical JSON last. Histories may eventually
use JSON Lines or a typed columnar format. These storage mechanics and a generalized
content-addressed `ArtifactRef` framework are deferred. Initial large or opaque values sit
behind a small replaceable reference boundary; arbitrary pickled Python objects are not
declared a portable contract.

Every boundary object must round-trip `typed object -> canonical serialization -> typed
object` without losing enum values, ordering with semantic meaning, identities, numeric
types promised by the schema, or optional-versus-absent distinctions. Round-trip tests also
recompute content hashes. Readers reject unsupported major schema versions; minor additive
migration, if later adopted, must be explicit and tested.

Milestone 3 implements only `NullEventSink` (or an equivalent internal name) so experiment
semantics do not depend on tracking. A future `WandbMirrorSink` may receive immutable
lifecycle, metric, artifact, and completed-result events. Production W&B integration,
failure policy, and artifact mirroring are deferred. W&B may never inject a sweep
candidate, rename split roles, return metrics to selectors, choose a checkpoint, or become
the only authoritative record.

## 10. Non-executable protocol examples

**Classification:** the split access and selection behavior shown here is approved
scientific contract. Concrete configuration/type spelling is illustrative and revisable.

These sketches show resolved relationships, not YAML syntax or executable configuration.

### CMNIST ERM

```text
dataset:
  source partitions = train_e01_sources, train_e02_sources
  supervised training splits = rendered train_e01 + train_e02
  validation = val_e01, val_e02, val_e05 (shared source IDs, distinct views)
  final_test = test_ood
representation = frozen OpenAI CLIP ViT-B/32, normalization none
pairs = disabled; projection = disabled; algorithm = ERM
selector = robust minimum(val_e01, val_e02, val_e05)
access:
  trainer -> supervised training view
  selector -> validation records only
  final evaluator -> test_ood only after candidate/checkpoint freezes and restoration
```

The source selector is an explicitly labeled secondary selection branch over the same
saved validation/checkpoint records. It may freeze a different candidate and therefore
produces separately labeled final results. The separately enabled test-oracle diagnostic
is a different root config/result type and never changes either ordinary ERM winner.

### CMNIST oracle GRIT

```text
supervised training and validation/final roles = same as CMNIST ERM
oracle pair source = clean recolor views of approved training sources only
oracle pair budget = primary 256; pair identities available only to oracle builder
projection = uncentered differences, deterministic full SVD, resolved rank in 0..24
algorithm = ordinary classification update over consistently projected features
selector = same robust validation selector as ERM, with lower-rank tie break
```

Pair endpoints are projection evidence, not extra supervised examples. Neither pair builder
nor projection fitter sees validation or `test_ood`.

### Waterbirds-CF ERM

```text
supervised training = all 4,795 CF records, including every counterfactual endpoint
training metadata = labels plus ERM-approved fields; no background label as a signal
validation = official validation; final_test = official test
representation = frozen OpenAI CLIP ViT-B/32, normalization none
pairs = disabled; projection = disabled; algorithm = ERM
selector = validation worst-group accuracy, then adjusted average weighted by the
           Waterbirds-CF training manifest, then earlier checkpoint
access:
  no trainer, algorithm, estimated builder, or selector receives oracle pair relations
  official test opens only after candidate/checkpoint freezes and restoration
```

The original-Waterbirds ERM control has its own dataset/protocol label and cannot be merged
with this result.

### Waterbirds-CF oracle GRIT

```text
supervised training = the same 4,795 records visible to every method
oracle relation capability = exactly 184 landbird + 56 waterbird training identities
oracle pair builder = sole consumer of that relation capability
projection = fitted from land-minus-water endpoints; classifier sees projected records
projection rank = one resolved candidate in 0..24, including rank-zero identity
validation/final roles and selector = identical to Waterbirds-CF ERM
final reporting = adjusted average, raw average, worst group, and all four group accuracies
```

Estimated builders can see the supervised endpoint records but not the oracle relation.
No Waterbirds test-oracle selector exists in the initial study.

## 11. Synthetic contract-test plan

### Required initial Milestone 3 tests

The following small test spine is the Milestone 3 exit criterion and is now covered by the
focused synthetic suite (test function spelling was allowed to remain internal):

| Proposed test | Behavior proved |
| --- | --- |
| `test_cmnist_boundary_config_is_strict_and_round_trips_canonical_json` | The minimum ERM/oracle-GRIT Pydantic config rejects unknown fields and round-trips canonical JSON. |
| `test_split_roles_issue_noninterchangeable_views` | Training, validation, final-test, and diagnostic views cannot be substituted or widened. |
| `test_validation_selector_rejects_final_and_diagnostic_metrics` | The ordinary selector accepts only validation metric records by construction. |
| `test_checkpoint_ties_prefer_earlier_epoch_then_stable_identity` | The approved total checkpoint order is deterministic. |
| `test_candidate_and_checkpoint_freezes_are_distinct` | Final-run validation may select an epoch but cannot change frozen hyperparameters. |
| `test_cmnist_selector_finalist_union_retains_separate_winners` | Primary/secondary finalist union is confirmed once while decisions remain separate. |
| `test_fake_checkpoint_restoration_recovers_selected_inference_state` | Minimal checkpoint identity restores selected fake state rather than last state. |
| `test_ordinary_and_test_oracle_results_are_discriminated_round_trips` | Ordinary and CMNIST test-oracle result types cannot be confused and serialize canonically. |
| `test_test_oracle_envelope_selects_across_trials_and_round_trips` | Distinct configurations, ranks, runs, and checkpoints remain eligible; maximum accuracy plus stable identity selects the traced diagnostic winner. |
| `test_tuning_finalists_gate_confirmation_and_freeze` | Exactly three tuning seeds produce each selector's ordered top three; only those candidates may enter its exact five-seed comparison and freeze. |
| `test_null_event_sink_has_no_semantic_effect` | Tracking absence cannot change selection or results. |
| `test_in_memory_lifecycle_blocks_final_metric_feedback` | Validation selects a fake checkpoint, restoration occurs before final evaluation, and final metrics cannot flow back to training or either ordinary selector. |

These tests use in-memory fixtures and fake state only. They do not require datasets,
PyTorch, feature caches, pair builders, projection mathematics, filesystem artifact
stores, W&B, or faithful training resume.

### Provisional and deferred validation inventory

The larger inventory below preserves requirements for CMNIST, Waterbirds, and later
production components. It is not an initial Milestone 3 checklist. Entries that overlap
the core spine may be reused, but artifact, projection, pair, Waterbirds, tracking, and
resumability tests become required only with the component they validate.

| Proposed test | Behavior proved |
| --- | --- |
| `test_experiment_config_round_trip_is_canonical` | A fully populated resolved config round-trips and produces identical canonical bytes/digest. |
| `test_config_rejects_unknown_nested_field` | Unknown keys fail at every nesting level rather than being ignored. |
| `test_config_rejects_unsupported_schema_version` | Unsupported versions require an explicit migration instead of best-effort loading. |
| `test_path_override_is_allowlisted_recorded_and_content_bound` | Only approved path fields can be overridden; locations are recorded, the same artifact keeps identity, and different content changes the manifest-bound candidate ID. |
| `test_config_rejects_inconsistent_erm_projection` | ERM plus enabled pairs/projection fails cross-field validation. |
| `test_config_rejects_unapproved_pair_definition` | Conditional/random and nearest configs cannot resolve without a registered approved definition ID. |
| `test_config_rejects_overlapping_seed_stages` | Tuning, confirmation, and final seed sets must be disjoint. |
| `test_split_role_capabilities_are_not_interchangeable` | Training, validation, final-test, and diagnostic views cannot be substituted or role-widened. |
| `test_selector_constructor_rejects_final_metric_record` | Ordinary selector input structurally excludes final-test metrics. |
| `test_final_test_handle_requires_frozen_restored_match` | Test access fails before selection/restoration and when checkpoint/config identities differ. |
| `test_final_stage_validation_can_select_checkpoint_but_not_candidate` | A final-seed validation table can freeze an epoch for the already-selected candidate but cannot change hyperparameters. |
| `test_search_writes_top_three_then_freezes_combined_five_seed_winner` | Three tuning seeds create a durable ordered top-three artifact; two fresh confirmation seeds create the five-seed candidate freeze. |
| `test_ten_final_seeds_require_frozen_candidate_and_individual_checkpoints` | Every fresh final seed uses the same candidate and must freeze/restore its own validation-selected checkpoint before test access. |
| `test_candidate_score_means_seed_level_selector_scores` | Search averages already-reduced seed selector scores rather than reducing averaged component metrics. |
| `test_selector_tie_break_is_total_and_stable` | Equal primary values follow mean/rank/epoch/stable-ID rules deterministically. |
| `test_checkpoint_restoration_recovers_selected_not_last_epoch` | Final inference loads the selected checkpoint and yields its predictions, not live last-epoch weights. |
| `test_resumable_checkpoint_round_trips_algorithm_optimizer_and_rng_state` | Model, optimizer, opaque algorithm state, counters, and declared RNG state restore exactly. |
| `test_pair_set_rejects_nontraining_or_invalid_indices` | Missing/out-of-range endpoints and non-training split roles are rejected. |
| `test_pair_identity_and_manifest_are_stable` | Equivalent inputs/seed/config produce stable pair and manifest identities with provenance intact. |
| `test_oracle_pair_builder_requires_relation_capability` | An ordinary/estimated builder cannot receive or infer oracle relations. |
| `test_cmnist_oracle_pairs_are_unique_clean_training_recolors` | Primary oracle has 256 without-replacement training sources with shared content/clean/noisy labels and opposite colors. |
| `test_waterbirds_oracle_pairs_are_184_56_and_land_minus_water` | Oracle manifest has the approved strata, shared foreground/source invariants, and canonical orientation. |
| `test_waterbirds_estimated_pair_eligibility_is_approved_but_definition_required` | Estimated pairs are training-only, same-label, opposite-background, and 240 in the primary config without inventing sampling/distance details. |
| `test_projection_preserves_shape_and_rank_zero_is_exact_identity` | `[P,D]` fit inputs and arbitrary `[...,D]` transforms obey shape rules; rank zero bypasses multiplication. |
| `test_projection_records_requested_numerical_and_effective_rank` | Rank truncation and a numerically rank-zero matrix remain explicit and reproducible. |
| `test_projection_rejects_rank_above_pair_or_feature_dimension` | Requested rank greater than `min(P,D)` fails before fitting. |
| `test_projection_uses_uncentered_left_minus_right_full_svd` | Fit input and required non-randomized `torch.linalg.svd` semantics cannot silently center, reverse, or use randomized low rank. |
| `test_projection_persists_basis_and_complete_spectrum` | Canonical basis and all singular values survive artifact round trip. |
| `test_projection_artifact_round_trip_verifies_hash_shape_and_dtype` | Metadata plus referenced array artifacts reload exactly and reject tampering. |
| `test_run_result_union_round_trip_preserves_failure_and_optional_fields` | Successful and failed ordinary/diagnostic variants round-trip without conflating absent values with zero/empty data. |
| `test_test_oracle_config_metrics_and_result_cannot_enter_ordinary_types` | CMNIST oracle diagnostics retain final-test provenance but cannot construct ordinary selection/result objects. |
| `test_tensor_heavy_results_use_artifact_references` | Checkpoints/features/projections cannot be embedded in canonical JSON. |
| `test_null_tracking_sink_has_no_semantic_effect` | Tracking disabled produces the same config identity, selection, and local result semantics. |
| `test_tracking_failure_cannot_change_selection` | A failing mirror follows its operational policy but cannot alter candidates/checkpoints/metrics. |
| `test_cmnist_validation_views_share_ordered_source_identities` | Three validation renderings have distinct example/view IDs and the exact same ordered source IDs; mismatch is rejected. |
| `test_cmnist_source_partitions_are_disjoint_and_views_share_labels_content` | Training/validation source membership is disjoint while repeated validation views preserve digit, clean/noisy label, and source content. |
| `test_feature_manifest_rejects_normalization_mixing` | Pair, projection, training, and evaluation artifacts cannot mix primary unnormalized and L2 sensitivity features. |
| `test_waterbirds_training_view_contains_all_counterfactual_endpoints` | Every method receives all 4,795 supervised records, including both endpoints. |
| `test_waterbirds_erm_training_metadata_redacts_background_pair_and_source_relation_fields` | ERM gets labels/endpoints but no explicit background signal, pair IDs, shared CUB source identity, or oracle relation. |
| `test_waterbirds_oracle_relations_are_visible_only_to_oracle_builder` | ERM and estimated builders cannot access the 240 identities while the oracle builder can. |
| `test_waterbirds_manifest_enforces_4795_and_four_group_counts` | Training manifest validates 4,795 records and the exact 3498/184/56/1057 group counts. |
| `test_waterbirds_adjusted_average_uses_training_manifest_weights` | All four evaluation groups are required and adjusted average uses training-manifest rather than evaluation-sample proportions. |
| `test_waterbirds_ordinary_config_rejects_test_oracle_selection` | The initial Waterbirds protocol cannot enable diagnostic test-oracle selection. |

Milestone 4 adds separate concrete CMNIST tests for exact source/view counts, partition
apportionment and ordering invariance, construction, oracle pairs, projection numerics,
feature-cache validation, rank-zero equivalence, and the gated end-to-end lifecycle.
Milestone 5 adds concrete Waterbirds-CF construction, group, feature, projection,
selection, final-gate, result, and smoke tests. Real source-asset execution remains
deferred; the hermetic fixtures are not reportable data.

## 12. Protocol trace and self-review

**Classification:** the information-flow and selection restrictions in this trace are
approved safeguards. CMNIST rows guide the first vertical slice. Waterbirds rows are
retained acceptance criteria for that later slice, not Milestone 3 implementation scope.

This trace is the checklist for reviewing the proposal against the approved protocols:

| Protocol item | Contract path | Selection or access result |
| --- | --- | --- |
| CMNIST 25,000-source `train_e01_sources` and `train_e02_sources` -> `train_e01`, `train_e02` | Source partitions -> rendered training-role splits -> supervised training view | Trainer sees rendered splits; pair builder separately receives approved training sources/renderings |
| CMNIST 10,000 `validation_sources` rendered as `val_e01/e02/e05` | One source partition, three view IDs, ordered shared source IDs -> validation views | Primary selector gets three accuracies; secondary source selector gets e01/e02 only |
| CMNIST 10,000 official-test `test_sources` rendered as `test_ood` | Opaque final-test handle | Opens only after five-seed candidate selection is frozen and each chosen checkpoint is restored |
| CMNIST 256 clean oracle recolors | Projection-only training source/rendering + oracle relation capabilities | Unique without-replacement sources; same content/clean/noisy labels; never extra supervised records or validation/test sources |
| CMNIST test-oracle diagnostic | `test_ood` retains final-test role -> explicit test-diagnostic capability/config/result union branch | Separately enabled/labeled; cannot enter ordinary selector or final fields |
| CMNIST seeds | 3 tuning -> top 3 -> +2 confirmation -> winner -> 10 fresh final | Stage labels and disjoint sets serialized; final metrics do not flow backward |
| Waterbirds-CF 4,795 training records | Supervised training view with metadata permissions | Every method sees all endpoints; ERM lacks explicit background metadata; GroupDRO may receive group IDs |
| Waterbirds 240 oracle relations (184 landbird, 56 waterbird) | Separate `OraclePairRelationView` over training records | Only oracle builder sees identities; land-minus-water orientation; estimated methods see records, not relations |
| Waterbirds official validation | Validation view with complete group spec plus training-manifest weight digest | Worst-group objective; adjusted-average/rank/epoch/stable ties as protocol specifies |
| Waterbirds official test | Opaque final-test handle | Opens only after frozen candidate, per-run checkpoint freeze, and restoration; no test-oracle selector |
| Waterbirds seeds | 3 tuning -> top 3 -> +2 confirmation -> winner -> 10 fresh final | Same structural stage separation; test remains final reporting only |

The proposal therefore traces every approved source split, validation objective, pair
permission, test restriction, and seed stage without granting a consumer the full bundle or
generic metric dictionary. It leaves the named unresolved scientific choices unimplemented.

## 13. Decision and implementation register

### Approved architectural core

The approved decisions are the core-contract list at the start of this document. In
particular, Pydantic v2, role-scoped views, validation-only ordinary selection, distinct
ordinary/test-oracle types, separate candidate/checkpoint freezes, selected-checkpoint
restoration, bounded algorithm updates, canonical local JSON, a null sink, CPU-float64
initial SVD, earlier-epoch ties, and CMNIST finalist-union handling are no longer awaiting
architectural approval.

Approval fixes behavior, not spelling. Type names, field layouts, module boundaries, and
private call signatures remain internal and revisable until both CMNIST and Waterbirds have
exercised them. Compatibility layers are not owed for pre-release internal refactors.

### Provisional guidance to test through CMNIST

- Start with the smallest Pydantic models that can express CMNIST ERM and oracle GRIT;
  avoid a universal experiment schema until a second dataset demonstrates shared fields.
- Represent role restrictions with distinct types and constructors, but let the synthetic
  lifecycle determine whether explicit phase-token classes add value.
- Keep candidate and checkpoint decision records auditable, but add only identifiers and
  contributing validation records needed by the approved selectors.
- The CMNIST slice now demonstrates bounded-update ownership with the concrete linear
  probe. Do not preinstall future-method hooks from this evidence.
- The CMNIST-specific selected-inference checkpoint uses two verified `.npy` arrays behind
  the existing restoration protocol. Do not generalize it into resume storage.
- Keep projection separate and CPU float64; basis persistence remains unresolved because
  the local slice fits once before its runs and only needs canonical diagnostics.
- Record local JSON sufficient to reproduce selection. Expand provenance and failure
  reporting from observed vertical-slice needs rather than an exhaustive framework.

### Explicitly deferred work

The deferred-extension list at the start of this document is binding. In particular,
Milestone 3 does not implement content-addressed storage, training resume, a universal
checkpoint/numerical container, future-method hooks, raw-image or accelerator
abstractions, W&B, production manifests/caches, or production pair/projection code. The
detailed pair, projection, result, provenance, and future-algorithm sections remain design
inventory only until a vertical slice demands them.

### Alternatives considered

- Frozen dataclasses plus a handwritten strict decoder were considered, but Pydantic v2 is
  approved for boundary schemas. Permissive dictionaries and dynamic `eval(...)`
  registries remain rejected.
- A trainer-owned universal `loss -> backward -> step` loop is too restrictive, while a
  completely algorithm-owned loop would surrender shared leakage and lifecycle controls.
  The approved middle ground is a bounded algorithm update inside trainer-owned iteration;
  its signature remains provisional.
- Passing a complete dataset bundle and filtering by names would recreate the inherited
  leakage surface. Role-scoped views are approved even if their internal representation
  changes.
- One configuration/result envelope with optional oracle fields permits invalid mixed
  states. Separate discriminated ordinary and diagnostic types are approved.
- A broad artifact framework would centralize storage early, but real CMNIST and
  Waterbirds artifacts should first reveal the smallest useful boundary.

### Scientific decisions deliberately unresolved

- The scientific definitions of conditional/random and nearest-pair construction.
- Whether to add the optional reporting-only paired ID rendering of CMNIST final-test
  sources.
- Search spaces and protocol details for Fish, SWAD, GroupDRO, IRM, REx, MatchDG, LISA, and
  other later algorithms.
- Final PyTorch, CUDA, and CLIP compatibility versions, including the upper supported
  Python bound.

These must become explicit protocol decisions before their configurations can resolve.

### Revised implementation order

1. **Implemented:** strict CMNIST-focused Pydantic config/result boundaries, role/metric
   types, canonical JSON, and rejection/round-trip tests.
2. **Implemented:** validation-only deterministic checkpoint/candidate selectors,
   separate freezes, minimal fake-state checkpoint restoration, and the null sink.
3. **Implemented:** the in-memory lifecycle and one-way final-test gate.
4. **Verified:** the spine was reviewed for leakage and accidental framework growth; the
   focused suite passes without production experiment components.
5. **Implemented and verified:** the CMNIST ERM/oracle-GRIT vertical slice exercised and
   revised the internal contracts, adding data, pair, projection, model, and training code
   only where that slice demanded it.
6. **Implemented and verified:** Waterbirds-CF exercised the construction, group,
   artifact-lineage, validation, final-gate, result, and smoke boundaries.
7. **Implemented and verified without a real grid:** Milestone 6A reuses the demonstrated
   lifecycle through strict production planning, run-level continuation, dataset-specific
   selection/final paths, canonical summaries, and a verified local index.
8. Review the real-server execution procedure before launching reportable work; add
   estimated pairing or W&B only in a separately approved Milestone 6B scope.

### Residual risks for real-server execution

- Numerical cost and reproducibility of full CPU-float64 SVD at actual CLIP dimensions and
  pair budgets.
- Feature/projection device and dtype conversion consistency across training and every
  evaluation role.
- The minimum real PyTorch checkpoint state needed for exact inference restoration.
- Operational duration, disk footprint, interruption behavior, and cache reuse over the
  complete 1,248-run tuning stage on the approved server.
- Whether reportable artifact paths and the pinned CPU/PyTorch environment need a narrowly
  documented server override without changing scientific identity.

Later-method extension points are not CMNIST risks to solve. They remain deferred until
their own protocols and vertical evidence exist.
