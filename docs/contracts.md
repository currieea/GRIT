# Shared contract proposal

Status: **Milestone 3 proposal awaiting user review.** The names and boundaries in this
document are a concrete implementation proposal, not an implemented or approved public
API. No production contract code or executable configuration is introduced by this
checkpoint.

This proposal turns the approved experiment protocols into shared interfaces for the
rewrite. It preserves useful mathematical behavior without preserving the inherited
architecture in which `ERM` owns dataset construction, models, optimization, evaluation,
selection, and reporting (`../solver/erm.py`, especially lines 25-60 and 90-158). The
CMNIST and Waterbirds protocol documents remain authoritative for scientific choices.
The decision register at the end identifies every recommendation that still needs
approval.

## 1. Design principles and ownership

The public experiment lifecycle should be assembled from small components by a runner.
Algorithms implement update behavior; they do not inherit a dataset, trainer, evaluator,
selector, or tracker. Data access is granted through narrow capability objects instead of
passing a dictionary of all loaders or metrics. A component may receive only the
capabilities needed in its current lifecycle phase.

The initial vertical slices use frozen feature vectors. A representation provider turns
source examples into a common `InputBatch`, so a later raw-image provider can preserve the
dataset, algorithm, evaluation, selection, and result contracts. This is an extension
point, not a promise that raw-image optimization, distributed training, or mixed precision
already has a settled design.

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

### Proposed hierarchy

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

The smallest plausible choices are:

1. Frozen standard-library dataclasses plus a handwritten strict decoder and validator.
   This avoids a dependency but requires custom discriminated-union, unknown-key, error,
   and canonical-serialization code. Kernel-GRIT demonstrates that frozen dataclasses are
   useful, but its parser manually reads recognized keys without a general unknown-key
   rejection pass (`../../Kernel-GRIT/src/grit/config.py`, lines 40-133 and 497-508).
2. Pydantic v2 models at the configuration and result boundaries, converted to small
   immutable internal records where useful. It provides nested validation,
   discriminated unions, strict unknown-field rejection, schema generation, and stable
   dump/load APIs at the cost of one runtime dependency.

The recommendation awaiting approval is Pydantic v2 for boundary schemas. The dependency
is not added by this proposal. If dependency minimization wins, the dataclass alternative
must first implement equivalent strictness and contract tests; permissive dictionary
access is not acceptable.

Validation rules are part of the public contract:

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

## 3. Dataset and split contracts

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

The CMNIST deterministic source-partition algorithm and the Waterbirds acquisition or
reconstruction implementation are deliberately unresolved. Their configurations require
registered `construction_method_id` values and manifests. Until a choice is approved and
implemented, resolution fails with an actionable unsupported-method error; it cannot fall
back to an arbitrary split or download.

## 4. Leakage-resistant lifecycle and access boundaries

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
  candidate. Validation within each fresh final-seed run produces `FrozenCheckpointSelection`,
  naming that run's selected epoch and checkpoint. A checkpoint restorer verifies the
  checkpoint and configuration/manifests and produces `RestoredCheckpoint`.
- The final evaluator requires a matching `FrozenCandidateSelection`,
  `FrozenCheckpointSelection`, and `RestoredCheckpoint`. Only then can the runner exchange its
  opaque `FinalTestHandle` for a final-test evaluation view. Final metrics are returned
  directly to result assembly, never back into the selector or search coordinator.
- Test-oracle evaluation is a separate diagnostic run kind with separate configuration,
  test-bearing diagnostic capability, selector, result type, and output namespace. It is
  permitted only where a protocol explicitly allows it and requires a conspicuous opt-in.
  Its selector may use the configured test diagnostic specifically because the result is
  an oracle envelope; it cannot populate ordinary selected-candidate or final-test fields.

The following capability trace is normative for the proposal:

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
implicit device transfer or unrecorded precision narrowing is forbidden. The proposal
recommends deterministic CPU float64 fitting followed by an explicit runtime conversion,
but that numerical policy awaits approval.

`ProjectionDiagnostics` contains requested, numerical, and effective ranks; the complete
singular-value spectrum; absolute and relative thresholds; discarded/retained energy;
pair/feature dimensions; nonfinite-input checks; orthonormality, symmetry, and idempotence
residuals; backend/dtype/device; and pair/feature manifest identities. A zero numerical
rank is valid and yields identity even when requested rank is positive, while retaining the
requested/effective distinction in results.

Serialization separates lightweight metadata from tensor-heavy artifacts. Canonical JSON
contains config, dimensions, ranks, diagnostics, hashes, and `ArtifactRef` values. The
removed right-singular-vector basis with shape `[D, effective_rank]` is the canonical
numerical payload; the square projector is derived and is not independently authoritative.
The complete singular-value spectrum is required, either inline when demonstrably small or
through its own artifact reference—it is never omitted as “optional.” Numerical payloads
use a non-pickle artifact such as versioned NPZ. Loading verifies hash, shape, dtype,
feature manifest, and pair-set identity before producing a fitted transform. The exact
artifact container is an awaiting-approval engineering decision.

## 7. Algorithm, model, and training contracts

### Minimum compositional interface

The runner constructs a model through `ModelFactory`, constructs optimizer service(s) from
explicit configuration, and injects them into an algorithm factory. Once constructed, the
algorithm owns training-time model mutation, optimizer stepping, and its algorithm-specific
state. The trainer owns iteration and lifecycle control. This division supports ERM and
GRIT immediately without assuming every method has one scalar loss and exactly one
optimizer step.

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

`on_validation` is optional and receives only `ValidationFeedback`, never evaluator or
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

`AlgorithmCheckpointState` is a typed envelope with separately named model state,
optimizer-state mapping, and opaque method-state payload; it is not one flattened state
dictionary. The algorithm supplies/restores it because the algorithm owns optimizer
stepping and may replace parameters, while the checkpoint coordinator supplies artifact
storage, identity validation, and trainer/RNG state around it.

The trainer owns epoch/update loops, batch-provider coordination, validation cadence,
counter advancement, cancellation/failure handling, checkpoint requests, and training
history. It does not calculate algorithm-specific penalties or assume one optimizer step.
The runner owns component construction, capability issuance, multi-run search stages,
selection, checkpoint restoration, final evaluation, provenance, and result assembly.

### Evidence and provisional future needs

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
protocols are approved.

## 8. Selection and checkpoint contracts

### Records and deterministic decisions

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
three for one method and selector. Confirmation adds only the declared fresh seeds; it does
not revisit test results. The combined five-seed decision produces a durable
`FrozenCandidateSelection`. Final runs use fresh final seeds and that frozen scientific
configuration. Each final-seed run then creates a `FrozenCheckpointSelection` using
validation only. The reported final metrics come from those restored selected checkpoints,
never implicitly from the last live weights.

Validation records from tuning and confirmation stages may contribute to candidate
selection according to the configured stage. A final-stage validation record may select
the epoch/checkpoint within that already-frozen candidate and final seed, but is rejected
by every candidate or hyperparameter selector. Stage-specific table constructors enforce
that distinction.

A checkpoint envelope contains:

- checkpoint/schema identity, run/candidate/config identities, epoch/update counters, and
  creation reason;
- model parameters and buffers, optimizer/scheduler state, algorithm-specific tensors and
  counters, trainer state, and the configured RNG-state coverage;
- dataset, feature, pair, and projection manifest identities; code/environment provenance;
- content hashes and `ArtifactRef` values for each tensor-heavy payload.

Restoration validates all identities and produces a receipt naming loaded components and
any intentionally unsupported resume fields. A selection-eligible checkpoint must support
faithful inference restoration; a checkpoint advertised as resumable must additionally
round-trip every declared mutable state and RNG source.

### Protocol mappings

For CMNIST, the primary selector maximizes the minimum accuracy across `val_e01`,
`val_e02`, and `val_e05`; ties prefer higher mean accuracy, then lower projection rank,
then stable configuration order. The secondary source selector uses the minimum of
`val_e01` and `val_e02`, with the same higher-mean, lower-rank, stable-order ties. An
earlier-epoch tie rule for otherwise identical checkpoint records is recommended to make
epoch selection total, but awaits approval because the protocol currently specifies the
candidate tie sequence, not this checkpoint detail.

Both CMNIST selector branches consume the same three-seed validation records, write their
own ordered top-three artifact, and freeze their own five-seed winner. Confirmation runs
the union of those finalist candidate IDs, so a candidate appearing in both branches is not
duplicated; its fresh records may contribute to both prespecified selectors.

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

## 9. Results, provenance, and tracking

Canonical local output is authoritative. Results form another strict discriminated union:

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
  diagnostic_history: diagnostic-only history artifact plus summary
  oracle_decision: CmnistTestOracleDecision | None
  oracle_metrics: tuple[DiagnosticMetricRecord, ...]

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

Local output uses an atomic run directory: a temporary/incomplete marker while running,
content-addressed artifacts, an artifact manifest, then one canonical JSON result committed
last. Histories may use versioned JSON Lines or a typed columnar format; checkpoints,
feature matrices, projection arrays, and predictions are referenced through `ArtifactRef`
rather than embedded in JSON or YAML. Pickled arbitrary Python objects are not a portable
contract.

Every boundary object must round-trip `typed object -> canonical serialization -> typed
object` without losing enum values, ordering with semantic meaning, identities, numeric
types promised by the schema, or optional-versus-absent distinctions. Round-trip tests also
recompute content hashes. Readers reject unsupported major schema versions; minor additive
migration, if later adopted, must be explicit and tested.

Tracking is an optional `EventSink` receiving immutable lifecycle, metric, artifact, and
completed-result events. `NullEventSink` is the default; `WandbMirrorSink` mirrors the same
information. A sink failure follows configured fail-open/fail-closed operational policy
and is recorded locally. W&B may not inject a sweep candidate, rename split roles, return
metrics to selectors, choose a checkpoint, or be the only location of an artifact required
to interpret a run.

## 10. Non-executable protocol examples

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

The next implementation goal should build tiny in-memory fakes and run these tests without
datasets, external encoders, GPUs, or W&B access:

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

Additional protocol fixtures should encode the expected CMNIST source/view counts and
Waterbirds-CF group counts, while keeping actual source-partition and acquisition logic out
of these contract tests.

## 12. Protocol trace and self-review

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

## 13. Decision register

### Recommended architectural decisions awaiting approval

1. Use Pydantic v2 for strict boundary schemas and immutable small internal records where
   useful; do not add it until approved.
2. Make role-scoped capability views and monotonic lifecycle tokens the leakage boundary,
   with ordinary selectors accepting a validation-only table.
3. Use discriminated ordinary versus CMNIST test-oracle configuration/result roots, and
   distinct frozen-candidate versus per-final-run frozen-checkpoint artifacts.
4. Let algorithms own bounded update behavior, injected model mutation, optimizer stepping,
   and opaque algorithm state while the trainer owns iteration and validation cadence.
5. Use one versioned checkpoint envelope for separately identified model, optimizer,
   algorithm, trainer, and configured RNG state, with explicit restoration receipts.
6. Make canonical local JSON plus content-addressed artifact references authoritative;
   treat W&B only as an optional event mirror.
7. Store numerical array artifacts in a safe, versioned non-pickle format; NPZ is the
   initial recommendation, subject to scale testing.
8. Fit the linear projection deterministically on CPU in float64, then perform an explicit,
   recorded conversion for runtime transformation.
9. Add earlier epoch as the final semantic checkpoint tie-break for CMNIST before stable
   checkpoint identity; the currently approved CMNIST candidate tie sequence remains
   unchanged.
10. Confirm the union of the primary and secondary CMNIST top-three candidate sets once,
    while retaining separate finalist and frozen-winner artifacts for each selector.

### Alternatives considered

- Frozen dataclasses plus a handwritten strict decoder remain viable if they demonstrate
  equivalent nested validation, discriminated unions, schema migration, and canonical
  round trips. Permissive dictionaries and dynamic `eval(...)` registries are rejected.
- A trainer-owned universal `loss -> backward -> step` loop is simpler but cannot represent
  verified Fish/SWAD behavior. A completely algorithm-owned loop is flexible but would
  surrender shared leakage, checkpoint, and provenance controls. The bounded-update
  interface is the proposed middle ground.
- Passing a complete dataset bundle and filtering by names is simpler but recreates the
  inherited leakage surface. Role-capability views are recommended despite added types.
- One configuration/result envelope with optional oracle fields is smaller, but permits
  invalid mixed states and weakens the audit boundary. A discriminated union is preferred.
- Embedding small arrays in JSON or serializing arbitrary Python objects is convenient but
  weakens portability, safety, and identity validation. Referenced typed artifacts are
  preferred.
- GPU SVD may be faster, but deterministic CPU fitting is the proposed starting point for
  the small-rank frozen-feature experiments. The vertical slice should measure its cost.

### Scientific decisions deliberately unresolved

- CMNIST's deterministic official-training source-partition algorithm.
- The scientific definitions of conditional/random and nearest-pair construction.
- Whether to add the optional reporting-only paired ID rendering of CMNIST final-test
  sources.
- Waterbirds acquisition, artifact-validation, reconstruction, and generated-artifact
  implementation details beyond the approved logical protocol.
- Search spaces and protocol details for Fish, SWAD, GroupDRO, IRM, REx, MatchDG, LISA, and
  other later algorithms.
- Final PyTorch, CUDA, and CLIP compatibility versions, including the upper supported
  Python bound.

These must become explicit protocol decisions before their configurations can resolve.

### Implementation order after approval

1. Implement strict config/result boundary models, canonical serialization, validation
   errors, and the corresponding synthetic round-trip/rejection tests.
2. Implement artifact/manifests and in-memory dataset/split capability types with leakage
   tests.
3. Implement validation metric, selection, checkpoint identity/restoration, and lifecycle
   tokens with synthetic models only.
4. Implement the minimal ERM/GRIT algorithm/trainer protocols using in-memory batches and
   checkpointable fake state.
5. Implement pair/projection interfaces and artifact validation separately from their
   scientific builders or production mathematics.
6. Implement local result writing and null/optional tracking sinks, then review the entire
   Milestone 3 implementation. Only a later approved goal may begin Milestone 4; the
   CMNIST vertical slice remains later still.

This order remains Milestone 3 implementation work. It does not authorize Milestone 4 or
production dataset/algorithm code merely because this proposal is approved.

### Risks to test in the CMNIST vertical slice

- Whether the representation boundary is genuinely sufficient for both cached frozen
  features and future raw-image batches without premature distributed/AMP abstractions.
- Numerical cost and reproducibility of full CPU float64 SVD and the chosen array format at
  actual CLIP dimensions and pair budgets.
- Feature/projection device and dtype conversion consistency across training and every
  evaluation role.
- Checkpoint size, write cadence, and exact inference restoration under real PyTorch state.
- The usefulness of step-level validation hooks without exposing evaluators or encouraging
  validation overfitting.
- Whether candidate/config identities remain stable under real path overrides, cache
  reuse, and multi-process execution.
- Which future-method extension points are truly shared; do not generalize provisional
  Fish, SWAD, MatchDG, or LISA behavior until their protocols and vertical tests require it.
