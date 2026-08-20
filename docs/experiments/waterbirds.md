# Waterbirds experiment protocol

Status: **Core dataset, split, counterfactual-access, and selection semantics approved;
numerical search details unresolved**

## Purpose

Define the rigorous Waterbirds-95 experiment used to compare ERM, GRIT/ECMP, pairing
strategies, and group-robust baselines without test-driven model selection.

The protocol distinguishes a training-side projection oracle from a model-selection
oracle. GRIT may use exact counterfactual pairs made from training birds while still
selecting hyperparameters and checkpoints using validation data alone.

## Research questions

The primary question is whether removing feature directions identified by controlled
background interventions improves worst-group bird classification relative to ERM when
both methods train their classifiers on the same released training examples.

The initial vertical slice includes:

- ERM;
- GRIT with training-side oracle pairs; and
- an `ERM + counterfactual augmentation` control that tests whether generated images are
  useful as additional supervised examples without projection.

Conditional and nearest-neighbor GRIT variants follow after the oracle vertical slice.
GroupDRO is the first group-aware baseline to port. Other inherited methods are later
scope and must use the same split and selection contracts when added.

A test-selected oracle envelope is not an ordinary result. Whether to retain one as a
separately labeled theoretical diagnostic remains unresolved.

## Canonical dataset

Use the released `waterbird_complete95_forest2water2` Waterbirds-95 artifact rather than
regenerating the base benchmark.

The released dataset was constructed from CUB-200-2011 bird images and segmentation
masks composited onto Places backgrounds. Its construction uses:

- the official CUB train/test partition;
- 20% of the CUB training partition for validation;
- 95% label/background agreement in training; and
- backgrounds balanced within each bird label in validation and test.

The authors intentionally balanced validation and test to make rare-group performance
and worst-group model selection less noisy. They also warn that rerunning the published
generator does not reproduce the released artifact exactly because of random-seed
differences. The rewrite therefore treats the released artifact, its metadata, and its
hashes as canonical.

Primary references:

- [GroupDRO Waterbirds documentation](https://github.com/kohpangwei/group_DRO#waterbirds)
- [Original Waterbirds generation script](https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py)

### Labels, backgrounds, and groups

Let:

- `y = 0` denote landbird and `y = 1` denote waterbird;
- `background = 0` denote land and `background = 1` denote water; and
- the evaluation group be the Cartesian product `(y, background)`.

The four required groups are landbird-on-land, landbird-on-water,
waterbird-on-land, and waterbird-on-water. Snow and desert are not canonical Waterbirds
backgrounds and are excluded from the primary experiment. If retained later, they must
form a separately named expanded-background benchmark rather than silently changing the
canonical group definition.

### Canonical metadata and manifest

Each released example must have an immutable example ID and record at least:

- image path and image hash;
- source CUB image ID and species when recoverable;
- binary bird label;
- binary land/water background;
- official split;
- Places background asset ID when recoverable; and
- dataset version and manifest schema version.

The completed manifest must record total counts and all four group counts for every
split. Expected counts are verified against the approved canonical artifact rather than
silently embedded as assumptions in training code.

## Split and access contract

| Resource | Definition | Permitted uses |
|---|---|---|
| Train | Released skewed Waterbirds-95 training split | Classifier optimization; training-only conditional and nearest pair estimation |
| Validation | Released background-balanced validation split | Checkpoint and hyperparameter selection; no optimization or projection fitting |
| Test | Released background-balanced test split | Final reporting after configuration and checkpoint selection |
| Oracle pair bank | Controlled land/water pairs derived only from training birds | Oracle pair differences and projection fitting; no primary classifier optimization |

Train, validation, and test retain the released split assignments. There is no new
held-out background domain in the canonical experiment: validation already contains
held-out examples from all four label/background groups and was designed for stable
worst-group tuning. An unseen snow/desert experiment would answer a different question
and requires its own protocol.

Counterfactual images and canonical training examples are separate logical resources,
even if an artifact store places them under one physical directory. Dataset loading must
not merge the pair bank into classifier training implicitly.

### Method-specific information access

- ERM receives training images and labels, but not training background/group labels.
- Oracle GRIT receives the same supervised training examples as ERM plus explicit
  training-only pair membership for projection estimation.
- Conditional and nearest variants may use training labels and background metadata only
  as required by their approved pair-builder definitions.
- GroupDRO may use `(y, background)` group labels during training because that access is
  part of the method definition.
- All ordinary methods may use validation group metadata through the prespecified
  selector. This validation access must be reported as part of the protocol.
- Test labels, group metadata, and metrics are unavailable to ordinary training,
  projection, checkpoint selection, and hyperparameter selection.

## Counterfactual pair-bank construction

### Clean oracle pairs

A clean Waterbirds oracle pair holds the bird foreground fixed while changing only the
background category:

$$
(x_i^{\text{land}}, x_i^{\text{water}}).
$$

Rules:

- The source bird must belong to the released training split.
- Validation and test birds are forbidden.
- Bird pixels, segmentation mask, crop, scale, and placement are identical across the
  two endpoints.
- One endpoint may be the released training composite; the counterfactual endpoint uses
  the opposite canonical background category.
- Auxiliary Places images come from a declared generation pool that excludes assets
  used by released validation or test examples.
- Pair extraction uses the same deterministic preprocessing for both endpoints. Random
  independent crops or augmentations are forbidden because their difference would
  contaminate the estimated nuisance direction.
- Generated endpoints do not become supervised training examples in the primary ERM or
  GRIT experiment.
- Pair direction is canonicalized as `land - water`. Reversing all pairs spans the same
  subspace, but a fixed orientation simplifies reproducibility.

The full training-source pair bank defines the primary oracle-information ceiling. A
pair-budget sensitivity study may use fixed seeded subsets, preferably balanced or
weighted across bird labels so the majority label does not dominate the estimated
subspace. Exact sensitivity budgets remain unresolved.

The pair manifest must record:

- pair ID and source CUB ID;
- source split and bird label;
- endpoint background categories and background asset IDs;
- segmentation-mask identity and hash;
- crop, scale, placement, interpolation, and compositing parameters;
- construction and sampling seeds;
- endpoint image or feature hashes; and
- generator and manifest schema versions.

### Counterfactual augmentation control

`ERM + counterfactual augmentation` is a distinct baseline in which generated training
counterfactuals are deliberately added as labeled classifier examples. Its name, result
type, training count, and sampling policy must make the extra supervised access visible.
It must not replace ordinary ERM.

### Estimated pairs

Conditional/random pairs use different training examples with the same bird label and
opposite background values. Nearest-neighbor pairs search training examples of the same
bird label in the opposite background and choose the nearest eligible representation.

Both builders must:

- use only the released training split;
- save explicit source indices and pair provenance;
- use a fixed deterministic tie-breaker;
- expose reuse and replacement policies in configuration; and
- receive the same declared pair budget as the oracle comparison unless an explicit
  pair-budget study says otherwise.

The exact nearest-search representation, normalization, reuse policy, and final budgets
remain unresolved.

## Representations and models

The first Waterbirds vertical slice will use a frozen image representation and a linear
classifier so projection behavior can be isolated. The inherited OpenAI CLIP ViT-B/32
setup is the leading compatibility candidate, but its exact weights, package identity,
preprocessing, feature normalization, and cache format must be pinned before
implementation.

Feature extraction must be deterministic and produce a manifest containing encoder
identity, weight hash, preprocessing configuration, source image manifest hash, output
shape, dtype, and feature-file hash. Canonical examples and pair endpoints must use the
same evaluation preprocessing for cached features.

A raw-image/end-to-end protocol is secondary and requires its own approved backbone,
initialization, augmentation, and optimizer configuration. Raw-image and frozen-feature
results must not be aggregated as one protocol.

## Projection

For feature rows `z_land` and `z_water`, construct the pair-difference matrix with rows

$$
d_i = z_i^{\text{land}} - z_i^{\text{water}}.
$$

GRIT removes the selected right-singular-vector subspace of this matrix. Projection rank
means the number of nuisance directions removed; rank zero is the identity operation.
Projection fitting may use only the configured training-side pair set.

The shared projection contract still must settle:

- whether and where frozen features are L2-normalized;
- whether pair differences are centered;
- decomposition precision and numerical tolerance;
- infeasible-rank behavior; and
- the rank search grid.

Every result records the requested and effective rank, pair count, singular spectrum,
explained-energy diagnostics, numerical tolerance, and any rank truncation.

## Group evaluation

Evaluation uses the four `(y, background)` groups.

- Worst-group accuracy is the minimum accuracy across the four groups and is the primary
  robustness metric.
- Per-group counts and accuracies are always reported.
- Adjusted-average accuracy weights the four group accuracies by their proportions in
  the released skewed training split and is the primary average-accuracy companion.
- Raw sample-average accuracy on the balanced validation/test splits may be reported but
  must be labeled `raw_average`, not substituted for adjusted average.
- An expected group with zero examples is an integrity failure rather than a silently
  ignored group.

The canonical adjusted-average weights are derived from and checked against the dataset
manifest. The inherited training counts—3498 landbird-on-land, 184
landbird-on-water, 56 waterbird-on-land, and 1057 waterbird-on-water—are expected but
must still be verified when the artifact is registered.

## Model selection

### Primary group-aware selector

The ordinary primary selector maximizes official validation worst-group accuracy.

Deterministic tie-breakers, in order:

1. Higher validation adjusted-average accuracy
2. Lower projection rank
3. Stable configuration ordering

This is explicitly a group-aware validation protocol. It follows the purpose of the
released balanced validation split and applies equally to all methods, even when a
method does not use group metadata during training.

### Optional group-blind sensitivity

A prespecified secondary selector may maximize validation adjusted-average accuracy
without consulting validation group identities. If run, it is labeled `group_blind` and
reported alongside rather than substituted for the primary result. Whether it is
required in the final study remains unresolved.

### Checkpoints and configurations

- Checkpoints are selected within each run using only the declared validation selector.
- The selected checkpoint is restored before final test evaluation.
- Hyperparameter configurations are compared after aggregating validation metrics across
  tuning seeds; the best individual seed is never the selection unit.
- One selected configuration is frozen before final evaluation seeds are launched.
- Fresh final seeds may select their own checkpoint epoch using validation data, but may
  not alter the frozen hyperparameter configuration.
- Test evaluation occurs only after the selection artifact identifies the configuration,
  selector, and checkpoint policy.

### Oracle diagnostics

Oracle pair access is a projection oracle, not permission to use test metrics. A
test-oracle envelope, if retained, must:

- use an explicit diagnostic configuration;
- include `test_oracle` in its result type and display label;
- never populate fields reserved for validation-selected results; and
- never be compared as though it used ordinary model selection.

## Parameter search

The search is configuration-driven and saves every resolved candidate and per-seed
validation result locally. W&B may mirror these records but does not define selection.

Required semantics:

- tune learning rate, regularization, projection rank, and other method-specific
  parameters over prespecified ranges;
- use declared, defensible budgets for every method;
- aggregate identical configurations across tuning seeds before ranking them;
- run a confirmation stage if the search strategy is adaptive; and
- evaluate the frozen selected configuration on independent final seeds.

Still unresolved:

- exact parameter ranges and search strategy;
- pair-budget sensitivity values;
- tuning, confirmation, and final seed counts;
- aggregation statistic and uncertainty method; and
- whether every method receives an equal trial count or a method-specific declared
  budget.

Historical sweep values may inform ranges but historical test-selected winners are not
valid selections.

## Configuration contract

Experiment settings belong in validated YAML rather than executable sweep modules.

Illustrative schema:

```yaml
data:
  name: waterbirds95
  artifact: waterbird_complete95_forest2water2
  splits:
    train: train
    validation: val
    test: test

pairs:
  strategy: oracle
  source_split: train
  endpoint_backgrounds: [land, water]
  classifier_access: false

selection:
  primary:
    split: validation
    metric: worst_group_accuracy
    group_fields: [label, background]
    tie_breakers: [adjusted_average_accuracy, lower_projection_rank]
```

This is a protocol example, not approval of a specific configuration library or final
serialized schema.

Typed validation must reject at least:

- pair builders referencing validation or test examples;
- an ordinary classifier implicitly receiving pair-bank endpoints;
- ordinary selectors referencing train or test metrics;
- test access before configuration selection is frozen;
- missing or unexpected evaluation groups;
- pair endpoints with different source birds, foreground geometry, or labels;
- snow/desert records presented as canonical land/water groups; and
- ordinary and test-oracle outputs sharing the same result type.

## Metrics and reporting

Required validation output:

- all four group counts and accuracies;
- worst-group accuracy;
- adjusted-average and raw-average accuracy;
- selected epoch, configuration, selector, and checkpoint; and
- aggregated tuning-seed score used to choose the configuration.

Required final output:

- test worst-group accuracy as the primary result;
- all four test group counts and accuracies;
- adjusted-average and raw-average test accuracy;
- mean, dispersion, and declared uncertainty across final seeds;
- resolved configuration and selected checkpoint epoch per seed;
- pair strategy, pair count, projection rank, and projection diagnostics; and
- any counterfactual-augmentation access or test-oracle diagnostic in an unmistakable
  result label.

## Reproducibility requirements

Use separate seeds for pair generation, pair subsampling, search/training, and data-loader
order. By default, dataset and pair artifacts remain fixed while training seeds vary.

Each completed run records:

- canonical dataset and metadata hashes;
- resolved configuration;
- feature encoder, weights, preprocessing, and cache manifest;
- pair manifest and generator version;
- construction, sampling, and training seeds;
- Git revision and dirty state;
- dependency and device information; and
- structured training, selection, checkpoint, and final-evaluation results.

The same artifact configuration and seeds must reproduce the same examples, pair
endpoints, pair subset, features, and selection ordering. Deterministic guarantees and
known nondeterministic GPU operations must be stated rather than implied.

## Required leakage and integrity tests

- Released train, validation, and test example IDs are mutually disjoint.
- Split totals and all group counts match the registered manifest.
- Oracle and estimated pair sources are training-only.
- Every oracle pair preserves source bird, label, foreground geometry, and mask while
  changing land/water background.
- Pair direction and pair count are deterministic under fixed seeds.
- Primary ERM and GRIT classifiers see identical supervised training example IDs.
- Pair-bank images cannot enter ordinary classifier training through the public API.
- Validation examples and labels do not enter classifier optimization or projection
  fitting.
- Test samples, labels, group metadata, and metrics cannot reach ordinary training,
  pair construction, checkpoint selection, or hyperparameter selection.
- Search ranks aggregated validation configurations rather than individual seeds.
- The selected checkpoint is restored before final test evaluation.
- Adjusted-average weights match the canonical training group proportions.
- Snow/desert records cannot be loaded as part of canonical Waterbirds-95.
- Large search summaries can be recomputed from saved per-run records.

## Legacy comparison

The inherited path is retained as historical evidence, not as the new protocol:

- it models counterfactual images as a dataset split and merges that split into the
  supervised training set of every ERM-derived method;
- its Waterbirds metadata admits noncanonical snow/desert backgrounds;
- its preprocessing infers pair order from a grouped loader rather than saving explicit
  endpoint records; and
- its W&B sweeps optimize test average accuracy.

The rewrite deliberately corrects those semantics. Historical artifacts and sweep
ranges may be retained for provenance and search priors, but numerical parity is not an
exit requirement.

## Remaining decisions

- Exact registered dataset archive, metadata, and source-asset hashes
- Frozen encoder identity, weights, preprocessing, and feature normalization
- Raw-image protocol, if included
- Exact counterfactual background pool and compositing implementation
- Oracle pair-budget sensitivity values and estimated-pair reuse policies
- Projection centering, tolerance, and rank-search policy
- Search ranges, budgets, aggregation, seed counts, and uncertainty method
- Whether the optional group-blind selector is required
- Whether to retain a separately labeled test-oracle envelope
- Additional baselines required beyond ERM, GRIT, counterfactual augmentation, and
  GroupDRO

## Approval checklist

- [x] Canonical released dataset and official split roles agreed
- [x] Binary land/water group definition agreed
- [x] Snow/desert excluded from the canonical protocol
- [x] Training-only oracle-pair construction agreed in principle
- [x] Counterfactual pair bank separated from ordinary classifier training
- [x] Counterfactual-augmentation control defined
- [x] Official validation worst-group accuracy approved as the primary selector
- [x] Final test isolated from ordinary selection
- [ ] Dataset and feature artifact identities registered
- [ ] Exact pair generator and pair budgets approved
- [ ] Estimated-pair details approved
- [ ] Projection and rank-search contract approved
- [ ] Search and seed budget approved
- [ ] Reporting uncertainty method approved
