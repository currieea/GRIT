# ColoredMNIST experiment protocol

Status: **Draft — construction, split roles, and selection policies agreed in principle**

## Purpose

Define a rigorous ColoredMNIST experiment for comparing ERM with GRIT/ECMP and its
pairing variants. The experiment is designed for meaningful parameter search and
validation-based model selection rather than exact reproduction of a historical table.

This document is the scientific protocol contract. Configuration files instantiate this
contract, and typed configuration validation must reject configurations that violate its
split or information-access rules.

## Primary research question

Given training environments in which color is more predictive of the noisy target than
digit shape, how well do ERM and GRIT recover a predictor that generalizes when the
color-target correlation reverses?

The initial vertical slice compares:

- ERM
- GRIT/ECMP with clean oracle invariant pairs

Conditional/random and nearest-neighbor pair construction are subsequent variants. The
set and implementation order of additional domain-generalization baselines remain
unresolved.

## Construction semantics

For an MNIST digit $d$, define:

$$
\begin{aligned}
\tilde y &= \mathbb{1}[d \ge 5], \\
y &= \tilde y \oplus B_y,\quad B_y \sim \operatorname{Bernoulli}(p_y), \\
c^e &= y \oplus B_c^e,\quad B_c^e \sim \operatorname{Bernoulli}(p_c^e).
\end{aligned}
$$

- $\tilde y$ is the clean binary label.
- $y$ is the noisy downstream target.
- $c^e$ is the color in environment $e$.
- $p_y$ is named `label_flip_prob` in configuration.
- $p_c^e$ is named `color_flip_prob`.
- Color-target agreement in environment $e$ is approximately $1-p_c^e$.
- The canonical encoding is $c=0$ for red and $c=1$ for green.

The proposed primary label flip probability is $p_y=0.25$, matching the conventional
IRM-style task. A noise-free $p_y=0$ experiment may be retained as a sensitivity
analysis, but is not the primary result.

Label noise is sampled once per underlying source image and shared across every rendering
of that source. Each environment samples color from its Bernoulli mechanism using a
deterministic stream keyed by the construction seed and stable environment name. Merely
reordering environment entries must not change the generated dataset.

## Source partitions and environments

Underlying MNIST source images are partitioned before they are rendered into colored
environments. A source partition identifies handwritten images; an environment identifies
how a source partition is rendered.

### Source partitions

| Source partition | MNIST source | Count | Permitted uses |
|---|---:|---:|---|
| `train_e01_sources` | Official train | 25,000 | Classifier optimization; train-only pair sourcing |
| `train_e02_sources` | Official train | 25,000 | Classifier optimization; train-only pair sourcing |
| `validation_sources` | Official train | 10,000 | Validation evaluation and selection only |
| `test_sources` | Official test | 10,000 | Final OOD evaluation only |

The two training partitions and validation partition are mutually disjoint. The official
MNIST test split is disjoint by construction and remains inaccessible until ordinary
selection is frozen.

The exact deterministic partitioning algorithm is not yet approved. The preferred design
is a seeded partition stratified by original digit so each partition has stable digit
coverage. The chosen algorithm and resulting source indices must be stored in the dataset
manifest.

### Rendered environments

| Environment | Role | Sources | Color flip | Approx. agreement | Permitted uses |
|---|---|---|---:|---:|---|
| `train_e01` | Train | `train_e01_sources` | 0.1 | 0.9 | Optimization |
| `train_e02` | Train | `train_e02_sources` | 0.2 | 0.8 | Optimization |
| `val_e01` | Validation | `validation_sources` | 0.1 | 0.9 | Selection and reporting |
| `val_e02` | Validation | `validation_sources` | 0.2 | 0.8 | Selection and reporting |
| `val_e05` | Validation | `validation_sources` | 0.5 | 0.5 | Selection and reporting |
| `test_ood` | Test | `test_sources` | 0.9 | 0.1 | Final reporting only |

The three validation environments are three renderings of the same 10,000 held-out
sources. They are not 30,000 independent examples. Their source image, source ID, digit,
clean label, and noisy target are shared; only color varies.

Reusing validation sources is intentional. It isolates sensitivity to the color
intervention and improves comparisons among validation environments. Metrics and
uncertainty calculations must retain this repeated-measures structure.

No separate final ID test is required by the primary protocol. Source-like validation
metrics provide ID diagnostics, while all 10,000 official MNIST test images are reserved
for the primary $p_c=0.9$ OOD result. A paired ID rendering of the test sources may be
added later as a reporting-only diagnostic, but it must never enter selection.

## Canonical records and rendering

The canonical dataset record should retain, at minimum:

- grayscale source image or immutable source reference
- official MNIST split and source index
- digit label
- clean binary label
- noisy downstream target
- source-partition name
- environment name and role
- color and configured color flip probability
- construction seed and schema version

Color rendering should be a separate deterministic component. The same construction must
support:

- a two-channel legacy-compatible rendering, if raw-pixel parity is needed;
- a three-channel RGB rendering for CLIP or other image encoders.

The primary raw-pixel architecture and frozen encoder remain unresolved. Rendering and
representation choices must not change source partitions, labels, or colors.

## Invariant pairs

### Clean oracle pairs

A clean oracle pair uses one training source image and renders it in both colors while
holding digit content and target fixed:

$$
(x_i^{\text{red}}, x_i^{\text{green}}).
$$

Rules:

- Pair sources must come only from the two training source partitions.
- Validation and test sources are forbidden.
- Sources are sampled without replacement for a fixed pair seed.
- Both endpoints share the same source ID, digit, clean label, and noisy target.
- Pair endpoints are available to projection estimation only; they do not become
  additional supervised classifier-training examples.
- Pair provenance records the source ID, endpoint colors, construction parameters, and
  artifact hashes.

The proposed initial budget is 256 unique pairs, matching the Kernel-GRIT development
setup. The final pair-count search or sensitivity range remains unresolved.

The primary proposal allows a pair source to also appear in ordinary classifier training.
This is valid because the pair is auxiliary training-side invariance information. A
disjoint pair-bank sensitivity experiment may be added later, but it is not required for
the initial vertical slice.

### Estimated pairs

Conditional/random and nearest-neighbor builders must use only training sources. Their
precise matching constraints, search representation, normalization, pair budgets, and
tie-breaking policies remain unresolved and must be approved before implementation.

For every strategy, save pair provenance sufficient to reproduce the selected examples.

## Model selection

Every search candidate is evaluated on all three validation environments. Two selectors
are defined in advance and applied to the same saved validation results.

### Primary robustness selector

$$
s_{\text{robust}} =
\min\{\operatorname{acc}(\text{val\_e01}),
       \operatorname{acc}(\text{val\_e02}),
       \operatorname{acc}(\text{val\_e05})\}.
$$

The neutral $p_c=0.5$ environment makes color uninformative without revealing the
direction of the final $p_c=0.9$ reversal. This is explicitly a robustness-aware
validation protocol, not strict source-only domain-generalization selection.

Proposed tie-breakers, in order:

1. Higher mean accuracy across the selector's validation environments
2. Lower projection rank
3. Stable configuration ordering

### Secondary source-only selector

$$
s_{\text{source}} =
\min\{\operatorname{acc}(\text{val\_e01}),
       \operatorname{acc}(\text{val\_e02})\}.
$$

This selector quantifies the practical model-selection gap when validation only mimics
the training domains. It is prespecified as a secondary protocol; it must not be chosen
or discarded after seeing test results.

### Checkpoints and configurations

- The selector used for checkpoint choice must be recorded.
- The selected checkpoint must be restored before final evaluation.
- Hyperparameter configurations are compared after aggregating the relevant validation
  score across tuning seeds; the best individual seed is never the selection unit.
- Primary and secondary selectors may choose different configurations from one search.
- Fresh final-evaluation seeds use the frozen selected configuration.
- The official test split is evaluated only after the selection artifact is finalized.

The exact tuning-seed count, final-seed count, aggregation statistic, and confirmation
procedure remain unresolved.

## Oracle diagnostics

Oracle pair access and oracle model selection are distinct concepts.

- **Projection oracle:** GRIT uses true same-source recolorings from training data to
  estimate nuisance directions. This is compatible with ordinary validation selection.
- **Test-oracle envelope:** rank, configuration, or checkpoint is chosen using
  `test_ood`. This is not an ordinary result and may be retained only as a theoretical
  or diagnostic upper envelope.

Any test-oracle run must:

- use a separate diagnostic configuration;
- opt into test selection explicitly;
- include `test_oracle` in its result type and display label;
- never populate fields reserved for validation-selected results;
- never be compared as though it used the ordinary selector.

## Configuration contract

Experiment settings belong in validated YAML rather than executable sweep modules. The
configuration model must distinguish source partitions from rendered environments so
multiple environments can reference the same validation sources.

Illustrative schema:

```yaml
seeds:
  construction: 0
  pairs: 0
  training: 0

data:
  name: colored_mnist
  data_root: data
  label_flip_prob: 0.25

  source_partitions:
    train_e01_sources: {mnist_split: train, count: 25000}
    train_e02_sources: {mnist_split: train, count: 25000}
    validation_sources: {mnist_split: train, count: 10000}
    test_sources: {mnist_split: test, count: 10000}

  environments:
    - {name: train_e01, role: train, sources: train_e01_sources,
       color_flip_prob: 0.1}
    - {name: train_e02, role: train, sources: train_e02_sources,
       color_flip_prob: 0.2}
    - {name: val_e01, role: validation, sources: validation_sources,
       color_flip_prob: 0.1}
    - {name: val_e02, role: validation, sources: validation_sources,
       color_flip_prob: 0.2}
    - {name: val_e05, role: validation, sources: validation_sources,
       color_flip_prob: 0.5}
    - {name: test_ood, role: test, sources: test_sources,
       color_flip_prob: 0.9}

  invariant_pairs:
    strategy: oracle
    sources: [train_e01_sources, train_e02_sources]
    num_pairs: 256
    classifier_access: false

selection:
  primary: robust
  policies:
    robust:
      environments: [val_e01, val_e02, val_e05]
      metric: worst_accuracy
      tie_breakers: [mean_accuracy, lower_projection_rank]
    source_only:
      environments: [val_e01, val_e02]
      metric: worst_accuracy
      tie_breakers: [mean_accuracy, lower_projection_rank]
```

This is a protocol example, not approval of a particular configuration library or final
serialized schema.

Typed validation must reject at least:

- overlap among train, validation, and test source partitions;
- ordinary selectors referencing train or test environments;
- pair builders referencing validation or test sources;
- test access before ordinary selection is frozen;
- duplicate source IDs within an environment;
- inconsistent labels for repeated renderings of one source;
- unknown environment names or roles;
- ordinary and test-oracle outputs sharing the same result type.

## Projection

The projection contract remains to be completed. It must define:

- rows-as-pairs difference orientation
- descriptive projection-rank parameter
- rank-zero identity semantics
- numerical decomposition and tolerance policy
- infeasible-rank handling
- centering and feature-normalization policy
- saved spectrum and numerical diagnostics

Projection estimation may use only the configured training-side pair set.

## Parameter search

The search must be configuration-driven and apply the same candidate space to both
selectors. Still unresolved:

- parameters and ranges for each method
- search strategy
- budget per method
- tuning and final seed counts
- aggregation statistic and uncertainty method
- confirmation procedure

The search runner must save every resolved candidate and its per-seed validation metrics.
W&B may mirror the search, but local structured results define selection semantics.

## Metrics and reporting

Required validation reporting:

- accuracy for each of `val_e01`, `val_e02`, and `val_e05`
- robust worst-environment score
- robust mean-environment score
- source-only worst-environment score
- source-only mean-environment score
- selected epoch, configuration, and selector

Required final reporting:

- $p_c=0.9$ OOD accuracy on all 10,000 official MNIST test sources
- mean, dispersion, and declared uncertainty across final seeds
- primary robustness-selected result
- secondary source-selected result
- projection rank, pair count, and pair/projection diagnostics
- test-oracle envelope only when separately enabled and labeled

The repeated validation renderings represent 10,000 sources, not 30,000 independent
examples. Any bootstrap or source-sampling confidence calculation must resample source IDs
jointly across the three renderings.

## Reproducibility and provenance

Use separate construction, pair, and training seeds. By default, construction and pair
seeds remain fixed while training seeds vary. Dataset-seed sensitivity is a separately
declared experiment.

The dataset manifest must record:

- official MNIST artifact identity and hashes
- construction schema version
- resolved configuration
- source IDs for every partition
- digit counts by partition
- clean and noisy label counts
- configured and realized flip rates by environment
- environment source-overlap matrix
- pair source IDs and endpoint metadata
- rendering version

The same resolved construction and pair seeds must reproduce all source indices, labels,
colors, and pair endpoints exactly.

## Required integrity tests

- Train, validation, and test source partitions are mutually disjoint.
- Training environments are disjoint and contain exactly 25,000 sources each.
- All validation environments contain the same 10,000 source IDs in the same canonical
  order.
- Repeated validation renderings share digit, clean label, and noisy target.
- Configured color-target correlations are achieved within an approved tolerance.
- Oracle-pair endpoints recover identical grayscale content and opposite colors.
- Pair sources are training-only and pair count is honored.
- Validation examples and labels do not enter optimization or projection estimation.
- Ordinary selectors cannot receive test metrics through their public interface.
- Search selects aggregated validation configurations rather than individual seeds.
- Final test evaluation occurs only after checkpoint and configuration selection.
- Dataset generation is deterministic under fixed construction and pair seeds.

## Historical context and compatibility

The original IRM experiment created two training environments and one $p_c=0.9$ test
environment from the 60,000-image MNIST training split. The inherited GRIT repository
adds source-like and OOD evaluation roles but contains ambiguous and unsafe selection
behavior. Kernel-GRIT provides a cleaner deterministic generator and exact training-side
oracle pairs, but its current development configuration has no validation role and splits
the official test set into 5,000 ID and 5,000 OOD sources.

The rewrite deliberately uses the official 10,000-image MNIST test split only for final
OOD evaluation. Historical parity is diagnostic, not an exit requirement.

Primary references:

- [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- [Original IRM ColoredMNIST code](https://github.com/facebookresearch/InvariantRiskMinimization/tree/main/code/colored_mnist)

## Remaining decisions

- Exact deterministic and stratified source-partition algorithm
- Whether $p_y=0$ is a required sensitivity experiment
- Raw-pixel model and rendering protocol
- Frozen encoder, weights, preprocessing, and feature normalization
- Final oracle-pair budgets and pair-count sensitivity range
- Conditional/random and nearest-neighbor pair definitions
- Projection numerics and rank-search space
- Search budgets, seed counts, aggregation, and confirmation policy
- Additional baseline methods required for the first complete study
- Whether a reporting-only paired ID rendering of final test sources is useful

## Approval checklist

- [x] Source partition sizes and official test role agreed
- [x] Training and validation environment correlations agreed
- [x] Shared-source validation renderings agreed
- [x] Primary robustness and secondary source-only selectors agreed in principle
- [x] Final OOD test isolated from ordinary selection
- [ ] Deterministic partition algorithm approved
- [ ] Representation and model protocols approved
- [ ] Pair budgets and estimated-pair definitions approved
- [ ] Projection contract and rank search approved
- [ ] Search and seed budget approved
- [ ] Reporting uncertainty method approved
