# ColoredMNIST experiment protocol

Status: **Implemented for ERM and oracle GRIT. GroupDRO is specified but not yet
implemented. Conditional and nearest-neighbor pair definitions are still open.**

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

GroupDRO is the first baseline extension. Conditional/random and nearest-neighbor pair
construction are subsequent GRIT variants. REx, IRM, and any additional
domain-generalization baselines remain deferred.

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
deterministic draw keyed by the construction seed, stable source ID, and stable environment
name. Merely reordering source rows or environment entries must not change the generated
dataset.

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

The approved construction method is `cmnist-stratified-hash-v1`:

1. Stable source identity is the official MNIST split plus its official source index.
2. Group all 60,000 official-training sources by original digit. Allocate exactly 10,000
   validation sources proportionally by digit using largest-remainder apportionment.
   Equal remainders are awarded in ascending digit order.
3. Remove those validation sources. Apportion exactly 25,000 of the remaining 50,000 to
   `train_e01_sources` with the same rule; `train_e02_sources` receives the rest.
4. Before membership assignment within each digit, order sources by SHA-256 of the
   null-separated UTF-8 fields `cmnist-stratified-hash-v1`, construction seed, official
   split, and official source index. The source index is the deterministic collision
   fallback.
5. The official 10,000-source test split becomes `test_sources` unchanged and never enters
   the official-training partition operation.

The canonical partition manifest records the method ID, construction seed, ordered
membership, per-digit counts, and SHA-256 membership digest for every partition. The
algorithm is invariant to input ordering. Changing the construction seed changes
membership reproducibly without changing any target count.

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

Rendering and representation choices must not change source partitions, labels, or
colors.

## Frozen representation and classifier

The primary representation is frozen OpenAI CLIP ViT-B/32 using the official OpenAI
weights and deterministic evaluation preprocessing. The cache stores the
512-dimensional unnormalized `encode_image` output.

Unnormalized features are primary because they match the paper and inherited linear
probe and preserve the additive Euclidean geometry assumed by linear GRIT. Per-example
L2 normalization is a prespecified, separately reported sensitivity. In that
sensitivity, every training, validation, test, and pair-endpoint feature is normalized
before pair differences or classifier fitting.

The initial classifier is a linear two-class head trained with Adam:

- batch size: 256;
- maximum epochs: 40;
- learning-rate candidates: `[1e-4, 3e-4, 1e-3, 3e-3]`; and
- weight-decay candidates: `[0, 1e-5, 1e-4, 1e-3]`.

Raw-pixel training is deferred. A future raw-pixel protocol must be reported separately
and must not be aggregated with frozen-feature results.

The feature manifest records encoder package and version, checkpoint identity and hash,
preprocessing, normalization mode, source manifest hash, output shape, dtype,
device/precision details, batch size, PyTorch/CUDA runtime, GPU identity/capability, and
feature-file hash. The initial supported CUDA preparation profile uses deterministic
single-GPU float32 computation with TF32 and mixed precision disabled. Training remains
CPU-only; CPU and CUDA feature caches are distinct attributable artifacts.

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
- Endpoint orientation is fixed as red minus green.

The primary budget is 256 unique pairs, matching the paper and Kernel-GRIT development
setup. Prespecified pair-budget sensitivities use 32, 64, 128, 256, and 512 unique
training sources; 256 remains the primary result rather than a validation-selected pair
count.

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

Every candidate runs on three tuning seeds. Candidate ranking uses the mean selector
score across those seeds. The top three configurations receive two additional
confirmation seeds, and the winner is chosen using its combined five-seed validation
mean. That configuration is frozen and evaluated using ten fresh final seeds shared
across methods. Tuning and confirmation seeds do not enter the final reported estimate.

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

features:
  encoder: openai_clip
  model: ViT-B/32
  normalize: false

projection:
  center_differences: false
  ranks: {start: 2, stop: 24, step: 1}

training:
  optimizer: adam
  learning_rates: [0.0001, 0.0003, 0.001, 0.003]
  weight_decays: [0.0, 0.00001, 0.0001, 0.001]
  batch_size: 256
  max_epochs: 40

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

For feature rows `z_red` and `z_green`, construct the uncentered pair-difference matrix

$$
D_i = z_i^{\text{red}} - z_i^{\text{green}}.
$$

The primary protocol does not subtract the mean difference. GRIT removes the selected
right-singular-vector subspace of `D`. Projection rank is the number of removed
directions; rank zero is the exact identity operation. The primary rank candidates are
every integer from 2 through 24, matching the paper's stated range. Rank zero remains a
supported identity control, but it is excluded from the primary GRIT search because ERM
already provides the explicit unprojected baseline. Any rank-zero result is reported as
a separately named sanity control rather than as a GRIT candidate eligible to win the
primary search. Rank one is outside the primary paper-aligned grid.

Use deterministic full `torch.linalg.svd` rather than randomized
`torch.svd_lowrank`. The implementation must:

- reject ranks above `min(number_of_pairs, feature_dimension)`;
- distinguish requested, numerical, and effective rank;
- use an explicit dtype and relative singular-value tolerance;
- produce an orthogonal projector within a tested tolerance; and
- save singular values, explained-energy diagnostics, tolerance, and effective rank.

Projection estimation may use only the configured training-side pair set. The normalized
feature sensitivity fits a separate projection after normalizing every endpoint.

## GroupDRO baseline

GroupDRO trains the same unprojected linear probe over frozen CLIP features as ERM. It
does not receive pair identities. Its four training groups are the Cartesian product of
the noisy downstream target and observed color, in fixed order `(y=0,c=0)`, `(0,1)`,
`(1,0)`, `(1,1)`. This privileged group annotation is part of the method definition; it
comes only from the two training environments.

Training follows the reference GroupDRO stochastic objective. For per-group minibatch
losses $L_g$ and adversarial probabilities $q_g$, initialize $q_g=1/4$ and update

$$
q_g \leftarrow \frac{q_g\exp(\eta L_g)}{\sum_j q_j\exp(\eta L_j)},
\qquad
L_{\mathrm{DRO}}=\sum_g q_gL_g.
$$

The training sampler assigns every example inverse-frequency weight for its group and
samples exactly the training-set size with replacement per epoch. This makes groups
uniform in expectation without requiring every minibatch to contain every group. The
sampler is deterministic from the run seed. Generalization adjustment is fixed to zero,
and loss normalization is disabled, matching the reference Waterbirds invocation.

The approved adversarial step-size candidates are `0.001`, `0.01`, and `0.1`: the
reference default `0.01` with one decade on either side. They are crossed with the same
learning-rate and weight-decay grid as ERM and selected using the same validation-only
selectors. See the [reference GroupDRO implementation](https://github.com/kohpangwei/group_DRO).

## Parameter search

The search is configuration-driven and applies the same saved candidate results to both
selectors.

- ERM searches the Cartesian product of the approved learning-rate and weight-decay
  candidates.
- GRIT searches that optimizer grid jointly with ranks 2 through 24.
- GroupDRO searches that optimizer grid jointly with adversarial step sizes `0.001`,
  `0.01`, and `0.1` once its implementation is enrolled.
- Every candidate runs on three tuning seeds.
- The top three configurations receive two confirmation seeds.
- The five-seed validation mean selects the frozen configuration.
- The winner runs on ten fresh final seeds shared across methods.
- Later methods receive prespecified method-specific ranges rather than generic
  `param1`, `param2`, or `param3` values.

The search runner saves every resolved candidate and per-seed validation metric. W&B may
mirror the search, but local structured results define selection semantics.

Milestone 6A implements this approved ERM/oracle-GRIT grid locally. The production schema
requires explicit dataset, feature-cache, and 256-pair manifest paths; the canonical
production inventory; pinned official OpenAI CLIP identity; one matching normalization;
and explicit construction, pair, 3 tuning, 2 confirmation, and 10 final seeds. Planning
emits all 384 ordered candidates (16 ERM and 368 GRIT) and expected stage counts without
loading arrays, training, checkpoints, or final-test access. The primary unnormalized
experiment and the named L2 sensitivity are distinct configurations and caches.

The run scheduler applies both selectors to the same saved tuning runs, confirms the
ordered union of their method-specific top threes once, and freezes separate winners.
Final tasks cannot be planned from validation records alone: they require the matching
frozen-winner artifact, then train on a fresh final seed, select an epoch from validation,
persist and restore that checkpoint, and only then open `test_ood`. Canonical stage results,
selection artifacts, ten-seed summaries, per-seed paired differences, and the verified
experiment index are local authority. No real 1,152-run tuning stage was executed while
implementing this system, so this status makes no scientific performance claim.

Final results report mean, standard deviation, and a 95% t-interval across final seeds.
Because methods use the same final seeds, comparisons also report paired per-seed
differences with a 95% t-interval.

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
- mean, standard deviation, and 95% t-interval across the ten final seeds
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

- Whether $p_y=0$ is a required sensitivity experiment
- Conditional/random and nearest-neighbor pair definitions
- Additional baseline methods required for the first complete study
- Whether a reporting-only paired ID rendering of final test sources is useful
- Upper supported Python and future training-accelerator versions beyond the initial
  PyTorch 2.11.0/CUDA 12.8 feature-preparation profile

## Approval checklist

- [x] Source partition sizes and official test role agreed
- [x] Training and validation environment correlations agreed
- [x] Shared-source validation renderings agreed
- [x] Primary robustness and secondary source-only selectors agreed in principle
- [x] Final OOD test isolated from ordinary selection
- [x] Unnormalized OpenAI CLIP ViT-B/32 primary representation approved
- [x] L2-normalized representation sensitivity approved
- [x] Primary and sensitivity pair budgets approved
- [x] Uncentered deterministic projection and rank search approved
- [x] Adam search, tuning/confirmation/final seeds, and aggregation approved
- [x] Reporting uncertainty method approved
- [x] Deterministic partition algorithm approved (`cmnist-stratified-hash-v1`)
- [ ] Estimated-pair definitions approved
