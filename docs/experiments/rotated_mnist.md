# RotatedMNIST experiment protocol

Status: **Approved for the first ERM versus oracle-GRIT vertical slice.**
GroupDRO, V-REx, IRMv1, and estimated pairing are deferred.

## Purpose

Define a validation-selected RotatedMNIST experiment on frozen CLIP features. The first
reportable result compares ERM with GRIT using exact training-image correspondences.
The construction borrows the useful rotation settings from the inherited experiment,
but deliberately does not inherit its inconsistent launch-time split overrides or its
test-selected model selection.

This document is the scientific protocol contract. Configuration validation and data
construction must reject violations of its split, pairing, or information-access rules.

## Source partitions and rotations

Underlying MNIST images are partitioned before any rotation is rendered. Stable source
identity is the official MNIST split plus its official source index.

| Source partition | MNIST source | Count | Permitted uses |
|---|---:|---:|---|
| `train_r0_sources` | Official train | 25,000 | Classifier optimization at 0 degrees; train-only pair sourcing |
| `train_r45_sources` | Official train | 25,000 | Classifier optimization at 45 degrees; train-only pair sourcing |
| `validation_sources` | Official train | 10,000 | Validation rendering and selection only |
| `test_sources` | Official test | 10,000 | Final 90-degree evaluation only |

The official-training partitions are mutually disjoint. The official MNIST test split
is disjoint by construction, and its cached feature table remains unavailable to the
training and selection interfaces until ordinary selection is frozen.
The approved partitioner is `rotated-mnist-stratified-hash-v1`: within each digit, it
uses largest-remainder apportionment for the exact requested totals and orders source
IDs by SHA-256 of the NUL-separated method ID, construction seed, official split, and
official index. The official index is the deterministic collision fallback. This is the
same order-independent partitioning principle used by the CMNIST protocol.

| Environment | Role | Sources | Rotation | Permitted uses |
|---|---|---|---:|---|
| `train_r0` | Train | `train_r0_sources` | 0 degrees | Optimization |
| `train_r45` | Train | `train_r45_sources` | 45 degrees | Optimization |
| `val_r0` | Validation | `validation_sources` | 0 degrees | Selection and reporting |
| `val_r45` | Validation | `validation_sources` | 45 degrees | Selection and reporting |
| `val_r60` | Validation | `validation_sources` | 60 degrees | Selection and reporting |
| `test_r90` | Test | `test_sources` | 90 degrees | Final reporting only |

The three validation environments are repeated renderings of the same 10,000 held-out
sources, not independent examples. The intermediate 60-degree validation environment
tests rotation robustness without rendering the final 90-degree target domain. Final
source pixels, features, and metrics must not enter checkpoint, rank, or hyperparameter
selection.

Rendering is deterministic: MNIST tensors are rotated about the image center using
bilinear interpolation with zero fill, values remain in `[0, 1]`, and the grayscale
channel is repeated into RGB before CLIP preprocessing. Rotation and renderer identity
are recorded in the dataset manifest.

## Frozen representation and classifier

The primary representation is frozen OpenAI CLIP ViT-B/32 with the same official
weights, deterministic evaluation preprocessing, and unnormalized 512-dimensional
`encode_image` output as CMNIST. Feature preparation is deterministic single-device
float32 computation with mixed precision and TF32 disabled. Training is CPU-only.

The classifier is a linear ten-class head trained with Adam:

- batch size: 256;
- maximum epochs: 40;
- learning-rate candidates: `[1e-4, 3e-4, 1e-3, 3e-3]`; and
- weight-decay candidates: `[0, 1e-5, 1e-4, 1e-3]`.

The feature manifest records encoder, checkpoint, preprocessing, source-manifest,
runtime, shape, dtype, normalization, and file identities under the same provenance
requirements as CMNIST.

## Oracle invariant pairs

An oracle pair is the **same training source image** rendered at 0 and 45 degrees:

$$
(x_i^{0}, x_i^{45}).
$$

It is not a same-class match between two different handwritten digits. Pair rules are:

- sources come only from `train_r0_sources` and `train_r45_sources`;
- validation and official-test sources are forbidden;
- 256 unique sources are selected without replacement by a deterministic pair seed;
- endpoints have the same official source ID and digit label;
- endpoint orientation is fixed as 0 degrees minus 45 degrees;
- pair endpoints estimate the projection only and are not appended to supervised
  classifier training; and
- pair source IDs, endpoint identities, selection positions, parameters, and artifact
  hashes are recorded.

A pair source may also occur in its ordinary training environment. This matches the
CMNIST treatment of auxiliary training-side invariance information. Prespecified ranks
are every integer from 2 through 24 inclusive. The pair budget and rank grid are fixed,
not selected using final results.

## Validation-only selection

Every candidate is evaluated on all validation environments. Two prespecified selectors
are applied to the same saved metrics.

The primary robustness score is

$$
s_{\mathrm{robust}} = \min\{\operatorname{acc}(\text{val\_r0}),
                                  \operatorname{acc}(\text{val\_r45}),
                                  \operatorname{acc}(\text{val\_r60})\}.
$$

The secondary source score is

$$
s_{\mathrm{source}} = \min\{\operatorname{acc}(\text{val\_r0}),
                                \operatorname{acc}(\text{val\_r45})\}.
$$

For both selectors, ties are resolved by higher mean accuracy across the selector's
environments, lower projection rank, then stable configuration ordering. Checkpoint
selection and configuration selection use validation records only. The selected
checkpoint is restored before final evaluation.

Each candidate runs on three tuning seeds. The top three configurations per method and
selector receive two additional confirmation seeds. The combined five-seed validation
mean freezes one configuration, which is then evaluated on ten fresh final seeds shared
between ERM and GRIT. Tuning and confirmation seeds do not enter the final estimate.
The final report contains mean accuracy, a 95% confidence interval over the ten seeds,
and paired GRIT-minus-ERM differences because both methods share final seeds.

## Information-access contract

- ERM receives only the two supervised training environments and their digit targets.
- Oracle GRIT receives those same supervised records plus the 256 exact training-source
  pair relationships.
- Validation records may be used only for checkpoint and configuration selection.
- No rendering of an official-test source is exposed to a selector. Its prepared feature
  table can be opened only after selection is frozen, and its metric is evaluated once
  per final run.
- Every reported run records its seed, resolved config, commit identity, and dataset,
  feature-cache, and pair-bank hashes.

The inherited RotatedMNIST code and DomainBed are implementation references rather than
protocol authority. In particular, DomainBed's construction assigns each source to one
rotation and therefore cannot supply exact same-source oracle pairs, while inherited
launchers used mutually inconsistent rotation splits. Neither behavior is copied here.
