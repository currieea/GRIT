# Waterbirds experiment protocol

Status: **Core scientific and reconstruction protocol approved; source acquisition,
implementation, and estimated-pair details remain**

## Purpose

Define a rigorous Waterbirds-CF experiment for comparing ERM, GRIT/ECMP pairing
strategies, and GroupDRO without test-driven model selection.

The protocol distinguishes the information in the supervised Waterbirds-CF training set
from the oracle pairing relation among those training examples. All primary methods see
the same labeled training records. Oracle GRIT additionally knows which 240 land/water
records share a bird foreground.

## Research questions and scope

The primary question is whether removing feature directions identified by controlled
background interventions improves worst-group bird classification relative to ERM on the
same Waterbirds-CF training set.

The first complete Waterbirds study includes:

- ERM on original Waterbirds as a dataset-construction control;
- ERM on Waterbirds-CF as the primary ERM baseline;
- GroupDRO on Waterbirds-CF;
- GRIT with oracle, conditional, and nearest-neighbor pairs on Waterbirds-CF; and
- rank-zero GRIT as an identity-projection sanity check.

The initial end-to-end implementation may begin with ERM and oracle GRIT before adding
the other approved methods. IRM, REx, Fish, LISA, MatchDG, and SWAD are deferred until
the Waterbirds vertical slice and selection workflow pass.

The initial study does not include raw-image training, group-blind selection,
test-selected model selection, or snow/desert backgrounds.

## Base Waterbirds construction

The base dataset is the released `waterbird_complete95_forest2water2` Waterbirds-95
artifact. It was constructed from CUB-200-2011 bird images and segmentation masks
composited onto Places backgrounds. Its construction uses:

- the official CUB train/test partition;
- 20% of the CUB training partition for validation;
- 95% label/background agreement in training; and
- backgrounds balanced within each bird label in validation and test.

The original authors intentionally balanced validation and test to make rare-group
performance and worst-group model selection less noisy. They also warn that rerunning
the published generator does not reproduce the released artifact exactly because of
random-seed differences. The released base artifact, metadata, and hashes are therefore
canonical.

References:

- [GroupDRO Waterbirds documentation](https://github.com/kohpangwei/group_DRO#waterbirds)
- [Original Waterbirds generation script](https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py)
- [WILDS Waterbirds loader](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/waterbirds_dataset.py)

## Source assets and server acquisition

No inherited Waterbirds-CF artifact or construction program is available in the
repository or in the currently checked server data locations. The legacy preprocessing
path consumes an already-built artifact; it does not construct the controlled images.
The rewrite therefore builds a versioned Waterbirds-CF artifact deterministically on the
experiment server.

The construction reuses released Waterbirds rather than rebuilding its standard images.
It requires:

| Asset | Purpose | Expected retained storage |
|---|---|---:|
| Released Waterbirds through WILDS or GroupDRO | Canonical images, metadata, and splits | A few hundred MB |
| CUB-200-2011 images and annotations | Recover the selected bird foreground sources | About 1.1 GB |
| CUB segmentation masks | Isolate the selected foregrounds | About 37 MB download |
| Places365 training backgrounds | Supply 184 water and 56 land interventions | Less than 100 MB when retaining only selected images |
| Generated images, manifests, and CLIP cache | Waterbirds-CF overlay and features | Well below 100 MB |

The expected persistent footprint is approximately 1.5--2 GB when only the selected
Places images are retained. Retaining all images from the four relevant Places
categories may instead require roughly 0.5--2 GB. These are operational estimates, not
dataset-integrity assertions.

The original GroupDRO construction uses high-resolution Places365 training images from
`bamboo_forest` and `forest/broadleaf` for land and `lake/natural` and `ocean` for water.
The full official high-resolution training archive is approximately 105 GB. Full
Places365 is not an experiment requirement, but its distribution format may make the
full archive a temporary network-transfer requirement.

Server acquisition order:

1. Use category-level official archives if the legacy Places365 access mechanism makes
   them available.
2. Otherwise stream the high-resolution archive and extract only the four required
   categories, avoiding a complete extracted copy.
3. Deterministically retain only the 184 selected water backgrounds and 56 selected land
   backgrounds after the construction manifest and hashes are finalized.
4. Treat the 256-by-256 Places release or a different licensed scene dataset as a
   separately named construction sensitivity, not the primary paper-aligned artifact.

Downloads, caches, and generated data live under configurable server data roots and are
never committed to Git. The preparation command must support existing local asset paths
as well as download/staging mode, verify source hashes where published, and never assume
the author's workstation paths.

References:

- [Official CUB-200-2011 downloads](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Official Places365 downloads](https://places2.csail.mit.edu/download.html)

## Waterbirds-CF training construction

The primary training dataset is the paper's Waterbirds-CF variant, not original
Waterbirds plus an unrestricted counterfactual augmentation bank.

The paper-aligned reconstruction defines the following construction:

1. Load the released Waterbirds metadata and preserve its source IDs and official split
   assignments.
2. Under a recorded construction seed, select without replacement 184 landbirds from
   the landbird-on-land training majority group and 56 waterbirds from the
   waterbird-on-water training majority group.
3. Resolve each selected record to its original CUB image and segmentation mask.
4. Build deterministic background pools from sorted Places filenames in the two land
   and two water categories, shuffle them with the construction RNG, and sample without
   replacement. Use 184 water backgrounds and 56 land backgrounds.
5. Apply the official GroupDRO crop, resize, mask, and compositing geometry to create 184
   landbird-on-water and 56 waterbird-on-land examples.
6. Retain each selected majority Waterbirds image and its generated opposite-background
   version as one controlled pair.
7. Replace the 240 original, unrelated minority training records with the 240 generated
   minority endpoints. Do not append an unrestricted augmentation bank.
8. Keep the released Waterbirds validation and test images and assignments byte-for-byte
   unchanged.

The primary reconstruction seed is part of configuration and the manifest; changing it
creates a different artifact version. Because the paper's original selection and
background-assignment seeds are unavailable, the rewrite does not claim byte-level
identity with the authors' historical Waterbirds-CF artifact.

Thus the expected Waterbirds-CF training set still has 4,795 records:

| Training component | Count | Role |
|---|---:|---|
| Unpaired majority records | 4,315 | Ordinary supervised training |
| Majority endpoints in controlled pairs | 240 | Supervised training and oracle pair endpoints |
| Generated minority endpoints in controlled pairs | 240 | Supervised training and oracle pair endpoints |
| Total supervised training records | 4,795 | Common training set for all primary methods |

The expected group counts remain:

| Bird/background group | Count |
|---|---:|
| Landbird on land | 3,498 |
| Landbird on water | 184 |
| Waterbird on land | 56 |
| Waterbird on water | 1,057 |

These counts are construction invariants enforced by the generator and validated before
feature extraction. The physical representation is an explicit manifest overlay:
4,315 unpaired majority records, 240 existing majority pair endpoints, and 240 generated
minority pair endpoints. Code must never infer pair relationships from directory names,
split labels, or loader ordering.

Primary reference:

- [GRIT paper Waterbirds-CF construction](https://openreview.net/pdf?id=wNQpq4HC5f)

### Labels, backgrounds, and groups

Let:

- `y = 0` denote landbird and `y = 1` denote waterbird;
- `background = 0` denote land and `background = 1` denote water; and
- the evaluation group be the Cartesian product `(y, background)`.

The four canonical groups are landbird-on-land, landbird-on-water,
waterbird-on-land, and waterbird-on-water. Snow and desert do not belong to the
paper-defined Waterbirds-CF protocol. They are excluded from the initial study and would
require a separately named expanded-background protocol.

### Canonical records and manifests

Every supervised record must have an immutable example ID and record at least:

- image path and image hash;
- source CUB image ID and species;
- binary bird label and land/water background;
- logical role, canonical split, and whether the pixels are released or generated;
- whether it is an unpaired, majority-endpoint, or generated-minority record;
- pair ID and endpoint role when applicable;
- Places background asset ID for every generated endpoint;
- source image, mask, and background hashes;
- construction seed and deterministic selection position; and
- dataset version and manifest schema version.

The manifest must record all split, component, group, and pair counts plus the base
Waterbirds, CUB, Places, generator, and configuration identities. The artifact is
accepted only after those counts, hashes, and all pair relationships pass integrity
checks.

## Split and information-access contract

| Resource | Definition | Permitted uses |
|---|---|---|
| Waterbirds-CF train | Expected union of 4,315 unpaired records and all 480 controlled-pair endpoints | Supervised optimization for every primary method; training-only estimated pairing |
| Oracle relation | The 240 exact majority/generated endpoint relationships within Waterbirds-CF train | Oracle GRIT projection estimation only |
| Validation | Released background-balanced Waterbirds validation split | Checkpoint and hyperparameter selection only |
| Test | Released background-balanced Waterbirds test split | Final reporting after selection is frozen |
| Original Waterbirds train | Released unmodified training split | ERM dataset-construction control only |

All primary Waterbirds-CF methods must receive exactly the same supervised training
record IDs. Oracle access consists of the pair relation, not additional images or labels.
The generated minority endpoints are ordinary Waterbirds-CF training records and must
not be described as an optional augmentation.

Validation already contains held-out examples from all four groups and was deliberately
balanced for stable worst-group tuning. No additional held-out background domain is
needed for the canonical experiment.

### Method-specific information access

- ERM receives Waterbirds-CF training images and bird labels but not pair identities or
  training background labels.
- Oracle GRIT receives the same supervised records plus the 240 exact pair identities.
- Conditional and nearest GRIT receive training labels and background metadata only as
  required by their approved pair-builder definitions; they do not receive oracle pair
  identities.
- GroupDRO may use `(y, background)` group labels during training because this is part
  of the method definition.
- Every method may use validation group metadata through the prespecified primary
  selector.
- Test samples, labels, group metadata, and metrics are unavailable to training,
  projection, checkpoint selection, and hyperparameter selection.

## Invariant pairs

### Clean oracle pairs

A clean Waterbirds-CF oracle pair holds the bird foreground fixed while changing the
background category:

$$
(x_i^{\text{land}}, x_i^{\text{water}}).
$$

The primary oracle uses exactly the 184 landbird pairs and 56 waterbird pairs defined by
the Waterbirds-CF construction.

Rules:

- Both endpoints must be members of the validated Waterbirds-CF training set.
- Validation and test birds are forbidden.
- Endpoints have the same source CUB image, bird label, foreground pixels, segmentation
  mask, crop, scale, and placement.
- Endpoints differ in canonical land/water background.
- Pair feature extraction uses identical deterministic preprocessing. Independent random
  crops or augmentations are forbidden.
- Pair orientation is canonicalized as `land - water`.
- Pair membership comes from an explicit manifest, never loader ordering.

The 240-pair budget is fixed for the primary paper-aligned experiment. Smaller seeded,
label-stratified subsets may be reported as a pair-budget sensitivity. A new pair for
every training bird would be a stronger-information oracle and must be labeled
`full_pair_oracle`; it does not replace the primary 240-pair result.

The pair manifest must record:

- pair ID and source CUB ID;
- bird label and both supervised record IDs;
- endpoint background categories and background asset IDs;
- segmentation-mask identity and hash;
- crop, scale, placement, interpolation, and compositing parameters;
- reconstruction seed and deterministic source/background selection positions;
- endpoint image and feature hashes; and
- generator and manifest schema versions.

### Original-Waterbirds control

ERM on original Waterbirds measures whether replacing the original minority records with
controlled counterfactuals materially changes the ERM task. This is a separately labeled
dataset-construction control. It is not the primary ERM baseline for GRIT, because GRIT
and its fair ERM comparator must train on the same Waterbirds-CF records.

### Estimated pairs

Conditional/random pairs use different Waterbirds-CF training examples with the same
bird label and opposite background values. Nearest-neighbor pairs use the same eligibility
constraint and choose the nearest eligible training representation without consulting
oracle pair IDs.

Both builders must:

- use Waterbirds-CF training records only;
- produce 240 pairs for the primary comparison;
- save explicit endpoint IDs and construction provenance;
- use deterministic tie-breaking; and
- expose endpoint reuse and replacement policies in configuration.

The nearest-search distance, reuse policy, and exact conditional sampling algorithm
remain unresolved.

## Frozen representation and classifier

The primary representation is frozen OpenAI CLIP ViT-B/32 using the official OpenAI
weights and deterministic evaluation preprocessing. The cache stores the 512-dimensional
unnormalized `encode_image` output.

Unnormalized features are primary because they match the paper and inherited linear
probe and preserve the additive Euclidean feature geometry assumed by linear GRIT.
Per-example L2 normalization is a prespecified, separately reported representation
sensitivity. In that sensitivity, every training, validation, test, and pair-endpoint
feature is normalized before pair differences or classifier fitting.

The initial classifier is a linear two-class head trained with Adam:

- batch size: 256;
- maximum epochs: 100;
- learning-rate candidates: `[1e-4, 3e-4, 1e-3, 3e-3]`; and
- weight-decay candidates: `[0, 1e-5, 1e-4, 1e-3]`.

Raw-image/end-to-end training is deferred. It requires a separate future protocol and
must not be aggregated with frozen-feature results.

The feature manifest records encoder package and version, model/checkpoint identity and
hash, preprocessing, normalization mode, source image manifest hash, output shape,
dtype, device/precision details, and feature-file hash.

## Projection

For feature rows `z_land` and `z_water`, construct the uncentered difference matrix

$$
D_i = z_i^{\text{land}} - z_i^{\text{water}}.
$$

The primary protocol does not subtract the mean difference. GRIT removes the selected
right-singular-vector subspace of `D`. Projection rank is the number of removed
directions; rank zero is the exact identity operation.

Use deterministic full `torch.linalg.svd` rather than randomized
`torch.svd_lowrank`. The implementation must:

- reject ranks above `min(number_of_pairs, feature_dimension)`;
- distinguish requested, numerical, and effective rank;
- use an explicit dtype and relative singular-value tolerance;
- produce an orthogonal projector within a tested tolerance; and
- save singular values, explained-energy diagnostics, tolerance, and effective rank.

The initial rank candidates are every integer from 0 through 24, extending the paper's
2-through-24 range with rank one and an identity control.

## Group evaluation

Evaluation uses the four `(y, background)` groups.

- Worst-group accuracy is the minimum of the four group accuracies and is the primary
  robustness metric.
- Every group count and accuracy is always reported.
- Adjusted-average accuracy weights group accuracies by the validated Waterbirds-CF
  training proportions and is the primary average-accuracy companion.
- Raw sample-average accuracy may also be reported but is labeled `raw_average`.
- A missing expected group is an integrity failure rather than a silently ignored group.

## Model and checkpoint selection

The only ordinary Waterbirds selector maximizes official validation worst-group
accuracy.

Deterministic tie-breakers are:

1. Higher validation adjusted-average accuracy
2. Lower projection rank when comparing configurations
3. Earlier epoch when comparing checkpoints
4. Stable configuration ordering

Checkpoints and configurations obey these rules:

- A checkpoint sees validation metrics only.
- The selected checkpoint is restored before final test evaluation.
- A configuration is ranked only after aggregating its validation score across tuning
  seeds.
- The best individual seed is never the selection unit.
- The selected hyperparameter configuration is frozen before final-evaluation runs.
- Each final run may select its checkpoint epoch using validation, but cannot change the
  frozen hyperparameters.
- Test evaluation starts only after the selection artifact is finalized.

No group-blind selector is included in the initial study. No test metric may select a
rank, optimizer setting, method parameter, seed, or checkpoint. The initial study does
not publish a test-oracle envelope.

## Search, confirmation, and final seeds

The search is configuration-driven and locally reproducible; W&B may mirror results but
does not define selection.

- Every candidate runs on three tuning seeds.
- Candidate ranking uses mean validation worst-group accuracy over those seeds.
- The top three configurations receive two additional confirmation seeds.
- The winner is chosen using its combined five-seed validation mean.
- The selected configuration is evaluated on ten fresh final seeds shared across
  methods.
- Final reporting uses only the ten final seeds, not tuning or confirmation seeds.
- Method-specific spaces and budgets are declared in advance; methods are not forced to
  waste trials merely to have identical trial counts.

For ERM and GRIT, the approved shared optimizer grid is the Cartesian product of the
learning-rate and weight-decay candidates above. GRIT additionally searches the approved
rank candidates. Later methods add only their prespecified method-specific parameters.

Final results report mean, standard deviation, and a 95% t-interval across final seeds.
Because methods use the same final seeds, method comparisons also report paired
per-seed differences with a 95% t-interval.

## Configuration contract

Experiment settings belong in validated YAML rather than executable sweep modules.

Illustrative schema:

```yaml
data:
  name: waterbirds_cf
  base_artifact: waterbird_complete95_forest2water2
  base_source: wilds
  expected_train_count: 4795
  expected_pair_count: 240
  validation_split: val
  test_split: test

construction:
  version: waterbirds_cf_v1
  seed: 0
  cub_root: ${CUB_ROOT}
  places_root: ${PLACES365_ROOT}
  land_categories: [bamboo_forest, forest/broadleaf]
  water_categories: [lake/natural, ocean]
  sample_backgrounds_without_replacement: true
  retain_selected_backgrounds_only: true
  preserve_released_validation_and_test: true

features:
  encoder: openai_clip
  model: ViT-B/32
  normalize: false

pairs:
  strategy: oracle
  relation_source: generated_manifest
  num_pairs: 240
  endpoint_records_are_supervised: true

projection:
  center_differences: false
  ranks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

training:
  optimizer: adam
  learning_rates: [0.0001, 0.0003, 0.001, 0.003]
  weight_decays: [0.0, 0.00001, 0.0001, 0.001]
  batch_size: 256
  max_epochs: 100

selection:
  split: validation
  metric: worst_group_accuracy
  group_fields: [label, background]
  tie_breakers: [adjusted_average_accuracy, lower_projection_rank,
                 earlier_epoch]
```

This is a protocol example, not approval of a particular configuration library.

Typed validation must reject at least:

- supervised Waterbirds-CF training counts inconsistent with the generated manifest;
- pair endpoints outside the supervised Waterbirds-CF training records;
- an oracle method without exactly 240 valid pair relationships;
- estimated pair builders receiving oracle pair identities;
- selectors referencing train or test metrics;
- test access before configuration selection is frozen;
- missing or unexpected evaluation groups;
- oracle endpoints with different source birds, foreground geometry, or labels;
- snow/desert records presented as canonical groups; and
- normalized and unnormalized feature artifacts mixed within one run.

## Metrics and reporting

Required validation output:

- all four group counts and accuracies;
- worst-group, adjusted-average, and raw-average accuracy;
- selected epoch, configuration, selector, and checkpoint; and
- per-seed and aggregated score used to choose the configuration.

Required final output:

- test worst-group accuracy as the primary result;
- all four test group counts and accuracies;
- adjusted-average and raw-average test accuracy;
- mean, standard deviation, and 95% interval across final seeds;
- paired method differences where comparisons are made;
- resolved configuration and selected checkpoint epoch per seed; and
- pair strategy, pair count, projection rank, and projection diagnostics.

Original-Waterbirds controls and normalized-feature sensitivities must be labeled by
dataset and representation. They do not replace the primary unnormalized
Waterbirds-CF result.

## Reproducibility requirements

Use separate seeds for Waterbirds-CF construction, pair subsampling,
training, and data-loader order. Dataset and pair artifacts remain fixed while training
seeds vary.

Each completed run records:

- base and Waterbirds-CF dataset hashes;
- validated split, component, group, and pair counts;
- resolved configuration;
- feature encoder, weights, preprocessing, normalization, and cache manifest;
- pair manifest and generator/reconstruction version;
- construction, sampling, and training seeds;
- Git revision and dirty state;
- dependency and device information; and
- structured training, selection, checkpoint, and final-evaluation results.

The same artifact configuration and seeds must reproduce the same records, pairs,
features, search ordering, and selection result.

## Required leakage and integrity tests

- Released train, validation, and test source IDs are mutually disjoint.
- Waterbirds-CF supervised training has the expected total and group counts.
- Exactly 240 oracle relationships map 240 majority endpoints to 240 generated minority
  endpoints within supervised training.
- Every oracle pair preserves source bird, label, foreground geometry, and mask while
  changing land/water background.
- ERM and all GRIT variants receive identical supervised Waterbirds-CF record IDs.
- ERM cannot access pair identities.
- Estimated pair builders cannot access oracle identities.
- Validation records do not enter optimization or projection fitting.
- Test records and metrics cannot reach training, pairing, checkpoint selection, or
  hyperparameter selection.
- Unnormalized and L2-normalized artifacts cannot be mixed.
- Pair direction and pair subsets are deterministic under fixed seeds.
- Search ranks aggregated validation configurations rather than individual seeds.
- The selected checkpoint is restored before final test evaluation.
- Adjusted-average weights match validated training group proportions.
- Snow/desert records cannot be loaded as canonical Waterbirds-CF groups.
- Search summaries can be recomputed from saved per-run records.

## Legacy comparison

The inherited path remains historical evidence, but no inherited Waterbirds-CF artifact
is present in the repository or the checked data locations. Its `train +
counterfactual` merge appears intended to consume the paper's 4,795-record
Waterbirds-CF layout rather than to create it. It is not a construction implementation
and is not a prerequisite for the rewrite.

Known inherited limitations remain:

- no checked-in code constructs the paper's 184/56 counterfactual selection and
  background replacement;
- the loader infers pair relationships from alternating grouped-loader output instead
  of an explicit manifest;
- physical split counts and pair endpoints are not validated;
- metadata admits unexplained snow/desert backgrounds;
- frozen Waterbirds experiments use SGD in code even though the paper specifies Adam;
- feature and pair artifacts lack complete provenance; and
- W&B sweeps optimize test average accuracy.

Historical results and ranges may be retained as diagnostics, but test-selected winners
are not valid ordinary selections and numerical parity is not an exit requirement.

## Remaining decisions and required evidence

- Acquire canonical Waterbirds, CUB images, CUB masks, and the required official
  Places365 training backgrounds on the server.
- Record and verify all available published source hashes and licenses/terms.
- Implement and validate the versioned construction without claiming byte-level identity
  with the unavailable historical artifact.
- Pin the exact Pillow/torchvision interpolation and image-encoding behavior used by the
  GroupDRO-compatible compositor.
- Approve the conditional/random sampling algorithm and nearest-neighbor distance/reuse
  policy.
- Set method-specific search ranges for GroupDRO and the estimated-pair variants.
- Pin supported Python, PyTorch, CLIP, CUDA, and deterministic-operation versions.

## Approval checklist

- [x] Original Waterbirds split roles and binary groups agreed
- [x] Paper-aligned 184/56 Waterbirds-CF construction adopted
- [x] Primary oracle budget fixed at 240 controlled training pairs
- [x] All primary methods receive the same supervised Waterbirds-CF records
- [x] Oracle access defined as pair identities rather than extra data
- [x] Official validation worst-group accuracy is the only primary selector
- [x] Final test isolated from all ordinary selection
- [x] Unnormalized OpenAI CLIP ViT-B/32 is the primary representation
- [x] L2-normalized features are a separately reported sensitivity
- [x] Adam optimizer search, seed aggregation, and uncertainty protocol approved
- [x] Initial method and diagnostic scope approved
- [x] Deterministic server-side Waterbirds-CF reconstruction plan approved
- [x] Minimal retained Places subset and storage plan approved
- [ ] Source datasets acquired and hashes verified on the experiment server
- [ ] Waterbirds-CF generator and integrity checks implemented
- [ ] Conditional and nearest-pair details approved
- [ ] Later-method search spaces approved
