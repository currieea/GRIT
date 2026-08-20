# Rewrite progress

This is the compact checkpoint log for the GRIT rewrite. Update it at milestone
boundaries and after material decisions; do not use it as a raw command transcript.

## Current state

- Branch: `rewrite`
- Active milestone: Shared contracts design (Milestone 3)
- Legacy implementation: Preserved
- New implementation: Package and tooling scaffold only; no algorithms or datasets ported

## Completed checkpoints

### Documentation skeleton

- Added the staged rewrite plan.
- Added the target architecture proposal.
- Added ColoredMNIST and Waterbirds protocol skeletons.
- Added rewrite-oriented agent guidance.

Verification:

- Documentation structure and internal references reviewed locally.
- No implementation or experiment behavior changed.

### ColoredMNIST construction and selection draft

- Defined disjoint 50,000-source training and 10,000-source validation partitions from
  the official MNIST training split.
- Reserved all 10,000 official MNIST test sources for final `0.9` OOD evaluation.
- Defined source-like `0.1` and `0.2` validation renderings plus a neutral `0.5`
  rendering over the same held-out validation sources.
- Defined primary robustness-aware and secondary source-only selectors.
- Separated training-side projection-oracle pairs from test-oracle model selection.
- Added an illustrative YAML contract that distinguishes underlying source partitions
  from rendered environments.

Verification:

- Protocol language and split-access rules reviewed locally.
- No dataset, training, or evaluation code changed.

### Waterbirds construction and selection draft

- Retained released Waterbirds-95 validation and test assignments while adopting the
  paper-defined Waterbirds-CF training construction.
- Defined Waterbirds-CF as 184 landbird and 56 waterbird controlled background-swap
  pairs, preserving the 4,795-record training size and original group proportions.
- Preserved the deliberately balanced official validation split for group-aware
  worst-group model selection.
- Clarified that both endpoints are supervised Waterbirds-CF training records for every
  method, while the 240 pair identities are oracle-only information.
- Added ERM on original Waterbirds as a separately labeled dataset-construction control.
- Excluded the inherited snow/desert categories from canonical Waterbirds and reserved
  any expanded-background study for a separate protocol.
- Defined final test worst-group accuracy, per-group results, and training-distribution
  adjusted average reporting.
- Confirmed that no inherited Waterbirds-CF artifact or construction program is locally
  available; the legacy preprocessing path only consumes a prebuilt artifact.
- Adopted a deterministic server-side reconstruction that reuses canonical WILDS
  Waterbirds, resolves selected foregrounds through CUB and its masks, and generates only
  the 240 required opposite-background endpoints.
- Limited retained Places365 data to 184 water and 56 land training backgrounds from the
  four GroupDRO categories. The complete Places365 archive is an acquisition-format
  contingency, not a persistent experiment dependency.
- Estimated the persistent dataset footprint at approximately 1.5--2 GB when only the
  selected Places backgrounds are retained.

Verification:

- Protocol language, split access, pair-bank access, and legacy-difference notes reviewed
  locally.
- Cross-checked the base split rationale against GroupDRO and the 184/56
  counterfactual-pair construction against the GRIT paper.
- No dataset, training, or evaluation code changed.

### Shared frozen-feature and search protocol

- Chose unnormalized OpenAI CLIP ViT-B/32 features as the primary CMNIST and Waterbirds
  representation, with L2-normalized features as a separately reported sensitivity.
- Deferred raw-image training.
- Chose uncentered pair differences, deterministic full SVD, explicit rank-zero identity
  semantics, and saved spectrum diagnostics.
- Defined a shared Adam learning-rate and weight-decay grid for frozen linear probes.
- Defined three tuning seeds, two additional confirmation seeds for the top three
  candidates, and ten fresh shared final seeds.
- Defined mean, standard deviation, 95% t-intervals, and paired-seed comparisons for final
  reporting.

Verification:

- Cross-checked inherited CLIP preprocessing, optimizer, projection, and sweep behavior
  against the GRIT paper and official OpenAI CLIP examples.
- Documentation-only changes; no experiment code or artifacts were modified.

### Milestone 2 package and tooling scaffold

- Approved the staged rewrite plan and target architectural direction while retaining
  milestone-scoped decisions for concrete interfaces.
- Added the locked `grit-research` package with a `src/grit/` layout and Python 3.10
  development baseline.
- Added empty, dataset-specific configuration roots for ColoredMNIST and Waterbirds;
  executable YAML remains deferred until typed configuration contracts are approved.
- Added Ruff, strict BasedPyright, and Pytest configuration scoped to the new rewrite
  path, plus a minimal installed-package import/version test.
- Kept the inherited `main.py`, `datasets/`, `models/`, `solver/`, `experiments/`, and
  legacy preprocessing scripts unchanged.

Verification:

- `uv lock --python 3.10`
- `uv sync --frozen --group dev`
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --frozen pytest` — 1 passed on Python 3.10.20
- Repeated synchronization and all checks in a fresh temporary environment created
  solely from `uv.lock`; an isolated interpreter imported `grit` version `0.1.0`.
- Confirmed the diff contains no changes to tracked legacy implementation files.

## Approved decisions

- The staged rewrite plan and target architectural direction are approved.
- The new development scaffold uses a `src/` package layout and Python 3.10 minimum.
- The rewrite will prioritize rigorous experiment semantics over matching historical table
  values.
- The inherited implementation remains available during vertical-slice development.
- ColoredMNIST uses all official MNIST test sources only for final `0.9` OOD evaluation.
- ColoredMNIST validation reuses 10,000 held-out source images across `0.1`, `0.2`, and
  `0.5` color renderings; these are repeated views, not independent samples.
- ColoredMNIST reports a primary robustness-aware selector and a secondary source-only
  selector, both fixed before test access.
- ColoredMNIST uses 256 oracle pairs as primary and reports fixed
  32/64/128/256/512-pair sensitivities separately.
- CMNIST and Waterbirds use unnormalized frozen OpenAI CLIP ViT-B/32 features as primary;
  L2-normalized features are a separate sensitivity.
- Frozen linear probes use the approved Adam grid and validation-selected checkpoints.
- Search aggregates three tuning seeds, confirms the top three candidates with two more
  seeds, and evaluates the frozen winner on ten fresh seeds.
- Waterbirds uses the paper-aligned 240-pair Waterbirds-CF training construction and the
  released Waterbirds validation/test splits.
- Waterbirds-CF is reconstructed deterministically on the experiment server; it does not
  depend on locating an inherited CF artifact.
- The reconstruction reuses released Waterbirds and generates only 240 minority
  endpoints from CUB foregrounds/masks and selected Places365 training backgrounds.
- Waterbirds selects ordinary configurations and checkpoints using official validation
  worst-group accuracy; test metrics are final-evaluation-only.
- All Waterbirds-CF methods receive the same supervised records; oracle GRIT additionally
  receives the 240 pair identities.
- Canonical Waterbirds groups are the four binary bird-label/land-water combinations;
  snow and desert are excluded.
- The initial Waterbirds study excludes group-blind and test-oracle selection, raw-image
  training, and expanded snow/desert backgrounds.

## Unresolved decisions

- ColoredMNIST deterministic source-partition algorithm
- ColoredMNIST conditional/random and nearest-pair definitions
- Waterbirds source acquisition and reconstruction implementation
- Waterbirds conditional/random and nearest-pair definitions
- Method-specific search spaces for GroupDRO and later methods
- Experimental PyTorch, CLIP, CUDA, deterministic-operation, and upper Python versions
- Configuration/schema implementation
- Exact algorithm and dataset public interfaces

## Next proposed checkpoint

Propose the Milestone 3 typed configuration, dataset bundle, pair, projection, algorithm,
selection, and result contracts for review. Keep protocol-unresolved construction and
estimated-pair choices out of the interfaces until they are explicitly approved.
