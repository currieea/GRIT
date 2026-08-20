# Rewrite progress

This is the compact checkpoint log for the GRIT rewrite. Update it at milestone
boundaries and after material decisions; do not use it as a raw command transcript.

## Current state

- Branch: `rewrite`
- Active milestone: Approved minimal contract spine; implementation pending (Milestone 3)
- Legacy implementation: Preserved and statically characterized; runtime reproduction deferred
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

### Inherited baseline and package-scaffold follow-ups

- Verified that local `main` and the `main`/`rewrite` merge base are both
  `66c282b6846e8564d165b73d643670be727a2cab`.
- Added a static baseline record covering inherited commands, responsibility boundaries,
  model-selection and oracle-information flow, artifact assumptions, known concerns,
  evidence levels, and runtime limitations.
- Confirmed all 60 inherited W&B sweep launchers optimize `test.acc_avg`, while
  `ERM.report()` separately tracks validation, in-domain-test, and test-best logs without
  restoring a validation-selected checkpoint.
- Defined `src/grit/cli/` plus future `[project.scripts]` entry points as the only path for
  new commands. Existing top-level scripts remain unchecked legacy references.
- Preserved the user-created `.python-version` pin to Python 3.10.20 and documented it as
  the reproducible development interpreter without choosing experiment dependencies or
  an upper compatibility bound.

Verification:

- `uv run python --version` — Python 3.10.20
- `uv lock --check` — passed
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --frozen pytest` — 1 passed on Python 3.10.20
- `git diff --check` — passed
- Static inspection only for the inherited implementation; no dataset download, W&B
  sweep, or costly legacy experiment was run.
- Neither the inherited default data root nor a repository-local data directory was
  available, and no dataset or feature artifacts are tracked.
- The path-restricted legacy diff from the merge base was empty.

### Milestone 3 shared-contract proposal

- Added a concrete, documentation-only contract proposal covering typed experiment
  configuration, dataset/split capabilities, pairs, projections, algorithms/trainers,
  leakage-resistant evaluation, validation-only selection, checkpoint restoration,
  results, provenance, and optional tracking.
- Traced the approved CMNIST and Waterbirds split roles, pair permissions, selectors,
  test restrictions, and multi-seed stages through the proposed lifecycle.
- Documented canonical serialization and tensor-artifact references, four non-executable
  protocol mappings, and named synthetic contract tests for the next implementation goal.
- Recorded evidence-backed inherited algorithm requirements separately from provisional
  future-method needs, without reproducing the inherited ERM ownership hierarchy.
- Integrated a read-only inherited-interface audit and an adversarial leakage/selection
  review, including separate candidate/checkpoint freezes and ordinary/diagnostic schema
  branches.
- At that checkpoint, collected architectural recommendations awaiting user approval and
  kept all unresolved scientific choices explicit. The scope revision below records the
  subsequent approval and reclassification.

Verification:

- Required documents were reread in full; the proposal was checked against both protocol
  traces and received separate read-only inherited-interface and adversarial leakage/
  selection reviews.
- Relative documentation links and milestone status statements were reviewed locally.
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --frozen pytest` — 1 passed on Python 3.10.20
- `git diff --check` — passed
- Path-restricted diffs confirmed no changes to legacy implementation, `pyproject.toml`,
  `uv.lock`, package code, or tests.
- Documentation only; no production contracts, executable configurations, runtime
  dependencies, dataset code, or algorithm code were added.

### Milestone 3 scope revision

- Preserved the detailed contract reasoning while classifying it as approved core,
  provisional CMNIST guidance, or explicitly deferred extension work.
- Approved Pydantic v2 boundary schemas, composition, role-scoped data access,
  validation-only ordinary selection, separate ordinary/test-oracle types, separate
  candidate/checkpoint freezes, inference restoration, bounded algorithm updates,
  canonical local JSON, the null sink, CPU-float64 initial SVD, earlier-epoch ties, and
  CMNIST dual-selector finalist-union handling.
- Narrowed implementation to a CMNIST-focused Pydantic/role/metric/selector/result spine,
  fake-state restoration, canonical round trips, a null sink, and one in-memory leakage
  lifecycle test with a fake bounded update under trainer-owned iteration.
- Deferred generalized artifact storage, faithful training resume, universal containers,
  later-method hooks, accelerator/raw-image abstractions, production W&B, production
  manifests/caches, and pair/projection production code.
- Reordered the next work so the minimal spine is implemented first and revised directly
  through CMNIST, then Waterbirds, before internal names or modules become stable.

Verification:

- Local-link check across the four revised documents — passed with no broken targets
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --frozen pytest` — 1 passed on Python 3.10.20
- `git diff --check` — passed
- Path-restricted diffs confirmed no dependency, package, test, protocol, or legacy
  implementation change was made.

## Approved decisions

- The staged rewrite plan and target architectural direction are approved.
- Milestone 3 uses Pydantic v2 for strict CMNIST-focused configuration/result boundary
  schemas; the dependency is added only with tested implementation code.
- Composition replaces inheritance from `ERM`; algorithms own bounded updates while
  trainers own iteration and lifecycle control.
- Training, validation, final-test, and diagnostic access use role-scoped views, and
  ordinary selectors accept validation metric records only.
- Training-side oracle pairs remain distinct from test-oracle model selection; ordinary
  and CMNIST test-oracle configurations/results are separate discriminated types.
- Hyperparameter/candidate selection is frozen separately from each final run's
  validation-selected checkpoint, which is restored before final-test evaluation.
- Canonical local JSON is authoritative; the initial implementation supplies only a null
  tracking sink and defers production W&B.
- The initial small pair-difference SVD uses deterministic CPU float64 fitting when the
  CMNIST slice implements it.
- Earlier epoch completes the semantic checkpoint tie order before stable identity.
- CMNIST confirms the union of primary/secondary finalists once while retaining separate
  frozen winners.
- The new development scaffold uses a `src/` package layout, Python 3.10 minimum, and
  Python 3.10.20 as its current reproducible development interpreter.
- New command implementations live under `src/grit/cli/` and receive `[project.scripts]`
  entry points only when implemented; inherited top-level scripts remain legacy evidence.
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
- Optional reporting-only paired ID rendering of ColoredMNIST final-test sources
- Waterbirds source acquisition and reconstruction implementation
- Waterbirds conditional/random and nearest-pair definitions
- Method-specific search spaces for GroupDRO and later methods
- Experimental PyTorch, CLIP, CUDA, deterministic-operation, and upper Python versions
- Exact internal type names, fields, and module boundaries pending CMNIST and Waterbirds
- Safe numerical/checkpoint artifact format, retention policy, and runtime conversion
  details

## Provisional and deferred engineering work

- Provisional type/field/module shapes are documented in
  [`contracts.md`](contracts.md) and must be revised from CMNIST and Waterbirds evidence.
- Generalized content-addressed storage, full optimizer/trainer/RNG resume, universal
  checkpoint/numerical containers, and final artifact-format selection are deferred.
- Fish/SWAD/MatchDG/LISA/GroupDRO-specific hooks and raw-image, distributed,
  mixed-precision, compilation, or multi-device abstractions are deferred.
- Production W&B, dataset manifests, feature caches, pair builders, and projection
  mathematics begin only when a real vertical slice requires them.

## Next proposed checkpoint

Implement only the approved Milestone 3 contract spine: CMNIST-focused strict Pydantic
boundaries, role-scoped views and metric types, validation-only selectors, minimal fake-
state restoration, ordinary/test-oracle results, canonical JSON, a null sink, and the
single in-memory leakage lifecycle with a fake bounded update under trainer-owned
iteration. After that spine is verified, exercise and revise it in the CMNIST vertical
slice; do not prebuild deferred frameworks.
