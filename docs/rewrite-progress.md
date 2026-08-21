# Rewrite progress

This is the compact checkpoint log for the GRIT rewrite. Update it at milestone
boundaries and after material decisions; do not use it as a raw command transcript.

## Current state

- Branch: `rewrite`
- Active milestone: Waterbirds ERM/oracle-GRIT vertical slice review (Milestone 5)
- Legacy implementation: Preserved and statically characterized; runtime reproduction deferred
- New implementation: Reviewed CMNIST plus implemented, server-ready, hermetically
  verified Waterbirds-CF ERM/oracle-GRIT vertical slice awaiting user review

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

### Milestone 3 minimal contract spine

- Added the bounded `pydantic>=2.11,<3` runtime dependency and strict, frozen,
  unknown-field-rejecting CMNIST configuration and result boundaries.
- Added protocol-bound split descriptors and role-scoped in-memory views. Final-test
  examples remain behind a gate requiring mutually matching frozen candidate, final-run
  checkpoint selection, and restoration receipt.
- Added structurally distinct validation, final-test, and CMNIST test-oracle diagnostic
  metrics plus validation-only primary/secondary selectors with deterministic mean, rank,
  earlier-epoch, and stable-identity ties.
- Bound oracle-GRIT configuration to the two approved training source partitions and 256
  pairs. Bound candidate freezes to the configured three tuning plus two confirmation
  seeds and final checkpoints to the configured ten-seed set.
- Kept selector-independent scientific candidate identity distinct from the full resolved
  configuration digest so primary and secondary selection can reuse saved validation
  records while retaining separate winners and result identities.
- Added separate candidate and per-final-run checkpoint freezes, minimal fake-state
  checkpoint restoration, canonical JSON/digests, ordinary versus diagnostic result
  unions, and a null event sink.
- Added an in-memory fake algorithm/trainer lifecycle proving trainer-owned iteration,
  validation-only checkpoint choice, restoration of selected rather than last state, and
  one-way final-test access. Canonical result parsing revalidates method, selector, seed,
  checkpoint, contributor, and metric identities.
- Kept all type and module names internal and revisable. Added no real dataset access,
  feature cache, pair builder, projection mathematics, PyTorch training, W&B integration,
  generalized artifact store, or faithful resume framework.

Verification:

- `uv lock --check` — passed
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --frozen pytest` — 25 passed on Python 3.10.20
- `uv run --offline --frozen pytest` — 25 passed without network access
- `git diff --check` — passed
- Documentation links, milestone statuses, and scope paths reviewed; no legacy or protocol
  file changed.

### Milestone 3 contract correction

- Replaced the single-winner-shaped CMNIST diagnostic result with a genuine test-oracle
  envelope. Eligible records retain distinct configuration/rank, run, checkpoint, epoch,
  seed, candidate, method, and metric identities; maximum `test_ood` accuracy plus an
  explicit stable identity tie-break selects the fully traced winner.
- Connected tuning, finalist, confirmation, and freeze boundaries. Each method/selector's
  exact three-seed tuning table produces an ordered top-three artifact; confirmation uses
  the deduplicated primary/secondary union, accepts only fresh confirmation records, and
  reuses the tuning decisions embedded in each selector's artifact. Each five-seed
  comparison and freeze remain restricted to that artifact.
- Added strict rejection for missing, duplicate, extra, and mis-staged seeds, non-finalist
  freezes, cross-selector finalist use, cross-method aggregation, inconsistent diagnostic
  identities, and diagnostic records passed to ordinary selectors.
- Enabled Pydantic instance revalidation and explicitly reconstruct external selector,
  seed-set, decision, and finalist inputs at public boundaries so ordinary parsing paths do
  not trust malformed `model_copy()` values.
- Bound strict BasedPyright explicitly to uv's project `.venv`; strict mode and the existing
  `src/grit`/`tests` include paths remain unchanged.
- Added no dataset, feature, pair-construction, projection, PyTorch training, production
  runner, W&B, artifact-store, or resume implementation.

Verification:

- `uv lock --check` — passed
- `uv run --frozen ruff check .` — passed
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes
- `uv run --offline --frozen pytest` — 27 passed on Python 3.10.20
- `git diff --check` — passed

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
- ColoredMNIST partitions official training sources with
  `cmnist-stratified-hash-v1`: digit-stratified largest-remainder allocation plus
  source-identity SHA-256 ordering.
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
- Production W&B remains deferred. The CMNIST slice implements only its required narrow
  dataset, feature-cache, pair, projection, and selected-checkpoint boundaries; it does not
  establish generalized storage or resume frameworks.

### Milestone 4 ColoredMNIST vertical slice

- Approved and implemented `cmnist-stratified-hash-v1`, with exact production source
  counts, deterministic environment rendering, and explicit repeated validation views.
- Added a training-source-only 256-pair clean oracle, red-minus-green orientation, and a
  deterministic CPU-float64 full-SVD projection with rank-zero identity behavior.
- Added the pinned official OpenAI CLIP ViT-B/32 cache boundary and a deterministic fake
  encoder used only by the non-reportable hermetic smoke path.
- Added composed linear ERM/GRIT training, validation-only checkpoint selection, narrow
  inference-checkpoint persistence/restoration, and the one-way final-test gate.
- Connected the three-seed tuning, primary/secondary top-three finalist artifacts, ordered
  confirmation union, two confirmation seeds, separate five-seed selector winners, and ten
  final seeds. Confirmation consumes only fresh confirmation records and reuses the exact
  tuning decisions stored in the applicable finalist artifact.
- Added `grit-cmnist-prepare` as the explicit download/cache boundary and
  `grit-cmnist-run` as the strict hermetic smoke runner. Canonical local JSON remains
  authoritative; no W&B or generalized artifact framework was added.

Verification:

- `uv lock --check` — passed (41 packages resolved).
- `uv run --frozen ruff check .` — passed.
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes.
- `UV_CACHE_DIR=/tmp/grit-uv-cache uv run --offline --frozen pytest` — 50 passed on
  Python 3.10.20.
- `UV_CACHE_DIR=/tmp/grit-uv-cache uv run --offline --frozen grit-cmnist-run
  configs/cmnist/smoke.yaml` — passed and produced 40 ordinary final-result records from
  the non-reportable fake-feature lifecycle.
- `git diff --check` — passed before the checkpoint commit.
- No real MNIST or CLIP artifact was downloaded, and no reportable scientific sweep was
  executed.

### Milestone 4 artifact-lineage correction

- Bound the CMNIST pair-source capability to the validated dataset manifest and exact
  official-training-pool content digest. The oracle builder now derives its dataset
  dependency internally, and dataset identity contributes to every stable pair ID.
- Made feature preparation reject mixed construction/pair identities, pair-set/manifest
  disagreement, inconsistent record metadata, and invalid clean recoloring endpoints
  before creating output or encoding inputs.
- Required the exact pair-manifest and feature-cache-manifest digests in canonical
  projection diagnostics and passed the validated runner identities used for fitting. The
  internal pair-manifest and projection-diagnostic schema versions advance because these
  identity fields are now required.
- Bound final-test handles/views to the exact feature-cache manifest, so another cache with
  identical test source IDs but different encoder, normalization, or feature content is
  rejected.
- Restricted the CMNIST oracle pair construction ID to
  `cmnist-clean-oracle-pairs-v1`. Added no generalized artifact framework or new execution
  scope.

Verification:

- `uv lock --check` — passed (41 packages resolved).
- `uv run --frozen ruff check .` — passed.
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes.
- `UV_CACHE_DIR=/tmp/grit-uv-cache uv run --offline --frozen pytest` — 56 passed on
  Python 3.10.20, including the deterministic end-to-end smoke lifecycle.
- `git diff --check` — passed before the checkpoint commit.
- No real dataset/feature download, reportable sweep, Waterbirds work, W&B integration,
  or legacy implementation change was performed.

## Milestone 5 implementation checkpoints

### Milestone 5 checkpoint: deterministic Waterbirds-CF construction

- Added strict parsers for caller-supplied released Waterbirds, CUB image/mask metadata,
  and the four approved Places categories; the implementation does not download assets.
- Added the versioned `waterbirds-cf-sha256-v1` construction. Stable SHA-256 ranking makes
  the 184/56 source and opposite-background selection seeded, without replacement, and
  independent of input enumeration order.
- Reproduced the official GroupDRO center-crop, LANCZOS resize, mask, and uint8 composite
  geometry while recording source, mask, foreground, background, generated-image, and
  geometry identities.
- The canonical dataset manifest validates production split/component/group counts,
  explicit land-minus-water relationships, unique endpoints, construction lineage, and
  byte-preserved released validation/test records. A separate strict fixture profile is
  always non-reportable and cannot weaken the production inventory.
- Hermetic tests cover replacement/count invariants, order and seed behavior, released
  byte preservation, compositor behavior, canonical round trips, tampered relationships,
  strict production counts, and missing approved background categories.
- Production source acquisition and execution remain deferred to the experiment server;
  no real Waterbirds, CUB, Places, or CLIP asset was downloaded and no reportable result
  was produced.

### Milestone 5 checkpoint: scoped supervision and oracle relationships

- Added a common Waterbirds-CF supervised capability whose records expose image identity
  and bird label but structurally omit training background and pair identities. ERM and
  oracle GRIT therefore receive identical endpoint record IDs without giving ERM oracle
  relation information.
- Added a separate oracle-relation capability and canonical pair manifest. It revalidates
  the dataset boundary, retains exact endpoint/construction provenance, binds the dataset
  manifest digest, fixes land-minus-water orientation, and enforces 184 landbird plus 56
  waterbird relations for production.
- Added a validation-only view that exposes the four approved group fields to evaluation,
  not optimization. View issuance rechecks all underlying image bytes against the dataset
  manifest.
- Hermetic tests prove common supervised membership, ERM metadata redaction, separate
  oracle access, fixture/production strata, canonical round trips, tamper rejection, and
  changed-image rejection. Estimated pair builders remain deferred.

### Milestone 5 checkpoint: frozen features and oracle projection

- Extended the pinned official OpenAI CLIP adapter to accept variable-sized PIL images
  through the same model preprocessing, without changing CMNIST behavior or downloading
  weights implicitly.
- Added a Waterbirds-CF float32 feature cache whose canonical manifest binds the dataset,
  image/order metadata, encoder revision, exact weights/preprocessing identity,
  normalization, referenced array shape/dtype, and byte digest. Cache output is written
  only after construction and encoder output validate.
- Added role-scoped cached tables: supervised training omits backgrounds while validation
  carries four-group metadata. Normalized and unnormalized caches have distinct identities
  and cannot satisfy one another's load requirements.
- Oracle pair feature resolution requires the same dataset manifest and rechecks both
  endpoints' training roles, labels, backgrounds, image hashes, and canonical land-minus-
  water ordering before fitting.
- Reused the small deterministic full `torch.linalg.svd` implementation on CPU float64.
  Diagnostics canonically retain the exact Waterbirds pair/cache digests, complete
  spectrum, requested/effective ranks, and numerical residuals; rank zero remains the
  identity.
- Hermetic tests cover cache determinism across preparation batch sizes, canonical round
  trips, file tampering, cross-construction pair rejection, normalization separation,
  write-late failure, exact projection lineage, and rank-zero behavior. No real CLIP
  weights or Waterbirds assets were used.

### Milestone 5 checkpoint: four-group metrics and validation selection

- Added strict, dataset-specific validation and final-test metric types with four required
  group counts/accuracies, worst-group, raw-average, and adjusted-average values. Boundary
  validation recomputes every aggregate and binds adjusted weights to the validated
  Waterbirds-CF training group counts.
- Implemented validation-only checkpoint selection by worst-group accuracy, adjusted
  average, earlier epoch, and stable checkpoint identity. Final-test metric values are a
  different type and are rejected by the ordinary selector API.
- Implemented exact three-seed tuning ranking, durable ordered top-three finalists, and
  exact two-seed confirmation. Confirmation reuses the tuning checkpoint decisions stored
  inside the finalist artifact; it cannot replace them with caller-supplied copies.
- Five-seed candidate freezes are selector/method/config/rank bound. Fresh final seeds may
  choose an epoch using validation but cannot change the frozen candidate or
  hyperparameters. Candidate ties use adjusted average, lower projection rank, then stable
  identity; methods are selected independently.
- Hermetic tests cover training-proportion weighting, missing groups, all tie stages,
  missing/extra/wrong-stage seeds, finalist membership, retained tuning decisions,
  method separation, final checkpoint freezing, and final-metric rejection.

### Milestone 5 checkpoint: training, final gate, results, and commands

- Reused the compositional linear-probe algorithm for Waterbirds ERM and oracle GRIT.
  Algorithms own bounded Adam updates; the Waterbirds trainer owns shuffled batch/epoch
  iteration, four-group validation, and in-memory epoch checkpoints.
- Added the narrow selected-inference-checkpoint persistence path and a Waterbirds-specific
  restoration receipt. No optimizer, RNG, scheduler, or resume state is promised.
- Bound an opaque final-test handle to the intended run, frozen candidate, final validation
  decision, restored checkpoint, seed, projection rank, and exact feature-cache manifest.
  Only the opened view can produce the structurally distinct final-test metric.
- Added strict resolved candidate and canonical run-result schemas. Results revalidate the
  entire candidate/checkpoint/restoration/validation/final identity chain and reference
  dataset, feature, pair, projection, and selected-checkpoint artifacts rather than
  embedding tensors.
- Added `grit-waterbirds-prepare`, which consumes explicit local server assets and never
  acquires Waterbirds/CUB/masks/Places, plus `grit-waterbirds-run`, which consumes only the
  strict non-reportable offline smoke profile. The null event sink remains the default;
  W&B is absent.
- The smoke lifecycle constructs fixture assets, caches fake features, fits the real
  CPU-float64 projection, runs exact 3+2 validation selection independently for ERM and
  GRIT, restores one checkpoint for each of ten final seeds, then writes canonical local
  group results. Its summary retains per-seed worst-group, adjusted-average, and raw-average
  values with mean, sample standard deviation, 95% t-intervals, and paired GRIT-minus-ERM
  worst-group differences. It is expressly non-reportable and contains no test-oracle
  branch.

Verification:

- `uv lock --check` — passed (41 packages resolved).
- `uv run --frozen ruff check .` — passed.
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes.
- `UV_CACHE_DIR=/tmp/grit-uv-cache uv run --offline --frozen pytest` — 84 passed on
  Python 3.10.20.
- `uv run --project ... --offline --frozen grit-waterbirds-run
  configs/waterbirds/smoke.yaml` — passed from `/tmp`, producing 20 canonical,
  non-reportable final-run results across the ten declared seeds for each method.
- Relative-link validation checked all nine Markdown files under `docs/` and `configs/`
  with no missing local target; `git diff --check` passed.
- No Waterbirds, CUB, Places, CLIP, or other real asset was downloaded. No reportable
  sweep, test-oracle diagnostic, W&B operation, or legacy implementation change ran.

### Milestone 5 artifact-boundary hardening

- Added a Waterbirds-specific adjusted-weight specification minted from a revalidated
  dataset manifest. It canonically binds the four-group order, training counts, dataset
  digest, and its own digest; metric constructors no longer accept caller-supplied group
  counts.
- Threaded dataset-manifest, feature-cache, normalization, and adjusted-weight lineage
  through validation metrics, checkpoint/candidate decisions, tuning finalists,
  confirmation, candidate/checkpoint freezes, final metrics, resolved configurations,
  smoke summaries, and run results. Every strict nested boundary rechecks its lineage.
- Made final reporting explicitly seed-addressed. Each run records its final seed, method
  summaries require the configured ten seed observations exactly once and in canonical
  order, and GRIT-minus-ERM differences are joined by matching seed identities.
- Strengthened construction provenance: retained majority endpoints now record the same
  deterministic selection position as their generated partner and relationship; replaced
  released IDs must be unique and absent from current records. Renamed the foreground
  digest to identify the canonical masked source-CUB foreground before compositing, without
  claiming released JPEG and generated PNG bytes are identical.
- Replaced generic Waterbirds result references with a small discriminated reference
  union. Successful ERM results require unique dataset, feature, and selected-checkpoint
  references; GRIT additionally requires matching pair and projection references. The
  parser rejects missing, duplicate, wrongly typed, or lineage-mismatched references.
- Added focused regression coverage for cross-dataset/cache/normalization/weight mixing,
  tuning-to-confirmation lineage changes, result weight mismatch, exact final seed sets,
  construction positions/replacements, and required artifact references. The smoke path
  remains hermetic and non-reportable; no Milestone 6 scope was introduced.

Verification:

- `uv lock --check` — passed (41 packages resolved).
- `uv run --frozen ruff check .` — passed.
- `uv run --frozen basedpyright` — 0 errors, warnings, or notes.
- `UV_CACHE_DIR=/tmp/grit-uv-cache uv run --offline --frozen pytest` — 95 passed on
  Python 3.10.20, including the deterministic Waterbirds smoke lifecycle.
- `git diff --check` — passed before the correction commit.
- No dependency, inherited implementation, dataset download, reportable sweep, W&B,
  estimated-pairing, or Milestone 6 change was made.

## Next proposed checkpoint

Review and approve Milestone 5, then decide the first bounded Milestone 6 pairing/search
task. Do not start estimated pairs, GroupDRO, production W&B, or the reportable scientific
sweep before that review.
