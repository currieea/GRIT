# GRIT full-scale rewrite plan

Status: **Approved**

## Objective

Build a rigorous, reproducible experiment system for GRIT, beginning with
ColoredMNIST and Waterbirds. The rewrite should make data access, invariant-pair
construction, projection, optimization, model selection, final evaluation, and result
provenance explicit and independently testable.

Matching historical paper values is useful diagnostic evidence, but it is not the
primary success criterion. The primary criteria are experimental validity,
reproducibility, maintainability, and transparent reporting.

## Scope

Initial scope:

- ColoredMNIST and Waterbirds
- ERM and linear GRIT/ECMP vertical slices
- Oracle, conditional, and nearest pair construction
- Validation-based model selection
- Local structured results with optional W&B tracking
- A configuration-driven search and seed-aggregation workflow

Later scope:

- GroupDRO, IRM, REx, MatchDG, Fish, LISA, and SWAD
- Additional inherited datasets
- Legacy command compatibility and eventual cutover

## Order of operations

Do not begin by rearranging every existing directory. Establish the behavioral and
scientific contracts first, scaffold the new path alongside the old path, and migrate
complete vertical slices before removing legacy code.

```text
Preserve baseline
    -> approve experiment protocols
    -> scaffold the new package
    -> implement the minimal shared-contract spine
    -> exercise and revise it in the CMNIST vertical slice
    -> complete Waterbirds vertical slice
    -> add pairing variants and production search/tracking
    -> port remaining methods
    -> compatibility review and cutover
```

## Milestone 0: Preserve and characterize the inherited baseline

Status: **Source preserved and statically characterized; runtime reproduction deferred**

### Work

- Record the inherited Git revision and relevant historical commands.
- Preserve the legacy implementation and audit branches.
- Add narrowly scoped characterization tests where practical.
- Record known ambiguities and known correctness concerns without silently correcting
  them in the legacy path.

### Exit criteria

- The inherited source remains available unchanged; runtime runnability is not claimed
  until its external data, services, and historical environment are available.
- Important behavioral changes can be distinguished from mechanical restructuring.
- Known legacy issues are documented.

## Milestone 1: Approve experiment protocols

### Work

- Complete `docs/experiments/cmnist.md`.
- Complete `docs/experiments/waterbirds.md`.
- Decide which information is allowed during training, pair construction, model
  selection, and final evaluation.
- Define how oracle results and diagnostic test envelopes are labeled.

### Exit criteria

- Every model decision maps to an explicitly permitted data split.
- Search metrics, seed policy, aggregation, and reporting are defined.
- Remaining unresolved decisions are blocking and clearly listed.

## Milestone 2: Scaffold the new package and tooling

Status: **Implemented and verified**

### Work

- Add `pyproject.toml`, `uv.lock`, and development tooling.
- Create `src/grit/`, `configs/`, and the new test structure.
- Keep `main.py` and legacy directories available as compatibility references.
- Establish lint, type-check, and test commands.

### Exit criteria

- The new package imports from a clean environment.
- A minimal test suite passes.
- No legacy algorithm behavior has been unintentionally changed.

## Milestone 3: Define shared contracts

Status: **Approved, implemented, and verified.** Detailed future guidance is retained in
[`contracts.md`](contracts.md), but only the contract spine below was a Milestone 3
deliverable.

### Work

- Add Pydantic v2 and implement strict boundary models only for CMNIST ERM/oracle-GRIT
  configuration and ordinary/test-oracle results.
- Implement split roles plus role-scoped training, validation, final-test, and diagnostic
  views.
- Implement distinct validation/final/diagnostic metric records and validation-only
  checkpoint/candidate selectors.
- Implement deterministic ties, separate candidate/checkpoint freezes, earlier-epoch
  checkpoint preference, and CMNIST dual-selector finalist-union handling.
- Implement minimal checkpoint identity and inference restoration with fake state.
- Implement canonical JSON round trips and a null event sink.
- Assemble one in-memory lifecycle test with a fake bounded algorithm update inside
  trainer-owned iteration, covering selection, restoration, and the one-way final-test
  gate.

### Exit criteria

- The focused Pydantic configuration/results round-trip through canonical JSON and reject
  unknown or inconsistent fields.
- Role types prevent final-test or diagnostic metrics from entering an ordinary selector.
- Validation deterministically selects a fake checkpoint, that exact state is restored,
  and final evaluation occurs only afterward without a feedback path.
- No production dataset, feature-cache, pair, projection, W&B, artifact-framework, or
  training-resume implementation is introduced.
- New type names and modules are explicitly internal and revisable.

## Milestone 4: ColoredMNIST vertical slice

Status: **Reviewed, implemented, and verified**

### Work

- Resolve the blocking deterministic source-partition decision before dataset work.
- Implement the smallest dataset/feature boundary required by the approved CMNIST
  protocol; do not generalize storage first.
- Implement the training-only oracle pair builder and classifier-independent projection
  as parts of this slice, including deterministic CPU-float64 SVD.
- ERM and oracle GRIT.
- Validation selection and checkpoint restoration.
- Canonical local results using the Milestone 3 spine.
- Smoke, reproducibility, and leakage tests.
- Revise internal types/modules when end-to-end evidence shows a simpler boundary.

### Exit criteria

- One command runs a small ERM-versus-GRIT experiment end to end.
- Repeated runs with the same seed are reproducible within documented guarantees.
- Final test metrics come from a validation-selected configuration and checkpoint.
- Pairing/projection remain independent of the trainer and use training sources only.
- No generalized artifact framework, faithful training-resume system, production W&B
  adapter, raw-image stack, or future-method hook layer was added without observed need.

## Milestone 5: Waterbirds vertical slice

Status: **Active**

### Work

- Dataset metadata, group definitions, manifests, and feature cache.
- ERM and oracle GRIT.
- Validation worst-group selection.
- Restored checkpoint followed by final test evaluation.
- Explicit separation of validation-selected and test-oracle diagnostic outputs.

### Exit criteria

- Split counts, group counts, and artifact hashes are recorded.
- Training, pair construction, and model selection pass leakage tests.
- Average and worst-group results are reproducibly reported across declared seeds.
- Shared names and boundaries are reviewed only after both CMNIST and Waterbirds have
  exercised them.

## Milestone 6: Pairing variants and experiment search

### Work

- Enable conditional and nearest pairs through configuration.
- Implement a local search runner with optional W&B tracking.
- Separate search, confirmation, and final evaluation stages.
- Aggregate identical configurations across seeds before selection.

### Exit criteria

- Local and W&B-backed runs apply identical selection semantics.
- Search never depends on final test metrics unless explicitly running a separately
  labeled test-oracle diagnostic.
- The selected configuration is reproducible from saved artifacts.

## Milestone 7: Port remaining algorithms

Recommended order:

1. GroupDRO
2. IRM and REx
3. MatchDG
4. Fish
5. LISA
6. SWAD

Each algorithm receives focused objective/update tests and small dataset smoke tests.
Adding an algorithm should not require modifications to dataset loading, selection, or
tracking semantics.

## Milestone 8: Compatibility and cutover

### Work

- Translate important legacy commands into new configurations.
- Add a compatibility shim or clear deprecation path for `main.py`.
- Compare legacy and new behavior under intentionally equivalent settings.
- Document deliberate differences.
- Remove or archive legacy directories only after approval.

### Exit criteria

- Every maintained experiment uses the new path.
- Documentation and commands refer to the new package.
- The full verification suite passes.
- A rollback point for the inherited implementation remains available in Git.

## Review gates

Require user review after:

1. Dataset protocols are drafted.
2. The minimal shared-contract spine is implemented and synthetically verified.
3. The ColoredMNIST vertical slice passes.
4. The Waterbirds vertical slice passes.
5. Before legacy code is removed or archived.

## Explicit non-goals during early milestones

- Reproducing every historical table row
- Porting every dataset at once
- Running large W&B sweeps before selection is verified
- Deleting the inherited implementation
- Designing abstractions solely for hypothetical future methods
- Treating pre-vertical-slice type names or module boundaries as stable public API
- Building generalized artifact, resume, tracking, or accelerator frameworks before a
  vertical slice demonstrates the need
