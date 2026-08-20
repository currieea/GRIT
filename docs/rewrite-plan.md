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
    -> define shared contracts
    -> extract pairing and projection
    -> complete CMNIST vertical slice
    -> complete Waterbirds vertical slice
    -> add search and aggregation
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

Status: **Proposal awaiting user review.** The documentation proposal exists, but the
contracts and synthetic fixtures in the exit criteria are not implemented. See
[`contracts.md`](contracts.md).

### Work

- Typed experiment configuration and validation.
- Dataset bundle and split interfaces.
- Pair records and pair-builder interface.
- Projection interface.
- Algorithm interface.
- Selection policy and checkpoint contract.
- Structured run result and provenance schema.
- Optional tracking interface.

### Exit criteria

- Contracts are covered using synthetic fixtures.
- Test metrics cannot flow into an ordinary selector through the public interface.
- Configuration and results round-trip through their serialized forms.

## Milestone 4: Extract pair construction and projection

### Work

- Implement oracle, conditional, and nearest pair builders as independent components.
- Implement classifier-independent linear nuisance projection.
- Add determinism, indexing, numerical, shape, and leakage tests.
- Document intentional corrections to inherited pair behavior.

### Exit criteria

- Pairing and projection run without constructing a trainer.
- Builders expose source indices and provenance.
- Pair construction is restricted to protocol-approved data.

## Milestone 5: ColoredMNIST vertical slice

### Work

- Dataset creation/loading and manifest.
- ERM and oracle GRIT.
- Validation selection and checkpoint restoration.
- Structured local output.
- Smoke, reproducibility, and leakage tests.

### Exit criteria

- One command runs a small ERM-versus-GRIT experiment end to end.
- Repeated runs with the same seed are reproducible within documented guarantees.
- Final test metrics come from a validation-selected configuration and checkpoint.

## Milestone 6: Waterbirds vertical slice

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

## Milestone 7: Pairing variants and experiment search

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

## Milestone 8: Port remaining algorithms

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

## Milestone 9: Compatibility and cutover

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
2. Shared public interfaces are proposed.
3. The ColoredMNIST vertical slice passes.
4. The Waterbirds vertical slice passes.
5. Before legacy code is removed or archived.

## Explicit non-goals during early milestones

- Reproducing every historical table row
- Porting every dataset at once
- Running large W&B sweeps before selection is verified
- Deleting the inherited implementation
- Designing abstractions solely for hypothetical future methods
