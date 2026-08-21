# Rewrite documentation

This directory contains the design and experimental contracts for the GRIT rewrite. Core
scientific protocols were established before implementation so decisions are reviewed
rather than inferred from legacy code. The minimal Milestone 3 contract spine is approved,
implemented, and verified. Milestone 4 CMNIST is reviewed. The implemented and hermetically
verified Milestone 5 Waterbirds vertical slice now provides the second dataset test and is
awaiting user review; shared names remain revisable until that review is accepted.

## Documents

- [`rewrite-plan.md`](rewrite-plan.md): order of operations, milestones, and exit gates.
- [`architecture.md`](architecture.md): target package layout and component boundaries.
- [`contracts.md`](contracts.md): approved Milestone 3 core contracts, provisional design
  guidance, deferred extensions, leakage boundaries, and synthetic-test evidence.
- [`rewrite-progress.md`](rewrite-progress.md): checkpoint log for completed work and
  unresolved decisions.
- [`legacy-baseline.md`](legacy-baseline.md): exact inherited revision, representative
  commands, static behavior characterization, and deferred runtime checks.
- [`experiments/cmnist.md`](experiments/cmnist.md): approved ColoredMNIST core protocol,
  deterministic partition algorithm, and remaining estimated-pair decisions.
- [`experiments/waterbirds.md`](experiments/waterbirds.md): approved Waterbirds-CF core
  protocol, deterministic server-side reconstruction, and source-acquisition plan.

## Status vocabulary

- **Proposed:** documented but not yet approved.
- **Approved:** accepted as the implementation contract.
- **Implemented:** present in the new code path.
- **Verified:** covered by the required checks and supporting evidence.
- **Unresolved:** requires an explicit scientific or architectural decision.

The inherited implementation remains the historical reference. These documents describe
the intended rewrite unless a section explicitly discusses legacy behavior.
