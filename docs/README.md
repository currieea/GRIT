# Rewrite documentation

This directory contains the design and experimental contracts for the GRIT rewrite.
The documentation is intentionally established before the new implementation so that
scientific decisions are reviewed rather than inferred from legacy code.

## Documents

- [`rewrite-plan.md`](rewrite-plan.md): order of operations, milestones, and exit gates.
- [`architecture.md`](architecture.md): target package layout and component boundaries.
- [`rewrite-progress.md`](rewrite-progress.md): checkpoint log for completed work and
  unresolved decisions.
- [`experiments/cmnist.md`](experiments/cmnist.md): ColoredMNIST protocol skeleton.
- [`experiments/waterbirds.md`](experiments/waterbirds.md): Waterbirds protocol skeleton.

## Status vocabulary

- **Proposed:** documented but not yet approved.
- **Approved:** accepted as the implementation contract.
- **Implemented:** present in the new code path.
- **Verified:** covered by the required checks and supporting evidence.
- **Unresolved:** requires an explicit scientific or architectural decision.

The inherited implementation remains the historical reference. These documents describe
the intended rewrite unless a section explicitly discusses legacy behavior.
