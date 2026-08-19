# Rewrite progress

This is the compact checkpoint log for the GRIT rewrite. Update it at milestone
boundaries and after material decisions; do not use it as a raw command transcript.

## Current state

- Branch: `rewrite`
- Active milestone: Documentation and protocol scaffolding
- Legacy implementation: Preserved
- New implementation: Not started

## Completed checkpoints

### Documentation skeleton

- Added the staged rewrite plan.
- Added the target architecture proposal.
- Added ColoredMNIST and Waterbirds protocol skeletons.
- Added rewrite-oriented agent guidance.

Verification:

- Documentation structure and internal references reviewed locally.
- No implementation or experiment behavior changed.

## Approved decisions

- The rewrite will prioritize rigorous experiment semantics over matching historical table
  values.
- The inherited implementation remains available during vertical-slice development.
- Dataset-specific scientific details remain unresolved until their protocols are
  reviewed.

## Unresolved decisions

- ColoredMNIST dataset and validation protocol
- Waterbirds counterfactual-training policy
- Waterbirds model-selection metric and tie-breakers
- Search spaces, budgets, and seed policy
- Supported Python/PyTorch versions
- Configuration/schema implementation
- Exact algorithm and dataset public interfaces

## Next proposed checkpoint

Review and approve `docs/rewrite-plan.md` and `docs/architecture.md`, then begin filling
the experiment protocol documents before creating the new package scaffold.
