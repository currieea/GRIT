# Rewrite progress

This is the compact checkpoint log for the GRIT rewrite. Update it at milestone
boundaries and after material decisions; do not use it as a raw command transcript.

## Current state

- Branch: `rewrite`
- Active milestone: Experiment protocol design
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

## Approved decisions

- The rewrite will prioritize rigorous experiment semantics over matching historical table
  values.
- The inherited implementation remains available during vertical-slice development.
- ColoredMNIST uses all official MNIST test sources only for final `0.9` OOD evaluation.
- ColoredMNIST validation reuses 10,000 held-out source images across `0.1`, `0.2`, and
  `0.5` color renderings; these are repeated views, not independent samples.
- ColoredMNIST reports a primary robustness-aware selector and a secondary source-only
  selector, both fixed before test access.

## Unresolved decisions

- ColoredMNIST deterministic source-partition algorithm
- ColoredMNIST representation, projection, pair-budget, and search details
- Waterbirds counterfactual-training policy
- Waterbirds model-selection metric and tie-breakers
- Search spaces, budgets, and seed policy
- Supported Python/PyTorch versions
- Configuration/schema implementation
- Exact algorithm and dataset public interfaces

## Next proposed checkpoint

Review the ColoredMNIST protocol's remaining decisions, then draft and approve the
Waterbirds protocol before creating the new package scaffold.
