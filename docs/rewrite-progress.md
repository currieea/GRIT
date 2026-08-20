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

### Waterbirds construction and selection draft

- Chose the released Waterbirds-95 artifact and its official train/validation/test
  assignments as the canonical base dataset.
- Preserved the deliberately balanced official validation split for group-aware
  worst-group model selection.
- Defined training-only controlled land/water oracle pairs as a separate auxiliary
  resource rather than extra supervised classifier data.
- Added a separately labeled counterfactual-augmentation control.
- Excluded the inherited snow/desert categories from canonical Waterbirds and reserved
  any expanded-background study for a separate protocol.
- Defined final test worst-group accuracy, per-group results, and training-distribution
  adjusted average reporting.

Verification:

- Protocol language, split access, pair-bank access, and legacy-difference notes reviewed
  locally.
- Cross-checked the dataset rationale against the original GroupDRO documentation and
  Waterbirds generation script.
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
- Waterbirds uses the released Waterbirds-95 split assignments without regeneration or
  resplitting.
- Waterbirds selects ordinary configurations and checkpoints using official validation
  worst-group accuracy; test metrics are final-evaluation-only.
- Waterbirds oracle pairs use training birds only and do not become supervised examples
  except in a separately labeled counterfactual-augmentation control.
- Canonical Waterbirds groups are the four binary bird-label/land-water combinations;
  snow and desert are excluded.

## Unresolved decisions

- ColoredMNIST deterministic source-partition algorithm
- ColoredMNIST representation, projection, pair-budget, and search details
- Waterbirds artifact registration, exact pair generator, and pair-budget details
- Waterbirds encoder, feature-normalization, and expanded baseline choices
- Search spaces, budgets, and seed policy
- Supported Python/PyTorch versions
- Configuration/schema implementation
- Exact algorithm and dataset public interfaces

## Next proposed checkpoint

Resolve the shared representation, projection, search, seed, and uncertainty contracts
needed by the first ColoredMNIST and Waterbirds vertical slices, then review whether the
protocol milestone is sufficiently complete to begin the new package scaffold.
