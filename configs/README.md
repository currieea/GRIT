# Experiment configurations

- `cmnist/production-search.yaml` and `waterbirds/production-search.yaml`: the primary
  unnormalized studies. Paths use `${PROJECT_SCRATCH}` so they run unmodified after
  `scratch-project`. Copy and edit for other machines or for the L2 sensitivity (which
  needs `--normalization l2` at prepare time, a separate artifact root, and
  `experiment_variant: l2_normalized_sensitivity`).
- `cmnist/smoke.yaml` and `waterbirds/smoke.yaml`: hermetic end-to-end checks with a
  fake encoder, run by `scripts/smoke.py`. Never report their numbers.

The scientific fields (grid, seeds, selectors, epochs) are fixed by the protocol docs
and validated on load; only paths and names are meant to change between machines.
