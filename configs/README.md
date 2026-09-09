# Experiment configurations

- `cmnist/production-search.yaml`, `rotated_mnist/production-search.yaml`, and
  `waterbirds/production-search.yaml`: the primary unnormalized studies. Paths use
  `${PROJECT_SCRATCH}` so they run unmodified after
  `scratch-project`. Copy and edit for other machines or for the L2 sensitivity (which
  needs `--normalization l2` at prepare time, a separate artifact root, and
  `experiment_variant: l2_normalized_sensitivity`).
- `cmnist/{groupdro,rex,irm,fish,lisa,swad,matchdg}-search.yaml`: the CMNIST
  baselines, each in its own output tree so adding one never reruns another.
- `cmnist/*-test-oracle.yaml`: the same grids and seeds selected on `test_ood`
  (`selectors: [test_oracle]`), the paper's "oracle validation" columns. Each writes a
  separate `*-test-oracle` output tree; every result and summary in it is labeled
  `test_oracle` and must be reported under that heading, never beside the ordinary
  numbers as if validation-selected.
- `cmnist/smoke.yaml` and `waterbirds/smoke.yaml`: hermetic end-to-end checks with a
  fake encoder, run by `scripts/smoke.py`. Never report their numbers.

`reportable: true` in a result means real data and the pinned CLIP encoder were used, as
opposed to the fake-encoder smoke path. It does not mean the run matched the paper grid;
judge that from the plan file next to the results.

The checked-in grids, seeds, epochs, and pair counts are the primary protocol from
`docs/experiments/`. Any of them can be changed in a copied config for a sensitivity or a
quick check; the plan and every result record the values that actually ran. A config's
`pair_count` may be any prefix of the prepared pair bank (pairs are stored in seeded
order), so bank 512 pairs once with `--pair-count 512` and run 32/64/128/256/512 from
five configs.
