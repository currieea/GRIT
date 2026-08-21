# Experiment configurations

Validated experiment settings live under the dataset-specific `cmnist/` and `waterbirds/`
directories. [`cmnist/smoke.yaml`](cmnist/smoke.yaml) is the implemented, explicitly
non-reportable hermetic Milestone 4 profile consumed by `grit-cmnist-run`. It uses injected
synthetic MNIST-like pools and a deterministic fake 512-dimensional encoder; it must not be
reported as a scientific result.

Real MNIST and official OpenAI CLIP preparation is performed explicitly with
`grit-cmnist-prepare`. Dataset and weight downloads occur only when `--allow-download` is
provided. [`cmnist/production-search.yaml`](cmnist/production-search.yaml) is the strict
Milestone 6A example for an already prepared cache. Its `__REQUIRED_*` paths are deliberate:
replace them with explicit real manifest/output paths, then use `grit-search plan` before
`run`. The schema fixes the 16 ERM and 400 oracle-GRIT candidates, two selectors, 3+2+10
seed stages, and unnormalized primary representation. An L2 sensitivity run must use a
separate configuration named `l2_normalized_sensitivity` and an L2 cache; changing only a
path or normalization field is rejected by lineage validation.

[`waterbirds/smoke.yaml`](waterbirds/smoke.yaml) is the implemented, explicitly
non-reportable and fully offline Milestone 5 profile consumed by `grit-waterbirds-run`.
It creates tiny synthetic Waterbirds/CUB/mask/Places assets under its output directory,
uses the real construction, capability, feature-cache, projection, training, selection,
restoration, and final-evaluation paths, and uses only the deterministic fake encoder.

`grit-waterbirds-prepare` is the separate server boundary for already acquired released
Waterbirds, CUB, mask, and approved Places assets. It never downloads those datasets.
Pinned official CLIP weights must already exist unless the operator explicitly passes
`--allow-clip-download`. [`waterbirds/production-search.yaml`](waterbirds/production-search.yaml)
is the corresponding strict 16-ERM/400-oracle-GRIT local-search example. It accepts only a
reportable 4,795/1,199/5,794 Waterbirds-CF manifest, exact 240-pair oracle manifest, pinned
official CLIP cache, matching normalization, and matching adjusted-weight lineage.

The production examples describe executable local orchestration, not completed scientific
results. The repository does not ship datasets and no full grid has been launched. Local
canonical files remain authoritative; W&B mirroring is deferred. Conditional/nearest
pairing and additional algorithms do not resolve through these schemas.
