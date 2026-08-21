# Experiment configurations

Validated experiment settings live under the dataset-specific `cmnist/` and `waterbirds/`
directories. [`cmnist/smoke.yaml`](cmnist/smoke.yaml) is the implemented, explicitly
non-reportable hermetic Milestone 4 profile consumed by `grit-cmnist-run`. It uses injected
synthetic MNIST-like pools and a deterministic fake 512-dimensional encoder; it must not be
reported as a scientific result.

Real MNIST and official OpenAI CLIP preparation is performed explicitly with
`grit-cmnist-prepare`. Dataset and weight downloads occur only when `--allow-download` is
provided. A reportable sweep configuration remains deferred until the full scientific
search is executed. Other YAML fragments in the experiment protocols remain illustrative
scientific contracts, not files to copy here without validation.

[`waterbirds/smoke.yaml`](waterbirds/smoke.yaml) is the implemented, explicitly
non-reportable and fully offline Milestone 5 profile consumed by `grit-waterbirds-run`.
It creates tiny synthetic Waterbirds/CUB/mask/Places assets under its output directory,
uses the real construction, capability, feature-cache, projection, training, selection,
restoration, and final-evaluation paths, and uses only the deterministic fake encoder.

`grit-waterbirds-prepare` is the separate server boundary for already acquired released
Waterbirds, CUB, mask, and approved Places assets. It never downloads those datasets.
Pinned official CLIP weights must already exist unless the operator explicitly passes
`--allow-clip-download`. A reportable Waterbirds search configuration remains deferred.
