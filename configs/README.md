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
