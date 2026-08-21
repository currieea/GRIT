# Real-server execution for Milestone 6A

Status: **Operational procedure implemented and hermetically tested; no real pilot or
reportable experiment has been run by the rewrite work.**

This runbook wraps the existing local production search. It does not change the approved
candidate grid, seeds, selectors, final-test gate, result schemas, or reportability rules.
The recommended pilot runs two canonical tuning tasks from the full plan and leaves their
published results available to the later unrestricted search.

## 1. Install and preserve exact code provenance

Use the repository's pinned Python 3.10.20 interpreter and locked environment:

```bash
uv python install 3.10.20
uv sync --frozen
uv run --frozen python --version
git status --short
git rev-parse HEAD
```

`grit-search plan` and `grit-search run` fail before writing or training unless `HEAD`
resolves to a real commit and the worktree is clean. Keep generated search output outside
the repository, or in a repository-local directory already covered by `.gitignore`.
`grit-search status` is read-only and may be used from a dirty worktree after a compatible
plan exists.

Use explicit absolute paths below. Tokens such as `/ABSOLUTE/PATH/...` are placeholders;
they do not assert that an asset exists on a server.

## 2. Prepare the real artifacts

CMNIST preparation needs a writable MNIST data root, an explicit CLIP weight-cache root,
and a new or empty preparation output root:

```bash
uv run --frozen grit-cmnist-prepare \
  --data-root /ABSOLUTE/PATH/TO/MNIST \
  --clip-weights-root /ABSOLUTE/PATH/TO/CLIP-WEIGHTS \
  --output-root /ABSOLUTE/PATH/TO/PREPARED/CMNIST \
  --construction-seed 0 \
  --pair-seed 0 \
  --normalization none
```

Without `--allow-download`, both MNIST and the pinned official OpenAI CLIP weights must
already be present. Adding that flag explicitly permits those two downloads. Successful
preparation writes `dataset-manifest.json`, `pair-manifest.json`, and
`feature-cache/manifest.json` alongside the cache arrays.

Waterbirds-CF preparation requires already acquired released Waterbirds metadata/images,
source CUB images, segmentation masks, and the approved Places background assets:

```bash
uv run --frozen grit-waterbirds-prepare \
  --released-root /ABSOLUTE/PATH/TO/RELEASED-WATERBIRDS \
  --cub-root /ABSOLUTE/PATH/TO/CUB \
  --masks-root /ABSOLUTE/PATH/TO/MASKS \
  --places-root /ABSOLUTE/PATH/TO/PLACES \
  --clip-weights-root /ABSOLUTE/PATH/TO/CLIP-WEIGHTS \
  --output-root /ABSOLUTE/PATH/TO/PREPARED/WATERBIRDS \
  --construction-seed 0 \
  --normalization none
```

That command never downloads Waterbirds, CUB, masks, or Places. The pinned CLIP weights
must also be installed unless `--allow-clip-download` is explicitly supplied. Successful
preparation writes `construction/dataset-manifest.json`, `pair-manifest.json`, and
`feature-cache/manifest.json` plus the constructed images and cache array.

The preparation and search output roots must be disjoint. The planner rejects an output
root that is a filesystem/repository root, equals or nests inside a prepared-artifact tree,
contains an input artifact, or is a nonempty unrelated directory.

## 3. Create a server-specific production configuration

Copy the relevant checked-in example to a server-owned location; do not edit the
placeholder example in the repository:

```bash
cp configs/cmnist/production-search.yaml /ABSOLUTE/PATH/TO/cmnist-search.yaml
cp configs/waterbirds/production-search.yaml /ABSOLUTE/PATH/TO/waterbirds-search.yaml
```

Use only the file for the dataset being run. Replace every `__REQUIRED_*` value with the
absolute prepared manifest or dedicated search-output path. Keep the checked-in scientific
grid, stage seeds, normalization variant, and other protocol fields unchanged. Use a
separate preparation cache, configuration, and output root for the L2 sensitivity.

Planning validates the dataset, pair, and feature manifests; production counts; official
CLIP identity; normalization; cross-artifact lineage; and referenced file hashes. It writes
the authored configuration copy, resolved configuration, and full 416-candidate plan:

```bash
uv run --frozen grit-search plan /ABSOLUTE/PATH/TO/cmnist-search.yaml
```

`plan` does not deserialize feature arrays, train, make checkpoints, or issue final-test
access. It does read referenced files as bytes to verify their declared hashes.

## 4. Identify and run the two-task pilot

Inspect the existing plan with the read-only presentation command:

```bash
uv run --frozen grit-search pilot-candidates /ABSOLUTE/PATH/TO/cmnist-search.yaml
```

The canonical JSON reports the first configured tuning seed, one deterministic ERM
candidate, and one deterministic GRIT candidate whose projection rank is nonzero. The
candidate records come directly from the saved full plan; the command neither replans nor
writes. Copy the reported `tuning_seed`, `erm.candidate_id`, and
`grit_nonzero_rank.candidate_id` values into the following command:

```bash
uv run --frozen grit-search run /ABSOLUTE/PATH/TO/cmnist-search.yaml \
  --stop-after tuning \
  --candidate-id 'CANDIDATE_ID_FROM_ERM_FIELD' \
  --candidate-id 'CANDIDATE_ID_FROM_GRIT_NONZERO_RANK_FIELD' \
  --tuning-seed TUNING_SEED_FROM_OUTPUT \
  --max-new-runs 2
```

Use the same sequence for Waterbirds with its server-specific YAML. This bounded command
executes exactly those two existing plan tasks: one ERM and one nonzero-rank oracle-GRIT
task at the same seed. The GRIT task fits and records the real oracle projection. Both tasks
exercise training, validation records, checkpoint selection, atomic run publication, and
status reporting.

The tuning-only loader materializes training, validation, and pair rows but exposes no
final-test handle or table. Waterbirds uses one combined array file, so integrity checking
may scan that file, but final rows are not materialized as experiment data. No final-test
capability can be issued in this path. The pilot is incomplete and non-reportable; its
metrics are operational evidence, not scientific results.

The JSON printed by bounded `run` is canonical progress. Confirm it independently:

```bash
uv run --frozen grit-search status /ABSOLUTE/PATH/TO/cmnist-search.yaml
```

Repeating the exact pilot command validates and reuses both completed tasks without new
training. A positive `--max-new-runs N` counts only missing tasks and stops between atomic
runs. Method or candidate filtering and `--tuning-seed` are restricted to
`--stop-after tuning`, so they cannot bypass confirmation or final selection.

## 5. Continue the authoritative search

After reviewing pilot timing, memory, validation records, projection diagnostics, and
status, remove all operational limits:

```bash
uv run --frozen grit-search run /ABSOLUTE/PATH/TO/cmnist-search.yaml
```

An unrestricted invocation retains the original full-search behavior. It reuses the two
pilot results byte-for-byte, completes missing tuning runs, derives the canonical
selector-specific finalists, runs confirmation, freezes winners, runs all final seeds, and
only then writes production and paired summaries plus the experiment index.

Interruption is safe only at run granularity. A completed task directory is reused after
strict validation. An interrupted staging directory is preserved/archived and that whole
task restarts; optimizer or partial-epoch resume is not promised. `tmux`, `screen`, or a
site scheduler can keep the local command alive, but this repository provides no
scheduler-specific integration and no GPU path. Decide whether either is needed only after
observing real pilot timings.

## 6. Preserve the authoritative outputs

Preserve or copy the complete dedicated output tree, including:

- `authored-config.yaml`, `resolved-config.json`, and `search-plan.json`;
- canonical per-task `result.json` files and final `final-result.json` files;
- selected-checkpoint manifests and parameter arrays;
- projection diagnostics;
- tuning finalist, confirmation-union (CMNIST), and frozen-winner artifacts;
- production and paired summaries; and
- `experiment-index.json`.

Copying only dashboard output or terminal logs is insufficient. Canonical local files are
authoritative; W&B mirroring is not implemented. Run `grit-search status` before transfer.
After transfer, it can validate the copy only when the authored configuration and absolute
output/input locations still match the saved plan; otherwise preserve the tree as an
archive rather than rewriting its provenance.

No real assets were available and no real pilot, full grid, final result, timing claim, or
scientific performance claim was produced while implementing this runbook and its bounded
execution controls.
