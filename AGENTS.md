# Working in this repository

GRIT (called ECMP in the inherited code) projects frozen CLIP features onto the null
space of nuisance directions estimated from counterfactual pairs. The new implementation
is `src/grit/`; everything under `legacy/` is the inherited code (its `experiments/`
W&B launchers show what the paper originally ran), kept only as a reference.

This is a research codebase. Prefer running experiments over adding infrastructure.
Before adding a schema, a manifest field, a CLI flag, or a document, ask whether it
changes a number in the paper or prevents a real leak. If not, leave it out.

## Scientific rules that must hold

- Hyperparameter, checkpoint, and rank selection use validation metrics only. Test
  metrics are computed once, after selection is frozen.
- Oracle pairs come from training sources only, never validation or test.
- Every reported result records its seed, config, commit, and input artifact hashes.
- Protocol details live in `docs/experiments/cmnist.md` and
  `docs/experiments/waterbirds.md`. Change the protocol there first, then the code.

## Layout and conventions

- Entry points are plain scripts in `scripts/` (`prepare_cmnist.py`, `run_search.py`,
  `search_status.py`, `smoke.py`). Each is a short argparse wrapper over `src/grit/`.
- `src/grit/` is organized by concern: `data/` (dataset construction and pair banks),
  `features/` (CLIP caches), `methods/` (projection, linear probe, checkpoints),
  `selection/` (validation-only selectors), `search/` (planner, scheduler, runners,
  summaries). Top-level modules hold shared schemas, config, results, and paths. The
  CMNIST and Waterbirds files inside each folder still duplicate each other; merge them
  rather than adding a third copy when porting a new dataset or method.
- Configs are YAML under `configs/`. Paths may use `${PROJECT_SCRATCH}`; that variable is
  exported by `scratch-project` from the user's dotfiles on the ECN servers. Large data,
  prepared artifacts, and outputs go under `$PROJECT_SCRATCH`, never the NFS home.
- `uv sync` installs the default PyTorch build; there are no extras to pass.
- Checks: `uv run ruff check .`, `uv run basedpyright`, `uv run pytest`. Add a test
  when a change affects selection, leakage, or data construction. Do not add tests for
  serialization round-trips or hash bookkeeping.

## Things not to do

- Do not refuse to run because the worktree is dirty; record it and continue.
- Do not add console-script entry points, milestone documents, or progress logs. Summarize
  decisions in the commit message or the relevant protocol doc.
- Do not spawn subagents for routine edits.
- Do not touch the inherited code under `legacy/` unless porting a baseline from it.

`docs/history/` holds the original rewrite plan, contracts, and progress log. They are
context, not instructions.
