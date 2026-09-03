# Working in this repository

GRIT (called ECMP in the inherited code) projects frozen CLIP features onto the null
space of nuisance directions estimated from counterfactual pairs. The new implementation
is `src/grit/`; the top-level `main.py`, `datasets/`, `models/`, `solver/`,
`experiments/`, and `scripts/` are the inherited code, kept only as a reference.

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

- One command, `grit` (`src/grit/cli/main.py`): `prepare`, `run`, `status`, `smoke`.
- Configs are YAML under `configs/`. Paths may use `${PROJECT_SCRATCH}`; that variable is
  exported by `scratch-project` from the user's dotfiles on the ECN servers. Large data,
  prepared artifacts, and outputs go under `$PROJECT_SCRATCH`, never the NFS home.
- `uv` with `--extra cu128` on GPU machines and `--extra cpu` elsewhere. One
  environment per machine is enough; CUDA torch runs the CPU training path fine.
- Checks: `uv run ruff check .`, `uv run basedpyright`, `uv run pytest`. Add a test
  when a change affects selection, leakage, or data construction. Do not add tests for
  serialization round-trips or hash bookkeeping.

## Things not to do

- Do not refuse to run because the worktree is dirty; record it and continue.
- Do not add new entry points, milestone documents, or progress logs. Summarize
  decisions in the commit message or the relevant protocol doc.
- Do not spawn subagents for routine edits.
- Do not touch the inherited top-level code unless porting a baseline from it.

`docs/history/` holds the original rewrite plan, contracts, and progress log. They are
context, not instructions.
