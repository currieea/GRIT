# AGENTS.md

This branch contains a staged rewrite of the GRIT research codebase. The checked-in
top-level `datasets/`, `models/`, `solver/`, `experiments/`, and `main.py` files are the
inherited implementation. Preserve them as a behavioral and historical reference until
the rewrite plan explicitly reaches the compatibility and cutover phase.

## Read before making changes

Read these documents in order:

1. `docs/rewrite-plan.md`
2. `docs/architecture.md`
3. The protocol document for the dataset being changed:
   - `docs/experiments/cmnist.md`
   - `docs/experiments/waterbirds.md`
4. `docs/rewrite-progress.md`

If an implementation request conflicts with an unresolved scientific decision in a
protocol document, stop and ask for a decision instead of silently choosing a protocol.

## Rewrite principles

- Correct experimental semantics and reproducibility take priority over reproducing a
  historical table value.
- Keep oracle information, model selection, and final test evaluation distinct.
- Ordinary model and hyperparameter selection must not use test metrics.
- Keep pair construction and nuisance projection independent of classifiers and training
  loops.
- Keep reusable code in the future `src/grit/` package. Keep `scripts/` thin and put
  experiment settings in `configs/`.
- Prefer composition over the inherited pattern in which every method inherits dataset,
  training, evaluation, selection, and logging behavior from `ERM`.
- Use explicit registries and typed configuration instead of dynamic `eval(...)` lookup.
- A run must be attributable to a resolved configuration, seed, dataset manifest, code
  revision, and dependency environment.

## Migration safety

- Do not delete or broadly rewrite the legacy path until the new vertical slices have
  passed their documented exit criteria.
- Separate mechanical moves from behavioral changes.
- Label intentional corrections to inherited behavior in tests and documentation.
- Do not treat legacy numerical parity as a requirement when the old path leaked test
  information or contained a known correctness bug.
- Preserve unrelated and untracked user files.

## Verification

Every core change must include proportionate tests. The rewrite will standardize on
commands of this form once its tooling milestone is complete:

```bash
uv run ruff check .
uv run basedpyright
uv run pytest
```

Until that tooling exists on this branch, record the commands that were actually
available and run in `docs/rewrite-progress.md`; do not claim unavailable checks passed.

## Working with Codex goals and subagents

- Work on one milestone-sized goal at a time.
- The primary agent owns shared interfaces and integration.
- Use subagents for bounded read-only audits, test-gap reviews, or independent work whose
  interfaces are already stable.
- Do not have multiple agents edit shared architecture files concurrently in one
  worktree.
- End each milestone with a concise progress entry containing changes, verification,
  unresolved decisions, and the next safe step.
