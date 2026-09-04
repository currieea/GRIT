"""Dataset-neutral tuning, confirmation, freeze, final, and summary lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

from grit.schemas import SeedStage, StrictBoundaryModel
from grit.search.plan import SearchPlan
from grit.search.run import (
    ProductionExecutionLimits,
    ProductionSearchStatus,
    complete_outputs_valid,
    limited_tuning_candidates,
    write_experiment_index,
)
from grit.search.scheduler import (
    CompletedStageRun,
    LocalRunScheduler,
    SearchRunTask,
    StageExecutor,
    make_search_task,
)

StageRunT = TypeVar("StageRunT", bound=StrictBoundaryModel)
FinalistsT = TypeVar("FinalistsT")
WinnersT = TypeVar("WinnersT")
SummaryT = TypeVar("SummaryT")


@dataclass(frozen=True, slots=True)
class ProductionLifecycleHooks(Generic[StageRunT, FinalistsT, WinnersT, SummaryT]):
    """Dataset semantics plugged into the shared production phase driver."""

    coerce_runs: Callable[[tuple[CompletedStageRun, ...]], tuple[StageRunT, ...]]
    execute_pre_final: StageExecutor
    make_finalists: Callable[[tuple[StageRunT, ...], Path], FinalistsT]
    confirmation_tasks: Callable[[FinalistsT], tuple[SearchRunTask, ...]]
    make_winners: Callable[
        [FinalistsT, tuple[StageRunT, ...], Path], WinnersT
    ]
    final_tasks: Callable[[WinnersT], tuple[SearchRunTask, ...]]
    final_executor: Callable[[WinnersT], StageExecutor]
    make_summary: Callable[
        [FinalistsT, WinnersT, tuple[StageRunT, ...]], SummaryT
    ]
    persist_summary: Callable[[SummaryT, Path], None]
    status: Callable[[], ProductionSearchStatus]


@dataclass(frozen=True, slots=True)
class ProductionStatusHooks(Generic[StageRunT, FinalistsT, WinnersT]):
    """Dataset selection artifacts plugged into shared read-only status checks."""

    coerce_runs: Callable[[tuple[CompletedStageRun, ...]], tuple[StageRunT, ...]]
    compute_finalists: Callable[[tuple[StageRunT, ...]], FinalistsT]
    finalists_complete: Callable[[FinalistsT, Path], bool]
    confirmation_tasks: Callable[[FinalistsT], tuple[SearchRunTask, ...]]
    compute_winners: Callable[
        [FinalistsT, tuple[StageRunT, ...]], WinnersT
    ]
    winner_count: Callable[[WinnersT, Path], int]
    expected_winner_count: int
    final_tasks: Callable[[WinnersT], tuple[SearchRunTask, ...]]


def run_production_lifecycle(
    plan: SearchPlan,
    limits: ProductionExecutionLimits | None,
    hooks: ProductionLifecycleHooks[StageRunT, FinalistsT, WinnersT, SummaryT],
) -> SummaryT | ProductionSearchStatus:
    """Run or resume every production phase without encoding dataset semantics."""

    output_root = Path(plan.resolved_config.output_root)
    scheduler = LocalRunScheduler(output_root, plan)
    remaining = None if limits is None else limits.max_new_runs

    def run_stage(
        tasks: tuple[SearchRunTask, ...], execute: StageExecutor
    ) -> tuple[CompletedStageRun, ...]:
        nonlocal remaining
        if limits is None:
            return scheduler.run_tasks(tasks, execute)
        allowance = len(tasks) if remaining is None else remaining
        results, newly_executed = scheduler.run_tasks_bounded(
            tasks,
            execute,
            max_new_runs=allowance,
        )
        if remaining is not None:
            remaining -= newly_executed
        return results

    tuning_candidates = limited_tuning_candidates(plan, limits)
    tuning_seeds = (
        plan.seeds.stages.tuning
        if limits is None or limits.tuning_seed is None
        else (limits.tuning_seed,)
    )
    tuning_tasks = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in tuning_candidates
        for seed in tuning_seeds
    )
    tuning_runs = hooks.coerce_runs(
        run_stage(tuning_tasks, hooks.execute_pre_final)
    )
    if limits is not None and (
        limits.stop_after == "tuning" or len(tuning_runs) != len(tuning_tasks)
    ):
        return hooks.status()

    finalists = hooks.make_finalists(tuning_runs, output_root)
    confirmation_tasks = hooks.confirmation_tasks(finalists)
    confirmation_runs = hooks.coerce_runs(
        run_stage(confirmation_tasks, hooks.execute_pre_final)
    )
    if limits is not None and len(confirmation_runs) != len(confirmation_tasks):
        return hooks.status()

    winners = hooks.make_winners(finalists, confirmation_runs, output_root)
    final_tasks = hooks.final_tasks(winners)
    final_runs = hooks.coerce_runs(
        run_stage(final_tasks, hooks.final_executor(winners))
    )
    if limits is not None:
        return hooks.status()

    summary = hooks.make_summary(finalists, winners, final_runs)
    hooks.persist_summary(summary, output_root)
    write_experiment_index(plan, output_root)
    return summary


def production_lifecycle_status(
    plan: SearchPlan,
    hooks: ProductionStatusHooks[StageRunT, FinalistsT, WinnersT],
) -> ProductionSearchStatus:
    """Reconstruct the canonical production phase without opening final test data."""

    root = Path(plan.resolved_config.output_root)
    scheduler = LocalRunScheduler(root, plan)
    tuning = tuple(
        make_search_task(plan, candidate, SeedStage.TUNING, seed)
        for candidate in plan.candidates
        for seed in plan.seeds.stages.tuning
    )
    tuning_complete = len(scheduler.status(tuning).complete_task_ids)
    if tuning_complete != len(tuning):
        return _status(
            plan,
            phase="tuning",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
        )

    tuning_runs = hooks.coerce_runs(scheduler.completed_results(tuning))
    finalists = hooks.compute_finalists(tuning_runs)
    if not hooks.finalists_complete(finalists, root):
        return _status(
            plan,
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
        )

    confirmation = hooks.confirmation_tasks(finalists)
    confirmation_complete = len(
        scheduler.status(confirmation).complete_task_ids
    )
    if confirmation_complete != len(confirmation):
        return _status(
            plan,
            phase="confirmation",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
        )

    confirmation_runs = hooks.coerce_runs(
        scheduler.completed_results(confirmation)
    )
    winners = hooks.compute_winners(finalists, confirmation_runs)
    winner_count = hooks.winner_count(winners, root)
    if winner_count != hooks.expected_winner_count:
        return _status(
            plan,
            phase="final",
            tuning_expected=len(tuning),
            tuning_complete=tuning_complete,
            confirmation_expected=len(confirmation),
            confirmation_complete=confirmation_complete,
            frozen_winner_count=winner_count,
        )

    final = hooks.final_tasks(winners)
    final_complete = len(scheduler.status(final).complete_task_ids)
    phase = (
        "complete"
        if final_complete == len(final) and complete_outputs_valid(plan)
        else "final"
    )
    return _status(
        plan,
        phase=phase,
        tuning_expected=len(tuning),
        tuning_complete=tuning_complete,
        confirmation_expected=len(confirmation),
        confirmation_complete=confirmation_complete,
        frozen_winner_count=winner_count,
        final_expected=len(final),
        final_complete=final_complete,
    )


def _status(
    plan: SearchPlan,
    *,
    phase: Literal["tuning", "confirmation", "final", "complete"],
    tuning_expected: int,
    tuning_complete: int,
    confirmation_expected: int = 0,
    confirmation_complete: int = 0,
    frozen_winner_count: int = 0,
    final_expected: int = 0,
    final_complete: int = 0,
) -> ProductionSearchStatus:
    return ProductionSearchStatus(
        schema_version="grit.production-search-status/v1",
        dataset=plan.dataset,
        plan_digest=plan.canonical_digest(),
        phase=phase,
        tuning_expected=tuning_expected,
        tuning_complete=tuning_complete,
        confirmation_expected=confirmation_expected,
        confirmation_complete=confirmation_complete,
        frozen_winner_count=frozen_winner_count,
        final_expected=final_expected,
        final_complete=final_complete,
    )
