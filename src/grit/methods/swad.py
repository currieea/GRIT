"""SWAD loss-valley weight averaging, ported from the DomainBed reference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from grit.methods.training_state import LinearProbeState


class RunningAverage:
    """DomainBed's ``AveragedModel``: a plain mean that the first update overwrites."""

    def __init__(self, initial: LinearProbeState) -> None:
        self._weight = initial.weight.detach().clone().to(torch.float32)
        self._bias = initial.bias.detach().clone().to(torch.float32)
        self.count = 0

    def add(self, state: LinearProbeState) -> None:
        weight = state.weight.detach().to(torch.float32)
        bias = state.bias.detach().to(torch.float32)
        if self.count == 0:
            self._weight = weight.clone()
            self._bias = bias.clone()
        else:
            self._weight += (weight - self._weight) / (self.count + 1)
            self._bias += (bias - self._bias) / (self.count + 1)
        self.count += 1

    def copy(self) -> RunningAverage:
        duplicate = RunningAverage(self.state)
        duplicate.count = self.count
        return duplicate

    @property
    def state(self) -> LinearProbeState:
        return LinearProbeState(weight=self._weight.clone(), bias=self._bias.clone())


@dataclass(frozen=True, slots=True)
class SwadSegment:
    """One evaluation segment: its averaged parameters and the live model's loss."""

    state: LinearProbeState
    start_update: int
    end_update: int
    end_loss: float


class LossValley:
    """Reference SWAD start/end detection over segment losses.

    Ported from ``domainbed/swad.py`` (``LossValley``) with its queue semantics kept
    intact, including the ``AveragedModel`` copy-then-overwrite behavior.
    """

    def __init__(
        self, *, n_converge: int, n_tolerance: int, tolerance_ratio: float
    ) -> None:
        if n_converge <= 0 or n_tolerance <= 0 or tolerance_ratio <= 0.0:
            raise ValueError("SWAD valley parameters must be positive")
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self._converge_queue: deque[SwadSegment] = deque(maxlen=n_converge)
        self._smooth_queue: deque[SwadSegment] = deque(maxlen=n_tolerance)
        self.final: RunningAverage | None = None
        self.converge_update: int | None = None
        self.dead = False
        self.threshold: float | None = None

    @property
    def converged(self) -> bool:
        return self.converge_update is not None

    def _smooth_loss(self, index: int) -> float:
        return min(segment.end_loss for segment in list(self._smooth_queue)[index:])

    def observe(self, segment: SwadSegment) -> None:
        if self.dead:
            return
        self._converge_queue.append(segment)
        self._smooth_queue.append(segment)
        if not self.converged:
            if len(self._converge_queue) < self.n_converge:
                return
            losses = [item.end_loss for item in self._converge_queue]
            min_index = min(range(len(losses)), key=losses.__getitem__)
            if min_index != 0:
                return
            until_min = self._converge_queue[0]
            self.converge_update = until_min.end_update
            self.final = RunningAverage(until_min.state)
            self.threshold = (sum(losses) / len(losses)) * (1.0 + self.tolerance_ratio)
            if self.n_tolerance < self.n_converge:
                for offset in range(self.n_converge - self.n_tolerance):
                    self.final.add(self._converge_queue[1 + offset].state)
            elif self.n_tolerance > self.n_converge:
                converge_index = self.n_tolerance - self.n_converge
                window = list(self._smooth_queue)[: converge_index + 1]
                start_index = 0
                for index in reversed(range(len(window))):
                    if window[index].end_loss > self.threshold:
                        start_index = index + 1
                        break
                for item in window[start_index + 1 :]:
                    self.final.add(item.state)
            return
        converge_update = self.converge_update
        if self.final is None or self.threshold is None or converge_update is None:
            raise AssertionError("converged SWAD valley lacks its final average")
        if self._smooth_queue[0].end_update < converge_update:
            return
        if self._smooth_loss(0) > self.threshold:
            # The reference stops here: segments above the threshold never enter
            # the final average, and training halts.
            self.dead = True
            return
        self.final.add(self._smooth_queue[0].state)

    def current_state(self, live: LinearProbeState) -> LinearProbeState:
        """The parameters SWAD would return now, without consuming its queues."""

        if not self.converged or self.final is None or self.threshold is None:
            return live
        if self.dead:
            return self.final.state
        average = self.final.copy()
        window = list(self._smooth_queue)[1:]
        while window:
            if min(item.end_loss for item in window) > self.threshold:
                break
            average.add(window.pop(0).state)
        return average.state
