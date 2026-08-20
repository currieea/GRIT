"""Tracking boundary with a semantics-free null implementation."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field, StrictStr

from grit.schemas import StrictBoundaryModel


class LifecycleEvent(StrictBoundaryModel):
    event_id: StrictStr = Field(min_length=1)
    run_id: StrictStr = Field(min_length=1)
    name: StrictStr = Field(min_length=1)


class EventSink(Protocol):
    def emit(self, event: LifecycleEvent) -> None: ...


class NullEventSink:
    """Discard events without retaining state or influencing control flow."""

    def emit(self, event: LifecycleEvent) -> None:
        del event
