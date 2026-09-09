"""The persisted linear-probe inference state shared by every CMNIST method."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LinearProbeState:
    weight: torch.Tensor
    bias: torch.Tensor
