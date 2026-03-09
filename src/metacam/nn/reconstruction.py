from __future__ import annotations

"""Minimal interfaces for upcoming neural reconstruction work."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ReconstructionBatch:
    measurement: torch.Tensor
    target: torch.Tensor | None = None
    metadata: dict | None = None


class ReconstructionModel(nn.Module):
    """Base class for trainable inverse reconstruction models."""

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError
