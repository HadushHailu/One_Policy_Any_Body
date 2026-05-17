"""
Exponential Moving Average (EMA) model wrapper.

Maintains a shadow copy of model weights updated as:
    θ_ema ← decay · θ_ema + (1 - decay) · θ

Use EMA weights for evaluation (more stable than raw training weights).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMAModel:
    """Lightweight EMA wrapper — no nn.Module subclass needed."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        self._original: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights one step."""
        for key, param in model.state_dict().items():
            if param.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(param)

    # ------------------------------------------------------------------
    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model weights with EMA shadow (saves originals for restore)."""
        self._original = deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore the original (training) weights after evaluation."""
        if self._original is not None:
            model.load_state_dict(self._original)
            self._original = None

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}
