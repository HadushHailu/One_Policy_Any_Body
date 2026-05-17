"""
Per-dimension normalization for actions and proprioception.

Computes zero-mean unit-variance statistics from training data
and applies them consistently during training and inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml


class Normalizer:
    """Zero-mean unit-variance normalization per dimension."""

    def __init__(self):
        self.stats: dict[str, dict[str, np.ndarray]] = {}

    def fit(self, data_dict: dict[str, np.ndarray]) -> None:
        """Compute stats from {key: (N, D)} arrays."""
        for key, data in data_dict.items():
            self.stats[key] = {
                "mean": data.mean(axis=0).astype(np.float32),
                "std": data.std(axis=0).astype(np.float32).clip(min=1e-6),
            }

    def normalize(self, key: str, data: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        mean = torch.as_tensor(s["mean"], device=data.device, dtype=data.dtype)
        std = torch.as_tensor(s["std"], device=data.device, dtype=data.dtype)
        return (data - mean) / std

    def unnormalize(self, key: str, data: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        mean = torch.as_tensor(s["mean"], device=data.device, dtype=data.dtype)
        std = torch.as_tensor(s["std"], device=data.device, dtype=data.dtype)
        return data * std + mean

    def save(self, path: str | Path) -> None:
        out = {}
        for key, s in self.stats.items():
            out[key] = {
                "mean": s["mean"].tolist(),
                "std": s["std"].tolist(),
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(out, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            raw = yaml.safe_load(f)
        for key, s in raw.items():
            self.stats[key] = {
                "mean": np.array(s["mean"], dtype=np.float32),
                "std": np.array(s["std"], dtype=np.float32),
            }

    def get_state(self) -> dict:
        """Serializable state for checkpointing."""
        return {
            k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
            for k, v in self.stats.items()
        }

    def load_state(self, state: dict) -> None:
        for k, v in state.items():
            self.stats[k] = {
                "mean": np.array(v["mean"], dtype=np.float32),
                "std": np.array(v["std"], dtype=np.float32),
            }
