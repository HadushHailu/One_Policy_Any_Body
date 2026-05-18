"""
Temporal Ensemble for Action Smoothing.

When diffusion policy predicts overlapping action chunks, this module
averages them with exponential decay weighting to produce smoother
trajectories (Zhao et al., ACT 2023).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class TemporalEnsemble:
    """
    Maintains a buffer of predicted action chunks and produces a
    weighted-average action at each timestep.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    decay : float
        Exponential decay rate. Lower = more smoothing.
        0.01 is typical for 10Hz control.
    max_chunks : int
        Maximum number of overlapping chunks to keep in buffer.
    """

    def __init__(self, action_dim: int = 4, decay: float = 0.01, max_chunks: int = 10):
        self.action_dim = action_dim
        self.decay = decay
        self.max_chunks = max_chunks
        self.buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=max_chunks)

    def reset(self):
        """Clear the buffer (call at episode start)."""
        self.buffer.clear()

    def add_chunk(self, global_step: int, action_chunk: np.ndarray):
        """
        Add a predicted action chunk to the buffer.

        Args:
            global_step: The environment timestep at which this chunk starts.
            action_chunk: (chunk_len, action_dim) array of predicted actions.
        """
        self.buffer.append((global_step, action_chunk.copy()))

    def get_action(self, global_step: int) -> np.ndarray:
        """
        Get the ensembled action for the current timestep.

        Args:
            global_step: Current environment timestep.

        Returns:
            (action_dim,) array — weighted average across all chunks
            that have a prediction for this timestep.
        """
        weights = []
        actions = []

        for start_step, chunk in self.buffer:
            idx = global_step - start_step
            if 0 <= idx < len(chunk):
                # Weight decays with how far into the chunk we are
                w = math.exp(-self.decay * idx)
                weights.append(w)
                actions.append(chunk[idx])

        if not actions:
            raise ValueError(
                f"No predicted actions available for step {global_step}. "
                f"Buffer has {len(self.buffer)} chunks."
            )

        weights = np.array(weights)
        weights /= weights.sum()
        return sum(w * a for w, a in zip(weights, actions))

    @property
    def has_actions(self) -> bool:
        """Whether the buffer has any active chunks."""
        return len(self.buffer) > 0
