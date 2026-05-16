"""
Training workspace for morphology-conditioned diffusion policy.

Orchestrates the full training lifecycle:
    data loading → training loop → checkpointing → evaluation → logging
"""

from __future__ import annotations

from omegaconf import DictConfig


class TrainMorphDPWorkspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # TODO: Initialize dataset, policy, optimizer, scheduler, logger

    def run(self):
        """Main training loop."""
        # TODO: Implement Week 2 deliverable
        raise NotImplementedError("Week 2 deliverable: full training loop")

    def eval(self):
        """Evaluate checkpoint in simulation."""
        raise NotImplementedError
