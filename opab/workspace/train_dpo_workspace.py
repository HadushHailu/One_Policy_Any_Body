"""Training workspace for DPO fine-tuning of diffusion policies."""

from __future__ import annotations

from omegaconf import DictConfig


class TrainDPOWorkspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self):
        """DPO fine-tuning loop."""
        # TODO: Implement Week 7 deliverable
        raise NotImplementedError("Week 7 deliverable: DPO training loop")

    def eval(self):
        raise NotImplementedError
