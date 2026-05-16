"""Workspace factory — creates the right workspace from Hydra config."""

from __future__ import annotations

from omegaconf import DictConfig


def create_workspace(cfg: DictConfig):
    """Instantiate the appropriate workspace based on policy type."""
    policy_name = cfg.policy.name

    if policy_name == "morphology_dp":
        from opab.workspace.train_morph_dp_workspace import TrainMorphDPWorkspace
        return TrainMorphDPWorkspace(cfg)
    elif policy_name == "morphology_dp_dpo":
        from opab.workspace.train_dpo_workspace import TrainDPOWorkspace
        return TrainDPOWorkspace(cfg)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def load_policy_from_checkpoint(cfg: DictConfig):
    """Load a trained policy from checkpoint for evaluation."""
    # TODO: Implement checkpoint loading
    raise NotImplementedError
