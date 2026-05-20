"""
train.py — Main training entrypoint

Usage:
    # Train morphology-conditioned diffusion policy on Franka pick-and-place
    python train.py policy=morphology_dp task=franka_pick robot=franka

    # Train on UR5
    python train.py policy=morphology_dp task=ur5_pick robot=ur5

    # Multi-embodiment training (Franka + UR5)
    python train.py policy=morphology_dp task=franka_pick,ur5_pick robot=franka,ur5

    # DPO fine-tuning from checkpoint
    python train.py policy=morphology_dp_dpo \
        dpo.reference_checkpoint=data/outputs/.../latest.ckpt

    # Multi-seed sweep
    python train.py --multirun training.seed=42,43,44
"""

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    # Lazy imports to avoid slow startup for --help
    from opab.workspace import create_workspace

    workspace = create_workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
