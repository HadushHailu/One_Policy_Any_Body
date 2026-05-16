"""
eval.py — Evaluate a trained checkpoint in simulation

Usage:
    python eval.py checkpoint=data/outputs/.../checkpoints/latest.ckpt robot=so101
"""

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="opab/config",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    from opab.workspace import create_workspace

    workspace = create_workspace(cfg)
    workspace.eval()


if __name__ == "__main__":
    main()
