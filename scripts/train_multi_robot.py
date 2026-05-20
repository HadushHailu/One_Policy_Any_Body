#!/usr/bin/env python3
"""
Multi-robot multi-task training — trains one policy on 3 robots × 5 tasks.

Usage:
    python scripts/train_multi_robot.py
    python scripts/train_multi_robot.py --epochs 200 --device cuda
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multi-robot diffusion policy training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--robots", nargs="+", default=["franka", "ur5", "so101"],
                        help="Robots to train on")
    parser.add_argument("--tasks", nargs="+",
                        default=["reach", "pick_place", "push", "stack", "peg_insertion"],
                        help="Tasks to train on")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (latest.pt or best.pt) to resume from")
    args = parser.parse_args()

    # Build config from defaults + overrides
    project_root = Path(__file__).resolve().parents[1]
    cfg_dir = project_root / "opab" / "config"

    default_cfg = OmegaConf.load(cfg_dir / "default.yaml")
    policy_cfg = OmegaConf.load(cfg_dir / "policy" / "morphology_dp.yaml")
    robot_cfg = OmegaConf.load(cfg_dir / "robot" / "franka.yaml")  # primary robot

    cfg = OmegaConf.merge(default_cfg, {"policy": policy_cfg, "robot": robot_cfg})

    # Override training params
    cfg.training.num_epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.device = args.device
    cfg.training.lr = args.lr
    cfg.training.robots = args.robots
    cfg.training.tasks = args.tasks

    # Set output dir
    import datetime
    ts = datetime.datetime.now().strftime("%Y.%m.%d/%H.%M.%S")
    n_robots = len(args.robots)
    n_tasks = len(args.tasks)
    cfg.output_dir = f"data/outputs/{ts}_multi_{n_robots}r_{n_tasks}t"
    cfg.checkpoint_dir = f"{cfg.output_dir}/checkpoints"
    cfg.media_dir = f"{cfg.output_dir}/media"

    # If resuming, use the same output dir as the checkpoint
    if args.resume:
        ckpt_path = Path(args.resume)
        cfg.checkpoint_dir = str(ckpt_path.parent)
        cfg.output_dir = str(ckpt_path.parent.parent)
        cfg.resume_checkpoint = str(ckpt_path)
        logger.info(f"  Resuming from: {args.resume}")

    logger.info(f"Training config:")
    logger.info(f"  Robots: {args.robots}")
    logger.info(f"  Tasks:  {args.tasks}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch:  {args.batch_size}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Output: {cfg.output_dir}")

    from opab.workspace.train_morph_dp_workspace import TrainMorphDPWorkspace
    workspace = TrainMorphDPWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
