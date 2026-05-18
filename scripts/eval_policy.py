#!/usr/bin/env python3
"""
Evaluate a trained diffusion policy in MuJoCo simulation.

Usage:
    python scripts/eval_policy.py --checkpoint data/outputs/checkpoints/best.pt --robot franka
    python scripts/eval_policy.py --checkpoint data/outputs/checkpoints/best.pt --robot franka --render
    python scripts/eval_policy.py --checkpoint data/outputs/checkpoints/best.pt --robot franka --n-episodes 20
"""

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.dataset.normalizer import Normalizer
from opab.env import make_env, SUPPORTED_TASKS
from opab.model.morphology_encoder import MorphologyEncoder
from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP
from opab.policy.temporal_ensemble import TemporalEnsemble

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_policy_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load trained policy + normalizer from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    policy = MorphologyConditionedDP(cfg.policy).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    normalizer = Normalizer()
    normalizer.load_state(ckpt["normalizer"])

    logger.info(
        f"Loaded checkpoint: epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}"
    )
    return policy, normalizer, cfg


def run_episode(
    env,
    policy,
    normalizer,
    morph_vec,
    task_id: int = 0,
    device="cuda",
    obs_horizon=2,
    execute_horizon=8,
    max_proprio_dim=7,
    render=False,
    use_ensemble=False,
    ensemble_decay=0.01,
):
    """
    Run one episode with the trained policy.

    Returns:
        success: bool
        total_steps: int
        trajectory: list of ee_pos at each step
    """
    obs = env.reset()
    obs_buffer = deque(maxlen=obs_horizon)

    # Fill buffer with initial obs
    for _ in range(obs_horizon):
        obs_buffer.append(obs)

    trajectory = []
    total_steps = 0
    action_queue = []

    # Temporal ensemble
    ensemble = None
    if use_ensemble:
        ensemble = TemporalEnsemble(
            action_dim=policy.action_dim, decay=ensemble_decay
        )

    # Task ID tensor
    task_id_tensor = torch.tensor([task_id], dtype=torch.long, device=device)

    while total_steps < env.max_episode_steps:
        # If no actions queued, run inference
        if len(action_queue) == 0:
            # Prepare observation tensors
            images = []
            proprios = []
            for o in obs_buffer:
                img = o["image"].astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # CHW
                images.append(img)

                proprio = o["proprioception"].astype(np.float32)
                # Pad to max_proprio_dim
                if len(proprio) < max_proprio_dim:
                    proprio = np.pad(
                        proprio, (0, max_proprio_dim - len(proprio))
                    )
                else:
                    proprio = proprio[:max_proprio_dim]
                proprios.append(proprio)

            obs_images = (
                torch.from_numpy(np.stack(images))
                .unsqueeze(0)
                .to(device)
            )  # (1, T_o, 3, H, W)
            obs_proprio = (
                torch.from_numpy(np.stack(proprios))
                .unsqueeze(0)
                .to(device)
            )  # (1, T_o, proprio_dim)

            # Normalize proprio
            obs_proprio = normalizer.normalize("proprioception", obs_proprio)

            morph_batch = morph_vec.unsqueeze(0).to(device)  # (1, 46)

            # Run policy inference (DDIM) with task conditioning
            action_chunk = policy.predict_action(
                obs_images, obs_proprio, morph_batch, task_id=task_id_tensor
            )  # (1, execute_horizon, action_dim)

            # Unnormalize actions
            action_chunk = normalizer.unnormalize(
                "actions", action_chunk[0]
            )  # (execute_horizon, action_dim)

            chunk_np = action_chunk.cpu().numpy()

            if use_ensemble and ensemble is not None:
                ensemble.add_chunk(total_steps, chunk_np)
                # Generate actions from ensemble for next execute_horizon steps
                action_queue = []
                for i in range(execute_horizon):
                    step = total_steps + i
                    try:
                        action_queue.append(ensemble.get_action(step))
                    except ValueError:
                        break
            else:
                action_queue = list(chunk_np)

        # Execute next action
        action = action_queue.pop(0)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_buffer.append(obs)
        trajectory.append(obs["ee_pos"].copy())
        total_steps += 1

        if render:
            import time
            time.sleep(env.control_dt)

        if terminated or truncated:
            break

    success = info.get("success", False)
    return success, total_steps, trajectory


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/outputs/checkpoints/best.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--robot", default="franka", choices=["franka", "ur5", "so101"]
    )
    parser.add_argument(
        "--task", default="pick_place", choices=["pick_place", "stack"]
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Render with viewer")
    parser.add_argument("--ensemble", action="store_true", help="Enable temporal ensemble")
    parser.add_argument("--ensemble-decay", type=float, default=0.01, help="Ensemble decay rate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load policy
    policy, normalizer, cfg = load_policy_from_checkpoint(
        args.checkpoint, device
    )

    # Get robot config for morphology vector
    robot_cfg = OmegaConf.load(
        Path(__file__).parents[1]
        / "opab"
        / "config"
        / "robot"
        / f"{args.robot}.yaml"
    )
    morph_vec = MorphologyEncoder.from_robot_config(robot_cfg)

    # Create env
    env = make_env(args.robot, seed=args.seed, task=args.task)

    logger.info(f"\nEvaluating {args.robot} / {args.task} for {args.n_episodes} episodes...")
    if args.ensemble:
        logger.info(f"  Temporal ensemble: ON (decay={args.ensemble_decay})")
    logger.info(f"{'='*50}")

    # Task ID for conditioning
    task_id = SUPPORTED_TASKS.index(args.task)  # 0=pick_place, 1=stack

    successes = []
    all_steps = []
    all_motion = []

    for ep in range(args.n_episodes):
        success, steps, traj = run_episode(
            env,
            policy,
            normalizer,
            morph_vec,
            task_id=task_id,
            device=device,
            obs_horizon=cfg.policy.action.obs_horizon,
            execute_horizon=cfg.policy.action.execute_horizon,
            max_proprio_dim=cfg.policy.proprio_encoder.input_dim,
            render=args.render,
            use_ensemble=args.ensemble,
            ensemble_decay=args.ensemble_decay,
        )

        # Compute total motion (how much did the EE move?)
        traj = np.array(traj)
        total_motion = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        all_motion.append(total_motion)

        successes.append(success)
        all_steps.append(steps)

        status = "SUCCESS" if success else "fail"
        logger.info(
            f"  Episode {ep+1:2d}: {status} | "
            f"steps={steps:3d} | motion={total_motion:.4f}m"
        )

    logger.info(f"{'='*50}")
    logger.info(f"Success rate: {sum(successes)}/{args.n_episodes} "
                f"({100*sum(successes)/args.n_episodes:.0f}%)")
    logger.info(f"Avg steps: {np.mean(all_steps):.0f}")
    logger.info(f"Avg EE motion: {np.mean(all_motion):.4f}m")
    logger.info(
        f"Motion check: {'INTENTIONAL (>0.05m)' if np.mean(all_motion) > 0.05 else 'RANDOM/STATIC (<0.05m)'}"
    )

    env.close()


if __name__ == "__main__":
    main()
