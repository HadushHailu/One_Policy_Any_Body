"""
Training workspace for morphology-conditioned diffusion policy.

Orchestrates: data loading → training loop → checkpointing → eval → logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from opab.dataset.multi_robot_dataset import MultiRobotDataset
from opab.model.common.ema import EMAModel
from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP

logger = logging.getLogger(__name__)


class TrainMorphDPWorkspace:
    """Full training lifecycle for the morphology-conditioned diffusion policy."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.training.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Device: {self.device}")

    # ==================================================================
    def run(self) -> None:
        """Main training entry point."""
        cfg = self.cfg

        # ----------------------------------------------------------
        # 1. Data — Load demos from one or many robots/tasks
        # ----------------------------------------------------------
        # Multi-robot mode: cfg.training.robots = ["franka", "ur5", "so101"]
        # Multi-task mode:  cfg.training.tasks  = ["pick_place", "stack"]
        # Falls back to single-robot mode via cfg.robot.name
        robot_names = list(cfg.training.get("robots", [cfg.robot.name]))
        task_names = list(cfg.training.get("tasks", ["pick_place"]))

        hdf5_paths = []
        robot_configs = []
        for robot_name in robot_names:
            robot_cfg = self._load_robot_config(robot_name, cfg)
            for task_name in task_names:
                path = os.path.join("data", "demos", f"{robot_name}_{task_name}.hdf5")
                if os.path.exists(path):
                    hdf5_paths.append(path)
                    robot_configs.append(robot_cfg)
                    logger.info(f"Found demo file: {path}")
                else:
                    logger.warning(f"Demo file not found, skipping: {path}")

        if not hdf5_paths:
            raise FileNotFoundError(
                f"No demo files found for robots={robot_names}, tasks={task_names}. "
                f"Run scripts/generate_sim_demos.py first."
            )

        dataset = MultiRobotDataset(
            hdf5_paths=hdf5_paths,
            robot_configs=robot_configs,
            obs_horizon=cfg.policy.action.obs_horizon,
            action_horizon=cfg.policy.action.horizon,
            max_proprio_dim=cfg.policy.proprio_encoder.input_dim,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        logger.info(
            f"Dataset: {len(dataset)} samples, "
            f"{len(dataloader)} batches/epoch  (bs={cfg.training.batch_size})"
        )

        # ----------------------------------------------------------
        # 2. Policy + EMA
        # ----------------------------------------------------------
        policy = MorphologyConditionedDP(cfg.policy).to(self.device)
        ema = EMAModel(policy, decay=cfg.training.ema_decay)

        # ----------------------------------------------------------
        # 3. Optimiser & LR scheduler
        # ----------------------------------------------------------
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.num_epochs
        )

        # ----------------------------------------------------------
        # 3b. Resume from checkpoint if specified
        # ----------------------------------------------------------
        start_epoch = 0
        global_step = 0
        best_loss = float("inf")
        ckpt_dir = Path(cfg.get("checkpoint_dir", "data/outputs/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        resume_path = cfg.get("resume_checkpoint", None)
        if resume_path and Path(resume_path).exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            policy.load_state_dict(ckpt["model_state_dict"])
            ema.load_state_dict(ckpt["ema_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_loss = ckpt.get("loss", float("inf"))
            # Advance LR scheduler to the correct epoch
            for _ in range(start_epoch):
                lr_scheduler.step()
            logger.info(
                f"  Resumed at epoch {start_epoch}, "
                f"global_step={global_step}, best_loss={best_loss:.6f}"
            )

        # ----------------------------------------------------------
        # 4. Training loop
        # ----------------------------------------------------------
        for epoch in range(start_epoch, cfg.training.num_epochs):
            policy.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss = policy.compute_loss(batch)
                optimizer.zero_grad()
                loss.backward()

                if cfg.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), cfg.training.gradient_clip_norm
                    )

                optimizer.step()
                ema.update(policy)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

            lr_scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # ---------- Logging ----------
            if epoch % cfg.training.log_every == 0:
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch:4d}/{cfg.training.num_epochs} | "
                    f"Loss {avg_loss:.6f} | LR {lr:.2e}"
                )

            # ---------- Checkpoint ----------
            should_save = (
                epoch % cfg.training.checkpoint_every == 0
                or avg_loss < best_loss
            )
            if should_save:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "normalizer": dataset.normalizer.get_state(),
                }
                # Save config safely (resolve=False avoids Hydra interpolation errors)
                try:
                    ckpt["config"] = OmegaConf.to_container(cfg, resolve=True)
                except Exception:
                    ckpt["config"] = OmegaConf.to_container(cfg, resolve=False)
                torch.save(ckpt, ckpt_dir / "latest.pt")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(ckpt, ckpt_dir / "best.pt")
                    logger.info(f"  New best loss: {best_loss:.6f}")

        logger.info(
            f"Training complete — {cfg.training.num_epochs} epochs, "
            f"best loss {best_loss:.6f}"
        )

    # ==================================================================
    @staticmethod
    def _load_robot_config(robot_name: str, cfg: DictConfig):
        """Load robot config by name, falling back to cfg.robot for the default."""
        from omegaconf import OmegaConf
        config_path = Path(__file__).resolve().parents[1] / "config" / "robot" / f"{robot_name}.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)
        # Fallback to the active config's robot section
        if cfg.robot.name == robot_name:
            return cfg.robot
        raise FileNotFoundError(f"Robot config not found: {config_path}")

    # ==================================================================
    def eval(self, policy=None, normalizer=None) -> float:
        """
        Run policy rollouts in sim, return success rate.

        Can be called standalone or during training for periodic eval.
        """
        from collections import deque

        import numpy as np

        from opab.env import make_env
        from opab.model.morphology_encoder import MorphologyEncoder

        cfg = self.cfg
        n_episodes = cfg.training.get("num_eval_episodes", 10)
        robot_name = cfg.robot.name

        if policy is None or normalizer is None:
            raise ValueError("Must pass policy and normalizer for eval")

        morph_vec = MorphologyEncoder.from_robot_config(cfg.robot)
        env = make_env(robot_name, seed=99)

        obs_horizon = cfg.policy.action.obs_horizon
        execute_horizon = cfg.policy.action.execute_horizon
        max_proprio_dim = cfg.policy.proprio_encoder.input_dim

        policy.eval()
        successes = []
        motions = []

        for ep in range(n_episodes):
            obs = env.reset()
            obs_buffer = deque(maxlen=obs_horizon)
            for _ in range(obs_horizon):
                obs_buffer.append(obs)

            action_queue = []
            trajectory = []
            steps = 0

            while steps < env.max_episode_steps:
                if len(action_queue) == 0:
                    images, proprios = [], []
                    for o in obs_buffer:
                        img = o["image"].astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))
                        images.append(img)
                        p = o["proprioception"].astype(np.float32)
                        if len(p) < max_proprio_dim:
                            p = np.pad(p, (0, max_proprio_dim - len(p)))
                        proprios.append(p[:max_proprio_dim])

                    import torch as th

                    obs_img = th.from_numpy(np.stack(images)).unsqueeze(0).to(self.device)
                    obs_prop = th.from_numpy(np.stack(proprios)).unsqueeze(0).to(self.device)
                    obs_prop = normalizer.normalize("proprioception", obs_prop)
                    morph_batch = morph_vec.unsqueeze(0).to(self.device)

                    with th.no_grad():
                        chunk = policy.predict_action(obs_img, obs_prop, morph_batch)
                    chunk = normalizer.unnormalize("actions", chunk[0])
                    action_queue = list(chunk.cpu().numpy())

                action = action_queue.pop(0)
                obs, _, terminated, truncated, info = env.step(action)
                obs_buffer.append(obs)
                trajectory.append(obs["ee_pos"].copy())
                steps += 1
                if terminated or truncated:
                    break

            traj = np.array(trajectory)
            motion = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) if len(traj) > 1 else 0.0
            successes.append(info.get("success", False))
            motions.append(motion)

        env.close()
        success_rate = sum(successes) / max(len(successes), 1)
        avg_motion = np.mean(motions)
        logger.info(
            f"  Eval: {sum(successes)}/{n_episodes} success, "
            f"avg_motion={avg_motion:.4f}m"
        )
        return success_rate
