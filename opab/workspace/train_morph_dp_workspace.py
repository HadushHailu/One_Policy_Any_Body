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
        # 1. Data
        # ----------------------------------------------------------
        robot_name = cfg.robot.name
        data_path = os.path.join("data", "demos", f"{robot_name}_pick_place.hdf5")

        dataset = MultiRobotDataset(
            hdf5_paths=[data_path],
            robot_configs=[cfg.robot],
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
        # 4. Training loop
        # ----------------------------------------------------------
        global_step = 0
        best_loss = float("inf")
        ckpt_dir = Path(cfg.get("checkpoint_dir", "data/outputs/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg.training.num_epochs):
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
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
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
    def eval(self) -> None:
        """Evaluate a checkpoint in simulation (Week 3)."""
        raise NotImplementedError("Week 3 deliverable: sim evaluation loop")
