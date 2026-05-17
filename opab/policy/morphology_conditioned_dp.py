"""
Morphology-Conditioned Diffusion Policy — Core implementation.

Combines:
  - 1D U-Net diffusion backbone for action-sequence denoising
  - FiLM conditioning from morphology descriptor
  - ResNet-18 observation encoder
  - Action chunking (predict H, execute k)
  - DDPM training / DDIM inference
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from opab.model.diffusion.conditional_unet1d import ConditionalUnet1D
from opab.model.morphology_encoder import MorphologyEncoder
from opab.model.vision.obs_encoder import ObsEncoder

logger = logging.getLogger(__name__)


class MorphologyConditionedDP(nn.Module):
    """
    Morphology-conditioned diffusion policy.

    Given observation (image + proprioception) and morphology descriptor,
    generates a chunk of future actions via iterative denoising.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---- Observation encoder ----
        self.obs_encoder = ObsEncoder(
            obs_horizon=cfg.action.obs_horizon,
            proprio_dim=cfg.proprio_encoder.input_dim,
            output_dim=cfg.obs_encoder.output_dim,
            pretrained=cfg.obs_encoder.pretrained,
            frozen=cfg.obs_encoder.get("frozen", False),
        )

        # ---- Morphology encoder ----
        self.morph_encoder = MorphologyEncoder(cfg)

        # ---- 1D U-Net noise network ----
        self.noise_net = ConditionalUnet1D(
            action_dim=cfg.action.action_dim,
            cond_dim=cfg.obs_encoder.output_dim,
            morph_dim=cfg.morphology_encoder.output_dim,
            down_dims=list(cfg.unet.down_dims),
            kernel_size=cfg.unet.kernel_size,
        )

        # ---- Noise scheduler (training) ----
        schedule_map = {"cosine": "squaredcos_cap_v2", "linear": "linear"}
        beta_schedule = schedule_map.get(
            cfg.diffusion.noise_schedule, cfg.diffusion.noise_schedule
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=False,
        )

        # ---- Inference scheduler config ----
        self._inference_scheduler_cfg = dict(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=False,
        )
        self._num_inference_steps = cfg.diffusion.num_inference_steps

        # ---- Dimensions ----
        self.action_dim = cfg.action.action_dim
        self.action_horizon = cfg.action.horizon
        self.execute_horizon = cfg.action.execute_horizon

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MorphologyConditionedDP: {n_params / 1e6:.1f}M parameters")

    # ==================================================================
    # Training
    # ==================================================================
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Diffusion denoising loss (ε-prediction).

        Args:
            batch: dict with keys obs_images, obs_proprio, action, morph_vec

        Returns:
            Scalar MSE loss.
        """
        obs_embed = self.obs_encoder(batch["obs_images"], batch["obs_proprio"])
        morph_embed = self.morph_encoder(batch["morph_vec"])

        # (B, horizon, action_dim) → (B, action_dim, horizon) for Conv1D
        action = batch["action"].permute(0, 2, 1)

        noise = torch.randn_like(action)
        B = action.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=action.device,
        ).long()

        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noise_pred = self.noise_net(noisy_action, timesteps, obs_embed, morph_embed)

        return F.mse_loss(noise_pred, noise)

    # ==================================================================
    # Inference
    # ==================================================================
    @torch.no_grad()
    def predict_action(
        self,
        obs_images: torch.Tensor,
        obs_proprio: torch.Tensor,
        morph_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate an action chunk via DDIM denoising.

        Returns:
            actions: (B, execute_horizon, action_dim)
        """
        obs_embed = self.obs_encoder(obs_images, obs_proprio)
        morph_embed = self.morph_encoder(morph_vec)
        B = obs_embed.shape[0]

        scheduler = DDIMScheduler(**self._inference_scheduler_cfg)
        scheduler.set_timesteps(self._num_inference_steps, device=obs_embed.device)

        action = torch.randn(
            B, self.action_dim, self.action_horizon, device=obs_embed.device
        )

        for t in scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(B)
            noise_pred = self.noise_net(action, t_batch, obs_embed, morph_embed)
            action = scheduler.step(noise_pred, t, action).prev_sample

        # (B, action_dim, horizon) → (B, horizon, action_dim), take first k
        return action.permute(0, 2, 1)[:, : self.execute_horizon, :]


class BaselineDP(nn.Module):
    """
    Standard diffusion policy WITHOUT morphology conditioning.
    Used as baseline B1/B2 in experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("Week 6 deliverable")
