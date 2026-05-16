"""
Morphology-Conditioned Diffusion Policy — Core implementation

The main policy that combines:
- 1D U-Net diffusion model for action generation
- FiLM conditioning from morphology descriptor
- ResNet-18 observation encoder
- Action chunking (predict H steps, execute k)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from opab.model.diffusion.conditional_unet1d import ConditionalUnet1D
from opab.model.morphology_encoder import MorphologyEncoder


class MorphologyConditionedDP(nn.Module):
    """
    Morphology-conditioned diffusion policy.

    Given observation (image + proprioception) and morphology descriptor,
    generates a chunk of future actions via iterative denoising.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # TODO: Initialize components
        # self.obs_encoder = ...        # ResNet-18 + proprio MLP
        # self.morph_encoder = ...      # URDF features → 32-dim
        # self.noise_net = ...          # 1D U-Net with FiLM
        # self.noise_scheduler = ...    # DDPM/DDIM
        # self.ema = ...                # EMA wrapper

        raise NotImplementedError("Week 1-2 deliverable")

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Diffusion training loss (behavior cloning).

        1. Sample noise ε ~ N(0, I)
        2. Sample timestep t ~ Uniform(1, T)
        3. Noise the action: a_t = √ᾱ_t · a_0 + √(1-ᾱ_t) · ε
        4. Predict noise: ε̂ = ε_θ(a_t, t, obs, morph)
        5. Loss = ||ε - ε̂||²
        """
        raise NotImplementedError

    @torch.no_grad()
    def predict_action(self, obs: dict) -> torch.Tensor:
        """
        Generate action chunk via DDIM denoising.

        1. Start from a_T ~ N(0, I)
        2. For t = T, T-Δ, ..., 0:
              ε̂ = ε_θ(a_t, t, obs, morph)
              a_{t-Δ} = DDIM_step(a_t, ε̂, t)
        3. Return a_0[:execute_horizon]
        """
        raise NotImplementedError


class BaselineDP(nn.Module):
    """
    Standard diffusion policy WITHOUT morphology conditioning.
    Used as baseline B1/B2 in experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError
