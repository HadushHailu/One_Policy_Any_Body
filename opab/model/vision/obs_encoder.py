"""
Observation encoder — ResNet-18 image backbone + proprioception MLP.

Processes multi-frame image observations and joint-position proprioception
into a single conditioning vector for the diffusion U-Net.

Architecture:
    images (B, T_o, 3, 84, 84) → shared ResNet-18 → avg-pool frames → 512 → proj → 256
    proprio (B, T_o, n_joints)  → shared MLP → avg-pool frames → 64
    concat [256, 64] = 320 → Linear → 256-dim obs_embed
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ObsEncoder(nn.Module):
    """Encode multi-frame observations into a fixed-size vector."""

    def __init__(
        self,
        obs_horizon: int = 2,
        proprio_dim: int = 7,
        output_dim: int = 256,
        pretrained: bool = True,
        frozen: bool = False,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon

        # ---- Image backbone (shared across frames) ----
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # Remove final FC and avgpool — we add our own
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # → (B, 512, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 512, 1, 1)
        self.img_proj = nn.Linear(512, 256)

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Proprioception MLP (shared across frames) ----
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # ---- Fusion ----
        self.fuse = nn.Sequential(
            nn.Linear(256 + 64, output_dim),
            nn.ReLU(),
        )

        # ImageNet normalization constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(
        self, images: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images:  (B, obs_horizon, 3, H, W) float32 in [0, 1]
            proprio: (B, obs_horizon, proprio_dim) float32 (already normalised)

        Returns:
            obs_embed: (B, output_dim)
        """
        B, T, C, H, W = images.shape

        # ---- Image encoding ----
        imgs = images.reshape(B * T, C, H, W)
        imgs = (imgs - self.img_mean) / self.img_std
        feats = self.backbone(imgs)          # (B*T, 512, H', W')
        feats = self.pool(feats).flatten(1)  # (B*T, 512)
        feats = self.img_proj(feats)         # (B*T, 256)
        feats = feats.view(B, T, 256).mean(dim=1)  # (B, 256)

        # ---- Proprioception encoding ----
        prop = proprio.reshape(B * T, -1)
        prop = self.proprio_mlp(prop)               # (B*T, 64)
        prop = prop.view(B, T, 64).mean(dim=1)      # (B, 64)

        # ---- Fusion ----
        return self.fuse(torch.cat([feats, prop], dim=-1))  # (B, output_dim)
