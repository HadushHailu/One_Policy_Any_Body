"""
Conditional 1D U-Net with FiLM modulation for diffusion policy.

Architecture:
    - 1D convolutions along action time dimension
    - Skip connections between encoder and decoder
    - FiLM conditioning from morphology + observation embeddings
    - Sinusoidal timestep embedding

See docs/planning/02_One_Policy_Any_Body_objective.md § F2, F3 for theory.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding (same as in transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ(c) ⊙ h + β(c)"""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(cond).unsqueeze(-1)   # (B, C, 1) for Conv1D
        beta = self.shift(cond).unsqueeze(-1)
        return gamma * h + beta


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.film = FiLMLayer(cond_dim, out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.film(x, cond)
        x = self.act(x + residual)
        return x


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net for diffusion policy with FiLM conditioning.

    Input:  (batch, action_dim, horizon) — noised action sequence
    Output: (batch, action_dim, horizon) — predicted noise ε

    Conditioning: timestep + observation embedding + morphology embedding
    """

    def __init__(
        self,
        action_dim: int = 7,
        cond_dim: int = 256,
        morph_dim: int = 32,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 5,
    ):
        super().__init__()

        # Timestep embedding
        time_dim = 128
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Combined conditioning: time + obs + morphology
        total_cond_dim = time_dim + cond_dim + morph_dim

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = action_dim
        for out_ch in down_dims:
            self.encoder_blocks.append(
                ConditionalResidualBlock1D(in_ch, out_ch, total_cond_dim, kernel_size)
            )
            self.downsample.append(nn.Conv1d(out_ch, out_ch, 2, stride=2))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConditionalResidualBlock1D(
            down_dims[-1], down_dims[-1], total_cond_dim, kernel_size
        )

        # Decoder (upsampling) — mirrors encoder fully (one level per encoder level)
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(down_dims)):
            in_ch = down_dims[-(i + 1)]
            skip_ch = in_ch  # skip from the corresponding encoder level
            out_ch = down_dims[-(i + 2)] if i + 1 < len(down_dims) else down_dims[0]
            self.upsample.append(nn.ConvTranspose1d(in_ch, in_ch, 2, stride=2))
            self.decoder_blocks.append(
                ConditionalResidualBlock1D(in_ch + skip_ch, out_ch, total_cond_dim, kernel_size)
            )

        # Final projection back to action_dim
        self.final_conv = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,       # (B, action_dim, horizon) — noised actions
        t: torch.Tensor,        # (B,) — diffusion timestep
        obs: torch.Tensor,      # (B, cond_dim) — observation embedding
        morph: torch.Tensor,    # (B, morph_dim) — morphology embedding
    ) -> torch.Tensor:
        """Predict noise ε given noised action, timestep, observation, and morphology."""
        # Build conditioning vector
        t_emb = self.time_embed(t.float())
        cond = torch.cat([t_emb, obs, morph], dim=-1)

        # Encoder with skip connections
        skips = []
        h = x
        for block, down in zip(self.encoder_blocks, self.downsample):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck(h, cond)

        # Decoder — symmetric with encoder
        for i, (block, up) in enumerate(zip(self.decoder_blocks, self.upsample)):
            h = up(h)
            skip = skips[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.final_conv(h)
