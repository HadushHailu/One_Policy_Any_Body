"""Smoke tests for OPAB — ensuring basic imports and shapes work."""

import pytest
import torch


def test_import():
    """Verify the package is importable."""
    import opab
    assert opab.__version__ == "0.1.0"


def test_morphology_encoder_shapes():
    """Morphology encoder produces correct output shape."""
    from opab.model.morphology_encoder import MorphologyEncoder
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "morphology_encoder": {
            "input_dim": 46,
            "hidden_dims": [64, 64],
            "output_dim": 32,
            "activation": "relu",
            "use_layer_norm": True,
        }
    })

    encoder = MorphologyEncoder(cfg)
    x = torch.randn(4, 46)
    out = encoder(x)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"


def test_unet_forward_shapes():
    """U-Net produces output with same shape as input."""
    from opab.model.diffusion.conditional_unet1d import ConditionalUnet1D

    model = ConditionalUnet1D(
        action_dim=7,
        cond_dim=256,
        morph_dim=32,
        down_dims=[64, 128],  # small for test
        kernel_size=3,
    )

    batch_size = 2
    horizon = 16
    x = torch.randn(batch_size, 7, horizon)
    t = torch.randint(0, 100, (batch_size,)).float()
    obs = torch.randn(batch_size, 256)
    morph = torch.randn(batch_size, 32)

    out = model(x, t, obs, morph)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_film_modulation():
    """FiLM layer modulates features correctly."""
    from opab.model.diffusion.conditional_unet1d import FiLMLayer

    film = FiLMLayer(cond_dim=32, feature_dim=64)
    h = torch.randn(4, 64, 16)    # (B, C, T)
    cond = torch.randn(4, 32)
    out = film(h, cond)
    assert out.shape == h.shape
