"""
Week 2 tests — dataset, forward pass, training step, and overfitting.

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_training.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def _make_policy_cfg(**overrides):
    """Return a minimal OmegaConf policy config for testing."""
    base = {
        "name": "morphology_dp",
        "diffusion": {
            "num_train_timesteps": 100,
            "num_inference_steps": 4,  # fast for tests
            "noise_schedule": "cosine",
            "prediction_type": "epsilon",
        },
        "action": {
            "horizon": 16,
            "obs_horizon": 2,
            "action_dim": 4,
            "execute_horizon": 8,
        },
        "unet": {
            "down_dims": [64, 128, 256],  # small for tests
            "kernel_size": 5,
            "n_groups": 8,
            "use_film": True,
        },
        "obs_encoder": {
            "type": "resnet18",
            "pretrained": False,  # no download in CI
            "frozen": False,
            "output_dim": 256,
            "image_size": 84,
            "crop_shape": [76, 76],
        },
        "proprio_encoder": {
            "input_dim": 7,
            "hidden_dims": [64, 64],
            "output_dim": 64,
        },
        "morphology_encoder": {
            "input_dim": 46,
            "hidden_dims": [64, 64],
            "output_dim": 32,
            "activation": "relu",
            "use_layer_norm": True,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_robot_cfg():
    """Minimal robot config (Franka-like) for testing."""
    return OmegaConf.create(
        {
            "name": "franka",
            "n_joints": 7,
            "dh_params": [[0.0, 0.333, 0.0, 0.0]] * 7,
            "joint_limits": {
                "lower": [-2.9] * 7,
                "upper": [2.9] * 7,
            },
            "workspace_radius": 0.855,
            "payload_kg": 3.0,
            "gripper_max_width": 0.08,
        }
    )


def _make_dummy_hdf5(path: str | Path, n_episodes: int = 2, T: int = 50):
    """Write a tiny HDF5 demo file for testing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["robot"] = "franka"
        f.attrs["n_episodes"] = n_episodes
        f.attrs["success_rate"] = 1.0
        f.attrs["total_steps"] = n_episodes * T
        for i in range(n_episodes):
            ep = f.create_group(f"episode_{i}")
            ep.create_dataset("images", data=np.random.randint(0, 255, (T, 84, 84, 3), dtype=np.uint8))
            ep.create_dataset("actions", data=np.random.randn(T, 4).astype(np.float32) * 0.01)
            ep.create_dataset("proprioception", data=np.random.randn(T, 7).astype(np.float32) * 0.1)
            ep.create_dataset("ee_pos", data=np.random.randn(T, 3).astype(np.float32))
            ep.create_dataset("ee_quat", data=np.random.randn(T, 4).astype(np.float32))
            ep.create_dataset("gripper_pos", data=np.random.randn(T, 1).astype(np.float32))
            ep.attrs["n_steps"] = T
            ep.attrs["success"] = True


# ===================================================================
# Dataset tests
# ===================================================================
class TestDataset:
    def test_dataset_loads(self, tmp_path):
        """MultiRobotDataset loads HDF5, returns correct shapes."""
        from opab.dataset.multi_robot_dataset import MultiRobotDataset

        hdf5_path = tmp_path / "test.hdf5"
        _make_dummy_hdf5(hdf5_path, n_episodes=2, T=50)

        ds = MultiRobotDataset(
            hdf5_paths=[str(hdf5_path)],
            robot_configs=[_make_robot_cfg()],
            obs_horizon=2,
            action_horizon=16,
        )
        assert len(ds) > 0

        sample = ds[0]
        assert sample["obs_images"].shape == (2, 3, 84, 84)
        assert sample["obs_proprio"].shape == (2, 7)
        assert sample["action"].shape == (16, 4)
        assert sample["morph_vec"].shape == (46,)

    def test_normalization(self, tmp_path):
        """Action normalization produces roughly zero-mean."""
        from opab.dataset.multi_robot_dataset import MultiRobotDataset

        hdf5_path = tmp_path / "test_norm.hdf5"
        _make_dummy_hdf5(hdf5_path, n_episodes=3, T=60)

        ds = MultiRobotDataset(
            hdf5_paths=[str(hdf5_path)],
            robot_configs=[_make_robot_cfg()],
        )
        actions = torch.stack([ds[i]["action"] for i in range(min(100, len(ds)))])
        # Normalized actions should have roughly zero mean
        assert actions.mean().abs() < 1.0  # loose bound
        assert actions.std() > 0.1  # not all zeros

    def test_images_in_0_1(self, tmp_path):
        """Observation images are float in [0, 1]."""
        from opab.dataset.multi_robot_dataset import MultiRobotDataset

        hdf5_path = tmp_path / "test_img.hdf5"
        _make_dummy_hdf5(hdf5_path)

        ds = MultiRobotDataset(
            hdf5_paths=[str(hdf5_path)],
            robot_configs=[_make_robot_cfg()],
        )
        imgs = ds[0]["obs_images"]
        assert imgs.dtype == torch.float32
        assert imgs.min() >= 0.0
        assert imgs.max() <= 1.0


# ===================================================================
# Model / forward-pass tests
# ===================================================================
class TestForwardPass:
    def test_obs_encoder_shapes(self):
        """ResNet-18 encoder: (B,2,3,84,84) + (B,2,7) → (B,256)."""
        from opab.model.vision.obs_encoder import ObsEncoder

        enc = ObsEncoder(obs_horizon=2, proprio_dim=7, output_dim=256, pretrained=False)
        imgs = torch.randn(4, 2, 3, 84, 84)
        prop = torch.randn(4, 2, 7)
        out = enc(imgs, prop)
        assert out.shape == (4, 256)

    def test_unet_forward(self):
        """U-Net output matches input shape."""
        from opab.model.diffusion.conditional_unet1d import ConditionalUnet1D

        unet = ConditionalUnet1D(
            action_dim=4, cond_dim=256, morph_dim=32, down_dims=[64, 128, 256]
        )
        x = torch.randn(2, 4, 16)
        t = torch.randint(0, 100, (2,))
        obs = torch.randn(2, 256)
        morph = torch.randn(2, 32)
        out = unet(x, t, obs, morph)
        assert out.shape == (2, 4, 16), f"Expected (2,4,16), got {out.shape}"

    def test_morph_encoder(self):
        """MorphologyEncoder: (B, 46) → (B, 32)."""
        from opab.model.morphology_encoder import MorphologyEncoder

        cfg = _make_policy_cfg()
        enc = MorphologyEncoder(cfg)
        x = torch.randn(3, 46)
        out = enc(x)
        assert out.shape == (3, 32)

    def test_compute_loss(self):
        """compute_loss returns a scalar with grad."""
        from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP

        cfg = _make_policy_cfg()
        policy = MorphologyConditionedDP(cfg)

        batch = {
            "obs_images": torch.randn(2, 2, 3, 84, 84),
            "obs_proprio": torch.randn(2, 2, 7),
            "action": torch.randn(2, 16, 4),
            "morph_vec": torch.randn(2, 46),
        }
        loss = policy.compute_loss(batch)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_predict_action(self):
        """predict_action returns (B, execute_horizon, action_dim)."""
        from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP

        cfg = _make_policy_cfg()
        policy = MorphologyConditionedDP(cfg)
        policy.eval()

        out = policy.predict_action(
            obs_images=torch.randn(1, 2, 3, 84, 84),
            obs_proprio=torch.randn(1, 2, 7),
            morph_vec=torch.randn(1, 46),
        )
        assert out.shape == (1, 8, 4)


# ===================================================================
# Training tests
# ===================================================================
class TestTraining:
    def test_loss_decreases(self, tmp_path):
        """Loss decreases over 10 training steps on a tiny batch."""
        from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP

        cfg = _make_policy_cfg()
        policy = MorphologyConditionedDP(cfg)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        batch = {
            "obs_images": torch.randn(4, 2, 3, 84, 84),
            "obs_proprio": torch.randn(4, 2, 7),
            "action": torch.randn(4, 16, 4),
            "morph_vec": torch.randn(4, 46),
        }

        losses = []
        for _ in range(10):
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_ema_updates(self):
        """EMA shadow weights differ from model weights after update."""
        from opab.model.common.ema import EMAModel
        from opab.policy.morphology_conditioned_dp import MorphologyConditionedDP

        cfg = _make_policy_cfg()
        policy = MorphologyConditionedDP(cfg)
        ema = EMAModel(policy, decay=0.99)

        # Do one gradient step
        batch = {
            "obs_images": torch.randn(2, 2, 3, 84, 84),
            "obs_proprio": torch.randn(2, 2, 7),
            "action": torch.randn(2, 16, 4),
            "morph_vec": torch.randn(2, 46),
        }
        loss = policy.compute_loss(batch)
        loss.backward()
        torch.optim.SGD(policy.parameters(), lr=0.1).step()

        ema.update(policy)

        # Shadow should now differ from model weights
        for key in ema.shadow:
            if policy.state_dict()[key].dtype == torch.float32:
                if not torch.equal(ema.shadow[key], policy.state_dict()[key]):
                    return  # found a difference — pass
        pytest.fail("EMA shadow weights should differ from model after update")


# ===================================================================
# Integration test with real demo data (skip if not available)
# ===================================================================
class TestIntegrationWithDemos:
    @pytest.fixture(autouse=True)
    def _check_demo_data(self):
        demo_path = ROOT / "data" / "demos" / "franka_pick_place.hdf5"
        if not demo_path.exists():
            pytest.skip("Demo data not found — run scripted collection first")
        self.demo_path = demo_path

    def test_load_real_demos(self):
        """Load actual Franka demos and verify dataset works."""
        from opab.dataset.multi_robot_dataset import MultiRobotDataset

        robot_cfg = OmegaConf.load(ROOT / "opab" / "config" / "robot" / "franka.yaml")
        ds = MultiRobotDataset(
            hdf5_paths=[str(self.demo_path)],
            robot_configs=[robot_cfg],
        )
        assert len(ds) > 100  # 5 eps × ~270 steps should be >>100
        sample = ds[0]
        assert sample["obs_images"].shape[1:] == (3, 84, 84)
        assert sample["action"].shape == (16, 4)
