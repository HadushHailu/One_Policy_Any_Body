"""
Week 4 tests — Task conditioning, temporal ensemble, multi-task training.

Tests:
  1. TaskEncoder produces correct output shapes
  2. Policy forward pass works with task_id
  3. Policy backward-compatible without task_id
  4. Dataset returns correct task_ids from filenames
  5. TemporalEnsemble smooths overlapping chunks
  6. TemporalEnsemble decay weighting is correct
  7. Multi-task training step converges (loss decreases)
  8. Eval script handles task_id argument
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.policy.morphology_conditioned_dp import (
    MorphologyConditionedDP,
    TaskEncoder,
)
from opab.policy.temporal_ensemble import TemporalEnsemble


# ============================================================
# Helper: build policy config
# ============================================================

def _make_cfg():
    """Minimal OmegaConf-like config for policy construction."""
    from omegaconf import OmegaConf

    return OmegaConf.load(
        str(Path(__file__).resolve().parents[1] / "configs/policy/morphology_dp.yaml")
    )


# ============================================================
# TaskEncoder Tests
# ============================================================

class TestTaskEncoder:
    """Test the task embedding module."""

    def test_output_shape(self):
        """TaskEncoder produces (B, embed_dim) output."""
        enc = TaskEncoder(n_tasks=3, embed_dim=16)
        task_ids = torch.tensor([0, 1, 2, 0])
        out = enc(task_ids)
        assert out.shape == (4, 16)

    def test_different_tasks_different_embeddings(self):
        """Different task IDs produce different embeddings."""
        enc = TaskEncoder(n_tasks=2, embed_dim=32)
        out0 = enc(torch.tensor([0]))
        out1 = enc(torch.tensor([1]))
        assert not torch.allclose(out0, out1)

    def test_same_task_same_embedding(self):
        """Same task ID always produces same embedding."""
        enc = TaskEncoder(n_tasks=2, embed_dim=32)
        out_a = enc(torch.tensor([0, 0, 0]))
        assert torch.allclose(out_a[0], out_a[1])
        assert torch.allclose(out_a[1], out_a[2])

    def test_gradient_flows(self):
        """Embedding is trainable — gradients flow through."""
        enc = TaskEncoder(n_tasks=2, embed_dim=8)
        out = enc(torch.tensor([0, 1]))
        loss = out.sum()
        loss.backward()
        assert enc.embedding.weight.grad is not None
        assert enc.embedding.weight.grad.abs().sum() > 0


# ============================================================
# Policy with Task Conditioning Tests
# ============================================================

class TestPolicyTaskConditioning:
    """Test policy forward/backward with task_id."""

    @pytest.fixture
    def policy(self):
        cfg = _make_cfg()
        return MorphologyConditionedDP(cfg).eval()

    @pytest.fixture
    def batch(self):
        return {
            "obs_images": torch.randn(2, 2, 3, 84, 84),
            "obs_proprio": torch.randn(2, 2, 7),
            "action": torch.randn(2, 16, 4),
            "morph_vec": torch.randn(2, 46),
            "task_id": torch.tensor([0, 1]),
        }

    def test_compute_loss_with_task_id(self, policy, batch):
        """compute_loss works when task_id is in batch."""
        policy.train()
        loss = policy.compute_loss(batch)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # positive loss

    def test_compute_loss_without_task_id(self, policy, batch):
        """Backward compatible: compute_loss works without task_id in batch."""
        policy.train()
        del batch["task_id"]
        loss = policy.compute_loss(batch)
        assert loss.ndim == 0

    def test_predict_action_with_task_id(self, policy, batch):
        """predict_action returns correct shape with task_id."""
        actions = policy.predict_action(
            batch["obs_images"],
            batch["obs_proprio"],
            batch["morph_vec"],
            task_id=batch["task_id"],
        )
        # (B, execute_horizon, action_dim)
        assert actions.shape == (2, 8, 4)

    def test_predict_action_without_task_id(self, policy, batch):
        """predict_action works without task_id (defaults to zeros)."""
        actions = policy.predict_action(
            batch["obs_images"],
            batch["obs_proprio"],
            batch["morph_vec"],
        )
        assert actions.shape == (2, 8, 4)

    def test_different_tasks_different_outputs(self, policy, batch):
        """Different task_ids should produce different denoising paths."""
        torch.manual_seed(42)
        actions_task0 = policy.predict_action(
            batch["obs_images"],
            batch["obs_proprio"],
            batch["morph_vec"],
            task_id=torch.tensor([0, 0]),
        )
        torch.manual_seed(42)
        actions_task1 = policy.predict_action(
            batch["obs_images"],
            batch["obs_proprio"],
            batch["morph_vec"],
            task_id=torch.tensor([1, 1]),
        )
        # They start from same noise but different conditioning → different result
        assert not torch.allclose(actions_task0, actions_task1, atol=1e-3)

    def test_morph_cond_dimension(self, policy, batch):
        """Internal _get_morph_cond produces correct dimension (32 morph + 32 task = 64)."""
        morph_cond = policy._get_morph_cond(batch["morph_vec"], batch["task_id"])
        assert morph_cond.shape == (2, 64)

    def test_morph_cond_without_task(self, policy, batch):
        """Without task_id, morph_cond still has 64 dims (zeros for task portion)."""
        morph_cond = policy._get_morph_cond(batch["morph_vec"], None)
        assert morph_cond.shape == (2, 64)
        # Last 32 dims should be zeros
        assert torch.allclose(morph_cond[:, 32:], torch.zeros(2, 32))


# ============================================================
# Dataset Task ID Tests
# ============================================================

class TestDatasetTaskId:
    """Test that dataset correctly assigns task IDs from filenames."""

    def test_infer_task_id_pick_place(self):
        from opab.dataset.multi_robot_dataset import MultiRobotDataset
        assert MultiRobotDataset._infer_task_id(Path("franka_pick_place.hdf5")) == 0

    def test_infer_task_id_stack(self):
        from opab.dataset.multi_robot_dataset import MultiRobotDataset
        assert MultiRobotDataset._infer_task_id(Path("ur5_stack.hdf5")) == 1

    def test_infer_task_id_default(self):
        """Unknown filenames default to task_id=0."""
        from opab.dataset.multi_robot_dataset import MultiRobotDataset
        assert MultiRobotDataset._infer_task_id(Path("some_robot_unknown.hdf5")) == 0

    @pytest.mark.skipif(
        not Path("/home/hadush/dev/One_Policy_Any_Body/data/demos/franka_pick_place.hdf5").exists(),
        reason="Demo data not available",
    )
    def test_dataset_returns_task_id(self):
        """Full dataset __getitem__ includes task_id."""
        from omegaconf import OmegaConf
        from opab.dataset.multi_robot_dataset import MultiRobotDataset

        paths = [
            Path("/home/hadush/dev/One_Policy_Any_Body/data/demos/franka_pick_place.hdf5"),
            Path("/home/hadush/dev/One_Policy_Any_Body/data/demos/franka_stack.hdf5"),
        ]
        cfgs = [OmegaConf.load("/home/hadush/dev/One_Policy_Any_Body/configs/robot/franka.yaml")] * 2

        ds = MultiRobotDataset(paths, cfgs)
        sample = ds[0]

        assert "task_id" in sample
        assert sample["task_id"].dtype == torch.int64
        assert sample["task_id"].item() in [0, 1]


# ============================================================
# Temporal Ensemble Tests
# ============================================================

class TestTemporalEnsemble:
    """Test temporal action ensemble smoothing."""

    def test_single_chunk_returns_exact_action(self):
        """With only one chunk, get_action returns the exact predicted action."""
        ens = TemporalEnsemble(action_dim=4, decay=0.01, max_chunks=5)
        chunk = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        ens.add_chunk(global_step=0, action_chunk=chunk)

        action = ens.get_action(global_step=0)
        np.testing.assert_allclose(action, [1.0, 2.0, 3.0, 4.0])

        action = ens.get_action(global_step=1)
        np.testing.assert_allclose(action, [5.0, 6.0, 7.0, 8.0])

    def test_overlapping_chunks_average(self):
        """Overlapping chunks are weighted averaged."""
        ens = TemporalEnsemble(action_dim=2, decay=0.0, max_chunks=5)

        # Two chunks that overlap at step 1
        chunk_a = np.array([[1.0, 0.0], [2.0, 0.0]])  # starts at step 0
        chunk_b = np.array([[0.0, 1.0], [0.0, 2.0]])  # starts at step 1

        ens.add_chunk(global_step=0, action_chunk=chunk_a)
        ens.add_chunk(global_step=1, action_chunk=chunk_b)

        # At step 1: chunk_a[1] = [2,0], chunk_b[0] = [0,1]
        # With decay=0, all weights = exp(0)=1 → equal average
        action = ens.get_action(global_step=1)
        np.testing.assert_allclose(action, [1.0, 0.5], atol=1e-6)

    def test_decay_weighting(self):
        """Newer (lower index) actions get higher weight with decay > 0."""
        ens = TemporalEnsemble(action_dim=1, decay=1.0, max_chunks=5)

        # Chunk A starts at step 0, has 3 actions
        chunk_a = np.array([[10.0], [10.0], [10.0]])
        # Chunk B starts at step 1, has 2 actions
        chunk_b = np.array([[0.0], [0.0]])

        ens.add_chunk(global_step=0, action_chunk=chunk_a)
        ens.add_chunk(global_step=1, action_chunk=chunk_b)

        # At step 1: chunk_a idx=1 (weight=exp(-1*1)≈0.368), chunk_b idx=0 (weight=exp(-1*0)=1)
        action = ens.get_action(global_step=1)
        # Expected: (0.368*10 + 1*0) / (0.368 + 1) ≈ 2.689
        import math
        w_a = math.exp(-1.0)
        w_b = 1.0
        expected = (w_a * 10.0 + w_b * 0.0) / (w_a + w_b)
        np.testing.assert_allclose(action, [expected], atol=1e-4)

    def test_reset_clears_buffer(self):
        """reset() empties the buffer."""
        ens = TemporalEnsemble(action_dim=2, decay=0.01)
        ens.add_chunk(0, np.ones((3, 2)))
        ens.reset()
        assert not ens.has_actions
        with pytest.raises(ValueError):
            ens.get_action(0)

    def test_max_chunks_eviction(self):
        """Buffer respects max_chunks limit."""
        ens = TemporalEnsemble(action_dim=1, decay=0.01, max_chunks=2)
        ens.add_chunk(0, np.ones((5, 1)))
        ens.add_chunk(1, np.ones((5, 1)) * 2)
        ens.add_chunk(2, np.ones((5, 1)) * 3)  # should evict first chunk

        assert len(ens.buffer) == 2
        # Step 0 should now have no valid chunks (first was evicted)
        with pytest.raises(ValueError):
            ens.get_action(0)

    def test_no_action_available_raises(self):
        """get_action raises ValueError when no chunk covers the timestep."""
        ens = TemporalEnsemble(action_dim=2, decay=0.01)
        ens.add_chunk(5, np.ones((3, 2)))
        with pytest.raises(ValueError, match="No predicted actions"):
            ens.get_action(0)


# ============================================================
# Integration: Training Step with Task Conditioning
# ============================================================

class TestTrainingIntegration:
    """Verify training converges with task conditioning."""

    def test_loss_decreases_over_steps(self):
        """5 gradient steps should reduce loss."""
        cfg = _make_cfg()
        policy = MorphologyConditionedDP(cfg).train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        batch = {
            "obs_images": torch.randn(4, 2, 3, 84, 84),
            "obs_proprio": torch.randn(4, 2, 7),
            "action": torch.randn(4, 16, 4),
            "morph_vec": torch.randn(4, 46),
            "task_id": torch.tensor([0, 1, 0, 1]),
        }

        losses = []
        for _ in range(5):
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (first > last) with fixed data
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_task_gradient_nonzero(self):
        """Task encoder receives gradients during training."""
        cfg = _make_cfg()
        policy = MorphologyConditionedDP(cfg).train()

        batch = {
            "obs_images": torch.randn(2, 2, 3, 84, 84),
            "obs_proprio": torch.randn(2, 2, 7),
            "action": torch.randn(2, 16, 4),
            "morph_vec": torch.randn(2, 46),
            "task_id": torch.tensor([0, 1]),
        }

        loss = policy.compute_loss(batch)
        loss.backward()

        grad = policy.task_encoder.embedding.weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0, "Task encoder got zero gradients"
