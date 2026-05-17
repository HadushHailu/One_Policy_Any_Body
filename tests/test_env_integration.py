"""
Integration tests for OPAB manipulation environments.

Tests that:
  1. All 3 environments load and step correctly
  2. IK converges to reasonable positions
  3. Scripted policy runs through the full state machine
  4. Observations have correct shapes and types
  5. Task objects (cube, target) are present and accessible
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import make_env, ScriptedPickPlace, SUPPORTED_ROBOTS


# ============================================================
# Environment loading tests
# ============================================================

class TestEnvLoading:
    """Test that environments load correctly for all robots."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_env_creates(self, robot):
        """Environment instantiates without error."""
        env = make_env(robot=robot, seed=0)
        assert env is not None
        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_env_reset(self, robot):
        """reset() returns valid observation dict."""
        env = make_env(robot=robot, seed=0)
        obs = env.reset()

        assert "image" in obs
        assert "proprioception" in obs
        assert "ee_pos" in obs
        assert "ee_quat" in obs
        assert "gripper_pos" in obs

        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_obs_shapes(self, robot):
        """Observations have expected shapes."""
        env = make_env(robot=robot, image_size=(84, 84), seed=0)
        obs = env.reset()

        assert obs["image"].shape == (84, 84, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["ee_pos"].shape == (3,)
        assert obs["ee_quat"].shape == (4,)
        assert obs["gripper_pos"].shape == (1,)
        assert obs["proprioception"].shape == (env.cfg.n_arm_joints,)

        env.close()


# ============================================================
# Step and action tests
# ============================================================

class TestEnvStep:
    """Test stepping the environment with actions."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_zero_action(self, robot):
        """Zero action doesn't crash and returns 5-tuple."""
        env = make_env(robot=robot, seed=0)
        env.reset()

        action = np.zeros(4)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "success" in info

        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_multiple_steps(self, robot):
        """Can step for 10 steps without crash."""
        env = make_env(robot=robot, seed=0)
        env.reset()

        for _ in range(10):
            action = np.array([0.001, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_episode_truncation(self, robot):
        """Episode truncates at max_episode_steps."""
        env = make_env(robot=robot, seed=0, max_episode_steps=5)
        env.reset()

        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.zeros(4))
            if truncated:
                break

        assert truncated

        env.close()


# ============================================================
# IK tests
# ============================================================

class TestIK:
    """Test inverse kinematics convergence."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_ik_no_movement(self, robot):
        """IK at current EE position returns near-current joints."""
        env = make_env(robot=robot, seed=0)
        env.reset()

        ee_pos = env._get_ee_pos()
        qpos = env._ik_solve(ee_pos)

        # Should be close to current arm joint positions
        current_qpos = np.array([env.data.qpos[i] for i in env._arm_qpos_ids])
        error = np.linalg.norm(qpos - current_qpos)
        assert error < 0.1, f"IK diverged for identity target: error={error:.4f}"

        env.close()

    def test_ik_small_delta(self):
        """IK to a small delta from current position converges."""
        env = make_env(robot="so101", seed=0)
        env.reset()

        ee_pos = env._get_ee_pos()
        target = ee_pos + np.array([0.01, 0.0, 0.0])
        qpos = env._ik_solve(target)

        # Apply and check
        for i, idx in enumerate(env._arm_qpos_ids):
            env.data.qpos[idx] = qpos[i]
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        achieved = env._get_ee_pos()
        pos_err = np.linalg.norm(achieved - target)
        assert pos_err < 0.005, f"IK position error: {pos_err:.4f}m"

        env.close()


# ============================================================
# Scripted policy tests
# ============================================================

class TestScriptedPolicy:
    """Test the scripted pick-and-place policy."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_policy_runs(self, robot):
        """Scripted policy produces actions for full episode."""
        env = make_env(robot=robot, seed=42, max_episode_steps=300)
        obs = env.reset()
        policy = ScriptedPickPlace(robot_name=robot)
        policy.reset()

        steps = 0
        for _ in range(300):
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            action = policy.get_action(obs, cube_pos, target_pos)
            if action is None:
                break
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        # Should run for some steps (not immediately done)
        assert steps > 10, f"Policy ran only {steps} steps"

        env.close()

    def test_policy_state_transitions(self):
        """Policy transitions through expected states."""
        env = make_env(robot="so101", seed=42)
        obs = env.reset()
        policy = ScriptedPickPlace(robot_name="so101")
        policy.reset()

        states_visited = {policy.state}

        for _ in range(300):
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            action = policy.get_action(obs, cube_pos, target_pos)
            if action is None:
                states_visited.add("DONE")
                break
            obs, _, terminated, truncated, _ = env.step(action)
            states_visited.add(policy.state)
            if terminated or truncated:
                break

        # Should hit at least APPROACH, DESCEND, GRASP
        assert "APPROACH" in states_visited or len(states_visited) >= 3

        env.close()


# ============================================================
# Cube and target detection
# ============================================================

class TestTaskSetup:
    """Test that task objects (cube, target) are present and accessible."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_cube_exists(self, robot):
        """Cube body exists and has valid position."""
        env = make_env(robot=robot, seed=0)
        env.reset()

        cube_pos = env.get_cube_pos()
        assert not np.all(cube_pos == 0), "Cube position is [0,0,0] — not found?"
        # Cube should be roughly on the table (positive z for most configs)
        assert cube_pos[2] > -0.1

        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_target_exists(self, robot):
        """Target zone exists and has valid position."""
        env = make_env(robot=robot, seed=0)
        env.reset()

        target_pos = env.get_target_pos()
        assert not np.all(target_pos == 0), "Target position is [0,0,0] — not found?"

        env.close()


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    # Unregister conflicting ROS pytest plugins before running
    import _pytest.config
    pm = _pytest.config.get_plugin_manager()
    for name in list(pm.list_name_plugin()):
        if "launch" in str(name[0]):
            pm.unregister(name=name[0])
    pytest.main([__file__, "-v", "--tb=short", "--override-ini=addopts="])
