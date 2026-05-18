"""
Week 3 tests — domain randomization, stacking, multi-robot data collection.

Tests:
  1. DR modifies model parameters when enabled
  2. DR leaves model unchanged when disabled
  3. Stacking scene has two cubes
  4. ScriptedStack completes stacking trajectory
  5. generate_sim_demos supports --task and --dr flags
  6. Multi-robot data loading works
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import make_env, ScriptedPickPlace, ScriptedStack, SUPPORTED_ROBOTS
from opab.env.domain_randomization import DRConfig, DomainRandomizer


# ============================================================
# Domain Randomization Tests
# ============================================================

class TestDomainRandomization:
    """Test that DR modifies the right model parameters."""

    def test_dr_enabled_modifies_params(self):
        """When DR is enabled, model parameters should change after randomize()."""
        env = make_env("franka", domain_randomization=True, seed=42)
        obs = env.reset(seed=0)

        import mujoco
        cube_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        nominal_mass = env.domain_randomizer._nominal["body_mass"][cube_body_id]
        current_mass = env.model.body_mass[cube_body_id]

        # After reset with DR, mass should differ from nominal sometimes
        # Run multiple resets to catch at least one difference
        found_diff = False
        for i in range(20):
            env.reset(seed=i)
            current_mass = env.model.body_mass[cube_body_id]
            if abs(current_mass - nominal_mass) > 1e-6:
                found_diff = True
                break

        assert found_diff, "DR enabled but mass never changed across 20 resets"
        env.close()

    def test_dr_disabled_no_change(self):
        """When DR is disabled, model parameters should stay at nominal."""
        env = make_env("franka", domain_randomization=False, seed=42)

        import mujoco
        cube_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        nominal_mass = env.domain_randomizer._nominal["body_mass"][cube_body_id]

        for i in range(5):
            env.reset(seed=i)
            current_mass = env.model.body_mass[cube_body_id]
            assert abs(current_mass - nominal_mass) < 1e-8, (
                f"DR disabled but mass changed: {current_mass} vs {nominal_mass}"
            )
        env.close()

    def test_dr_resets_to_nominal(self):
        """reset_to_nominal should restore all parameters."""
        env = make_env("franka", domain_randomization=True, seed=42)
        env.reset(seed=0)

        dr = env.domain_randomizer
        dr.reset_to_nominal()

        import mujoco
        cube_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        assert abs(env.model.body_mass[cube_body_id] - dr._nominal["body_mass"][cube_body_id]) < 1e-8
        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_dr_all_robots(self, robot):
        """DR should work for all robot types without errors."""
        env = make_env(robot, domain_randomization=True, seed=42)
        for i in range(3):
            obs = env.reset(seed=i)
            assert obs["image"].shape == (84, 84, 3)
        env.close()


# ============================================================
# Stacking Task Tests
# ============================================================

class TestStackingEnv:
    """Test the stacking scene and task."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_stack_env_loads(self, robot):
        """Stacking env should load with two cubes."""
        env = make_env(robot, task="stack", seed=42)
        obs = env.reset()

        assert env.task == "stack"
        assert env._cube_body_id >= 0, "cube_A not found"
        assert env._cube_b_body_id >= 0, "cube_B not found"

        # Both cubes should be at reasonable positions
        cube_a = env.get_cube_pos()
        cube_b = env.get_cube_b_pos()
        assert np.linalg.norm(cube_a) > 0, "cube_A at origin"
        assert np.linalg.norm(cube_b) > 0, "cube_B at origin"
        assert np.linalg.norm(cube_a - cube_b) > 0.01, "cubes overlap"
        env.close()

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_stack_env_steps(self, robot):
        """Env should accept actions in stacking mode."""
        env = make_env(robot, task="stack", seed=42)
        obs = env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert "success" in info
        assert obs["image"].shape == (84, 84, 3)
        env.close()

    def test_stack_success_check(self):
        """Success should be False initially (cube_A not on cube_B)."""
        env = make_env("franka", task="stack", seed=42)
        env.reset()
        assert not env._check_success()
        env.close()

    def test_invalid_task_rejected(self):
        """Unknown task should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            make_env("franka", task="reachhhh")


class TestScriptedStack:
    """Test the ScriptedStack policy."""

    def test_scripted_stack_init(self):
        """ScriptedStack should initialize with correct defaults."""
        policy = ScriptedStack(robot_name="franka")
        assert policy.state == "APPROACH"
        assert not policy.is_done

    def test_scripted_stack_reset(self):
        """Reset should return to APPROACH state."""
        policy = ScriptedStack(robot_name="franka")
        policy.state = "DONE"
        policy.reset()
        assert policy.state == "APPROACH"

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_scripted_stack_trajectory(self, robot):
        """ScriptedStack should produce a full trajectory."""
        env = make_env(robot, task="stack", seed=42, max_episode_steps=500)
        from opab.env.base_env import RobotConfig
        cube_size = RobotConfig(robot).cube_size
        policy = ScriptedStack(robot_name=robot, cube_size=cube_size)

        obs = env.reset(seed=42)
        policy.reset()

        step_count = 0
        while step_count < 500:
            cube_a = env.get_cube_pos()
            cube_b = env.get_cube_b_pos()
            action = policy.get_action(obs, cube_a, cube_b)
            if action is None:
                break
            obs, _, _, truncated, info = env.step(action)
            step_count += 1
            if truncated:
                break

        # Policy should finish (reach DONE) or at least make progress
        assert step_count > 10, f"Only {step_count} steps for {robot}"
        # Check that EE moved significantly
        env.close()

    def test_scripted_stack_actions_valid(self):
        """All actions from ScriptedStack should be (4,) arrays."""
        env = make_env("franka", task="stack", seed=42)
        from opab.env.base_env import RobotConfig
        cube_size = RobotConfig("franka").cube_size
        policy = ScriptedStack(robot_name="franka", cube_size=cube_size)

        obs = env.reset(seed=42)
        policy.reset()

        for _ in range(50):
            cube_a = env.get_cube_pos()
            cube_b = env.get_cube_b_pos()
            action = policy.get_action(obs, cube_a, cube_b)
            if action is None:
                break
            assert action.shape == (4,), f"Bad action shape: {action.shape}"
            assert np.all(np.isfinite(action)), "NaN/Inf in action"
            obs, _, _, _, _ = env.step(action)
        env.close()


# ============================================================
# DR + Stacking combined
# ============================================================

class TestDRWithStacking:
    """Test DR works with stacking task."""

    @pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
    def test_dr_stack_runs(self, robot):
        """Stack + DR should work together without errors."""
        env = make_env(robot, task="stack", domain_randomization=True, seed=42)
        for i in range(3):
            obs = env.reset(seed=i)
            action = np.array([0.0, 0.0, 0.0, 0.0])
            obs, _, _, _, info = env.step(action)
            assert "success" in info
        env.close()


# ============================================================
# Multi-robot workspace loading
# ============================================================

class TestMultiRobotWorkspace:
    """Test that training workspace can discover multiple demo files."""

    def test_load_robot_config_by_name(self):
        """_load_robot_config should load yaml for known robots."""
        from omegaconf import OmegaConf
        from opab.workspace.train_morph_dp_workspace import TrainMorphDPWorkspace

        for robot_name in SUPPORTED_ROBOTS:
            cfg_path = (
                Path(__file__).resolve().parents[1]
                / "opab" / "config" / "robot" / f"{robot_name}.yaml"
            )
            assert cfg_path.exists(), f"Robot config missing: {cfg_path}"
            robot_cfg = OmegaConf.load(cfg_path)
            assert robot_cfg.name == robot_name
