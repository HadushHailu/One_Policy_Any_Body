"""Rollout runner for evaluating policies across robots and tasks."""
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path


class RolloutRunner:
    """Runs evaluation rollouts across multiple robot x task combinations.
    
    Usage:
        runner = RolloutRunner(robots=["franka", "lite6"], tasks=["reach", "push"])
        results = runner.run(policy_fn, n_episodes=10)
    """
    
    ROBOTS = ["franka", "ur5", "widowx", "lite6", "so101"]
    TASKS = [
        "reach", "pick_place", "push", "stack", "peg_insertion",
        "drawer_open", "turn_faucet", "button_press", "door_open",
        "lever_pull", "sweep",
    ]
    
    def __init__(
        self,
        robots: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        max_steps: int = 300,
        render: bool = False,
        render_size: tuple = (480, 480),
        video_dir: Optional[Path] = None,
    ):
        self.robots = robots or self.ROBOTS
        self.tasks = tasks or self.TASKS
        self.max_steps = max_steps
        self.render = render
        self.render_size = render_size
        self.video_dir = Path(video_dir) if video_dir else None
    
    def run(
        self,
        policy_fn: Callable,
        n_episodes: int = 10,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Run evaluation across all robot x task combos.
        
        Args:
            policy_fn: Callable(obs, robot, task) -> action (np.ndarray shape (4,))
            n_episodes: Number of episodes per robot x task
            seeds: Optional list of seeds (length n_episodes)
        
        Returns:
            List of result dicts with robot, task, success, steps, total_reward.
        """
        from opab.env.base_env import PickPlaceEnv
        
        if seeds is None:
            seeds = list(range(n_episodes))
        
        results = []
        
        for robot in self.robots:
            for task in self.tasks:
                env = PickPlaceEnv(robot=robot, task=task)
                
                for seed in seeds:
                    obs = env.reset(seed=seed)
                    total_reward = 0.0
                    success = False
                    
                    for step in range(self.max_steps):
                        action = policy_fn(obs, robot, task)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        
                        if info.get("success", False):
                            success = True
                            break
                        if terminated or truncated:
                            break
                    
                    results.append({
                        "robot": robot,
                        "task": task,
                        "seed": seed,
                        "success": success,
                        "steps": step + 1,
                        "total_reward": total_reward,
                    })
                
                env.close()
        
        return results
    
    def run_scripted(self, n_episodes: int = 3) -> List[Dict]:
        """Run evaluation using scripted policies (for env validation)."""
        from opab.env.scripted_policies import (
            ScriptedReach, ScriptedPickPlace, ScriptedPush,
            ScriptedStack, ScriptedPegInsertion,
        )
        from opab.env.base_env import PickPlaceEnv, RobotConfig
        
        SCRIPTED_TASKS = ["reach", "pick_place", "push", "stack", "peg_insertion"]
        results = []
        tasks_to_run = [t for t in self.tasks if t in SCRIPTED_TASKS]
        
        for robot in self.robots:
            for task in tasks_to_run:
                env = PickPlaceEnv(robot=robot, task=task)
                
                for seed in range(n_episodes):
                    obs = env.reset(seed=seed)
                    policy = self._get_scripted_policy(task, robot)
                    policy.reset()
                    success = False
                    
                    for step in range(self.max_steps):
                        action = self._get_scripted_action(policy, task, env, obs)
                        if action is None:
                            break
                        obs, reward, terminated, truncated, info = env.step(action)
                        if info.get("success", False):
                            success = True
                            break
                    
                    results.append({
                        "robot": robot,
                        "task": task,
                        "seed": seed,
                        "success": success,
                        "steps": step + 1,
                    })
                
                env.close()
        
        return results
    
    @staticmethod
    def _get_scripted_policy(task, robot):
        from opab.env.scripted_policies import (
            ScriptedReach, ScriptedPickPlace, ScriptedPush,
            ScriptedStack, ScriptedPegInsertion,
        )
        from opab.env.base_env import RobotConfig
        
        if task == "reach":
            return ScriptedReach(robot_name=robot)
        elif task == "push":
            return ScriptedPush(robot_name=robot)
        elif task == "stack":
            return ScriptedStack(robot_name=robot, cube_size=RobotConfig(robot).cube_size)
        elif task == "peg_insertion":
            return ScriptedPegInsertion(robot_name=robot)
        else:
            return ScriptedPickPlace(robot_name=robot)
    
    @staticmethod
    def _get_scripted_action(policy, task, env, obs):
        if task == "reach":
            return policy.get_action(obs, env.get_reach_target_pos())
        elif task == "push":
            return policy.get_action(obs, env.get_cube_pos(), env.get_target_pos())
        elif task == "stack":
            return policy.get_action(obs, env.get_cube_pos(), env.get_cube_b_pos())
        elif task == "peg_insertion":
            return policy.get_action(obs, env.get_peg_pos(), env.get_hole_pos())
        else:
            return policy.get_action(obs, env.get_cube_pos(), env.get_target_pos())
