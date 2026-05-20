#!/usr/bin/env python3
"""Test all robots × all tasks and report success rates.

Usage:
    python scripts/test_all_robots_tasks.py
    python scripts/test_all_robots_tasks.py --n_trials 50
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import (
    make_env, ScriptedPickPlace, ScriptedStack, ScriptedReach,
    ScriptedPush, ScriptedPegInsertion,
    SUPPORTED_ROBOTS, SUPPORTED_TASKS,
)
from opab.env.base_env import RobotConfig


ROBOTS = ["franka", "ur5", "widowx", "lite6", "so101"]
TASKS = ["reach", "pick_place", "push", "stack", "peg_insertion"]


def get_policy(task: str, robot: str):
    if task == "reach":
        return ScriptedReach(robot_name=robot)
    elif task == "push":
        return ScriptedPush(robot_name=robot)
    elif task == "stack":
        cube_size = RobotConfig(robot).cube_size
        return ScriptedStack(robot_name=robot, cube_size=cube_size)
    elif task == "peg_insertion":
        return ScriptedPegInsertion(robot_name=robot)
    else:
        return ScriptedPickPlace(robot_name=robot)


def run_episode(env, policy, task):
    """Run one episode, return (success, steps)."""
    obs = env.reset()
    policy.reset()
    terminated = truncated = False
    steps = 0
    info = {}

    while not terminated and not truncated:
        if task == "reach":
            target_pos = env.get_reach_target_pos()
            action = policy.get_action(obs, target_pos)
        elif task == "push":
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            action = policy.get_action(obs, cube_pos, target_pos)
        elif task == "stack":
            cube_pos = env.get_cube_pos()
            cube_b_pos = env.get_cube_b_pos()
            action = policy.get_action(obs, cube_pos, cube_b_pos)
        elif task == "peg_insertion":
            peg_pos = env.get_peg_pos()
            hole_pos = env.get_hole_pos()
            action = policy.get_action(obs, peg_pos, hole_pos)
        else:
            cube_pos = env.get_cube_pos()
            target_pos = env.get_target_pos()
            action = policy.get_action(obs, cube_pos, target_pos)

        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    success = info.get("success", False) if steps > 0 else False
    return success, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    n_trials = args.n_trials
    results = {}  # (robot, task) -> (successes, avg_steps)

    print(f"{'='*70}")
    print(f"  OPAB Robot × Task Success Rate Report  ({n_trials} trials each)")
    print(f"{'='*70}")
    print()

    total_start = time.time()

    for robot in ROBOTS:
        for task in TASKS:
            try:
                env = make_env(robot=robot, seed=0, max_episode_steps=args.max_steps, task=task)
                policy = get_policy(task, robot)

                successes = 0
                total_steps = 0

                for ep in range(n_trials):
                    env.reset(seed=ep)  # seed the reset
                    obs = env.reset(seed=ep)
                    policy.reset()

                    success, steps = run_episode(env, policy, task)
                    successes += int(success)
                    total_steps += steps

                avg_steps = total_steps / n_trials
                rate = successes / n_trials * 100
                results[(robot, task)] = (successes, n_trials, rate, avg_steps)
                status = "✓" if rate >= 90 else ("~" if rate >= 50 else "✗")
                print(f"  {status} {robot:8s} | {task:14s} | {successes:2d}/{n_trials} ({rate:5.1f}%) | avg {avg_steps:.0f} steps")

            except Exception as e:
                results[(robot, task)] = (0, n_trials, 0.0, 0)
                print(f"  ✗ {robot:8s} | {task:14s} | ERROR: {e}")

        print()

    elapsed = time.time() - total_start

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE (% success)")
    print(f"{'='*70}")
    header = f"{'Robot':<10}" + "".join(f"{t:<15}" for t in TASKS)
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for robot in ROBOTS:
        row = f"{robot:<10}"
        for task in TASKS:
            if (robot, task) in results:
                s, n, rate, _ = results[(robot, task)]
                cell = f"{rate:.0f}%"
                if rate >= 90:
                    cell += " ✅"
                elif rate >= 50:
                    cell += " ⚠️"
                else:
                    cell += " ❌"
            else:
                cell = "N/A"
            row += f"{cell:<15}"
        print(f"  {row}")

    # Overall stats
    total_success = sum(r[0] for r in results.values())
    total_trials = sum(r[1] for r in results.values())
    overall_rate = total_success / total_trials * 100 if total_trials > 0 else 0

    perfect = sum(1 for r in results.values() if r[2] >= 90)
    failing = sum(1 for r in results.values() if r[2] < 50)

    print(f"\n  {'─'*50}")
    print(f"  Overall: {total_success}/{total_trials} ({overall_rate:.1f}%)")
    print(f"  Passing (≥90%): {perfect}/25 combos")
    print(f"  Failing (<50%): {failing}/25 combos")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
