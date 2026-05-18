#!/usr/bin/env python3
"""Generate simulated pick-and-place demonstrations with scripted policies.

Collects expert demonstrations across multiple robot embodiments and saves
them in HDF5 format compatible with downstream policy training.

Usage:
    python scripts/generate_sim_demos.py --robot so101 --n_demos 50
    python scripts/generate_sim_demos.py --robot franka --n_demos 50
    python scripts/generate_sim_demos.py --robot ur5 --n_demos 50
    python scripts/generate_sim_demos.py --all --n_demos 50
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np

# Make sure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import make_env, ScriptedPickPlace, ScriptedStack, SUPPORTED_ROBOTS, SUPPORTED_TASKS


def collect_demos(
    robot: str,
    n_demos: int,
    output_dir: Path,
    seed: int = 0,
    max_episode_steps: int = 300,
    verbose: bool = True,
    domain_randomization: bool = False,
    task: str = "pick_place",
):
    """
    Collect demonstrations using the scripted pick-and-place policy.

    Args:
        robot: Robot name ('franka', 'ur5', 'so101')
        n_demos: Number of episodes to collect
        output_dir: Where to save HDF5 file
        seed: Base random seed
        max_episode_steps: Max steps per episode
        verbose: Print progress

    Returns:
        Path to saved HDF5 file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        robot=robot, seed=seed, max_episode_steps=max_episode_steps,
        domain_randomization=domain_randomization, task=task,
    )

    if task == "stack":
        from opab.env.base_env import RobotConfig
        cube_size = RobotConfig(robot).cube_size
        policy = ScriptedStack(robot_name=robot, cube_size=cube_size)
    else:
        policy = ScriptedPickPlace(robot_name=robot)

    # Storage for all episodes
    all_episodes = []
    successes = 0

    t_start = time.time()

    for ep in range(n_demos):
        obs = env.reset(seed=seed + ep)
        policy.reset()

        episode_data = {
            "actions": [],
            "ee_pos": [],
            "ee_quat": [],
            "proprioception": [],
            "gripper_pos": [],
            "images": [],
        }

        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            # Get action from scripted policy
            cube_pos = env.get_cube_pos()
            if task == "stack":
                cube_b_pos = env.get_cube_b_pos()
                action = policy.get_action(obs, cube_pos, cube_b_pos)
            else:
                target_pos = env.get_target_pos()
                action = policy.get_action(obs, cube_pos, target_pos)

            if action is None:
                # Policy says done
                break

            # Record pre-step observation + action
            episode_data["actions"].append(action.copy())
            episode_data["ee_pos"].append(obs["ee_pos"].copy())
            episode_data["ee_quat"].append(obs["ee_quat"].copy())
            episode_data["proprioception"].append(obs["proprioception"].copy())
            episode_data["gripper_pos"].append(obs["gripper_pos"].copy())
            episode_data["images"].append(obs["image"].copy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        success = info.get("success", False) if step_count > 0 else False
        successes += int(success)

        # Convert lists to numpy arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])

        episode_data["success"] = success
        episode_data["n_steps"] = step_count
        all_episodes.append(episode_data)

        if verbose:
            elapsed = time.time() - t_start
            rate = (ep + 1) / elapsed
            print(
                f"  [{ep+1:4d}/{n_demos}] steps={step_count:3d} "
                f"success={success} "
                f"({rate:.1f} ep/s, total_success={successes}/{ep+1})"
            )

    # Save to HDF5
    task_name = "stack" if task == "stack" else "pick_place"
    save_path = output_dir / f"{robot}_{task_name}.hdf5"
    _save_hdf5(all_episodes, save_path, robot)

    elapsed = time.time() - t_start
    success_rate = successes / n_demos * 100
    print(f"\nDone: {n_demos} episodes in {elapsed:.1f}s")
    print(f"Success rate: {success_rate:.1f}% ({successes}/{n_demos})")
    print(f"Saved to: {save_path}")

    env.close()
    return save_path


def _save_hdf5(episodes: list[dict], path: Path, robot: str):
    """
    Save demonstrations in HDF5 format.

    Structure:
        /attrs: robot, n_episodes, total_steps
        /episode_0/
            actions: (T, 4) float32
            ee_pos: (T, 3) float32
            ee_quat: (T, 4) float32
            proprioception: (T, n_joints) float32
            gripper_pos: (T, 1) float32
            images: (T, H, W, 3) uint8
            attrs: success, n_steps
        /episode_1/
            ...
    """
    with h5py.File(path, "w") as f:
        # Global metadata
        f.attrs["robot"] = robot
        f.attrs["n_episodes"] = len(episodes)
        f.attrs["total_steps"] = sum(ep["n_steps"] for ep in episodes)
        f.attrs["success_rate"] = (
            sum(ep["success"] for ep in episodes) / len(episodes)
        )
        f.attrs["domain_randomization"] = True  # flag for downstream

        for i, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{i}")
            grp.attrs["success"] = ep["success"]
            grp.attrs["n_steps"] = ep["n_steps"]

            # Store arrays with compression
            for key in ("actions", "ee_pos", "ee_quat", "proprioception", "gripper_pos"):
                if len(ep[key]) > 0:
                    grp.create_dataset(
                        key, data=ep[key].astype(np.float32),
                        compression="gzip", compression_opts=4
                    )

            # Images: store with heavier compression
            if len(ep["images"]) > 0:
                grp.create_dataset(
                    "images", data=ep["images"],
                    compression="gzip", compression_opts=6,
                    chunks=(1, *ep["images"].shape[1:]),
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate sim pick-and-place demonstrations"
    )
    parser.add_argument(
        "--robot", choices=list(SUPPORTED_ROBOTS),
        help="Robot to generate demos for"
    )
    parser.add_argument("--all", action="store_true", help="Generate for all robots")
    parser.add_argument("--n_demos", type=int, default=50, help="Episodes per robot")
    parser.add_argument("--task", choices=list(SUPPORTED_TASKS), default="pick_place",
                        help="Task type: pick_place or stack")
    parser.add_argument("--out", type=str, default="data/demos", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps/episode")
    parser.add_argument("--dr", action="store_true", help="Enable domain randomization")
    args = parser.parse_args()

    if not args.robot and not args.all:
        parser.error("Specify --robot or --all")

    robots = list(SUPPORTED_ROBOTS) if args.all else [args.robot]

    for robot in robots:
        print(f"\n{'='*60}")
        print(f"Generating {args.n_demos} demos for: {robot}")
        print(f"{'='*60}")
        collect_demos(
            robot=robot,
            n_demos=args.n_demos,
            output_dir=Path(args.out),
            seed=args.seed,
            max_episode_steps=args.max_steps,
            domain_randomization=args.dr,
            task=args.task,
        )


if __name__ == "__main__":
    main()

