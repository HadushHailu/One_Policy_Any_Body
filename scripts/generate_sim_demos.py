#!/usr/bin/env python3
"""Generate simulated teleoperation demonstrations in MuJoCo.

Usage:
    python scripts/generate_sim_demos.py --robot so101 --task pick --n_demos 50 --out data/so101_pick
    python scripts/generate_sim_demos.py --robot franka --task pick --n_demos 50 --out data/franka_pick
"""
import argparse
from pathlib import Path


def make_env(robot: str, task: str):
    """Create a MuJoCo environment for the given robot and task."""
    # TODO: Build MuJoCo env from XML scene configs.
    # Use opab/config/task/{robot}_{task}.yaml for scene parameters.
    raise NotImplementedError(f"MuJoCo env for {robot}/{task} not yet built")


def scripted_policy(obs, robot: str):
    """Simple scripted pick-and-place policy for demonstration generation.

    Uses known object pose from sim state to compute IK waypoints.
    """
    # TODO: Implement per-robot scripted policy
    # 1. Move above object
    # 2. Lower to grasp height
    # 3. Close gripper
    # 4. Lift
    # 5. Move to target
    # 6. Release
    raise NotImplementedError


def collect_demos(robot: str, task: str, n_demos: int, output_dir: Path):
    """Collect n_demos demonstrations and save as HDF5."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(robot, task)
    demos = []

    for i in range(n_demos):
        obs = env.reset()
        episode = {"observations": [], "actions": []}

        done = False
        while not done:
            action = scripted_policy(obs, robot)
            episode["observations"].append(obs)
            episode["actions"].append(action)
            obs, reward, done, info = env.step(action)

        demos.append(episode)
        print(f"  Demo {i+1}/{n_demos} — steps: {len(episode['actions'])}, success: {info.get('success', 'N/A')}")

    # Save as HDF5 (LeRobot format)
    save_path = output_dir / f"{robot}_{task}_demos.hdf5"
    # TODO: Serialize demos list to HDF5
    print(f"Saved {n_demos} demos to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sim demos")
    parser.add_argument("--robot", required=True, choices=["franka", "ur5", "so101"])
    parser.add_argument("--task", default="pick", choices=["pick"])
    parser.add_argument("--n_demos", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/demos")
    args = parser.parse_args()

    print(f"Generating {args.n_demos} demos for {args.robot}/{args.task}")
    collect_demos(args.robot, args.task, args.n_demos, Path(args.out))


if __name__ == "__main__":
    main()
