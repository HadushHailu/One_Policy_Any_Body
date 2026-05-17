#!/usr/bin/env python3
"""
Launch MuJoCo interactive viewer with the OPAB pick-and-place scene.

Usage:
  python scripts/visualize.py                    # default: franka
  python scripts/visualize.py --robot franka
  python scripts/visualize.py --robot ur5
  python scripts/visualize.py --robot so101
  python scripts/visualize.py --robot franka --run-policy   # run scripted policy
"""
import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opab.env import make_env
from opab.env.scripted_policies import ScriptedPickPlace


def main():
    parser = argparse.ArgumentParser(description="OPAB MuJoCo Visualizer")
    parser.add_argument("--robot", default="franka",
                        choices=["franka", "ur5", "so101"])
    parser.add_argument("--run-policy", action="store_true",
                        help="Run the scripted pick-and-place policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.robot} scene...")
    env = make_env(args.robot, seed=args.seed)
    obs = env.reset()

    if args.run_policy:
        # Run scripted policy in the viewer
        policy = ScriptedPickPlace(args.robot)
        policy.reset()

        def controller(model, data):
            """Called at each physics step by the viewer."""
            pass  # We control via env.step, not here

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            print("Running scripted policy... (close window to stop)")
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = env.cfg.table_pos

            step = 0
            while viewer.is_running():
                cube_pos = env.get_cube_pos()
                target_pos = env.get_target_pos()
                action = policy.get_action(obs, cube_pos, target_pos)

                if action is None:
                    # Policy done — pause and show result
                    print(f"Policy finished at step {step}")
                    print(f"  Cube: {env.get_cube_pos()}")
                    print(f"  Target: {env.get_target_pos()}")
                    dist = np.linalg.norm(
                        env.get_cube_pos()[:2] - env.get_target_pos()[:2]
                    )
                    print(f"  Distance: {dist:.4f}m")
                    # Just spin the viewer
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break

                obs, _, term, trunc, info = env.step(action)
                viewer.sync()
                step += 1

                # Slow down to ~20 Hz for visibility
                time.sleep(env.control_dt)

                if term or trunc:
                    print(f"Episode ended at step {step}: "
                          f"success={info['success']}")
                    break
    else:
        # Just launch the passive viewer for inspection
        print("Launching viewer... (close window to stop)")
        print("  - Click and drag to rotate")
        print("  - Scroll to zoom")
        print("  - Double-click to select bodies")
        print("  - Ctrl+click to apply forces")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = env.cfg.table_pos

            while viewer.is_running():
                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                time.sleep(env.model.opt.timestep)

    env.close()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
