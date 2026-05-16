"""
demo_teleop.py — Teleoperate SO-101 and record demonstrations

Usage:
    python demo_teleop.py --robot so101 --task pick --save_dir data/demos/so101_pick/
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Teleoperate robot and record demonstrations")
    parser.add_argument("--robot", type=str, default="so101", choices=["franka", "ur5", "so101"])
    parser.add_argument("--task", type=str, default="pick", choices=["pick", "stack", "pour"])
    parser.add_argument("--save_dir", type=str, default="data/demos/")
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    # TODO: Implement teleoperation recording
    # 1. Connect to robot (real) or sim
    # 2. Start camera recording
    # 3. Capture joint states + images at 10Hz
    # 4. Save as HDF5 episode files
    raise NotImplementedError("Teleoperation recording not yet implemented")


if __name__ == "__main__":
    main()
