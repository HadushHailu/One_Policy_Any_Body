#!/usr/bin/env python3
"""
Test full pose randomization for Lite6 pick_place task (Section 6.1).

Randomization budget:
  Cube X:  [0.18, 0.42]  (±0.12 from 0.30)
  Cube Y:  [-0.10, 0.22] (±0.12 from 0.10) — cube side of table
  Target X: [0.18, 0.42]
  Target Y: [-0.22, 0.10] — opposite side from cube
  Cube θ_z: [0, 2π]      — free rotation
  Min dist: ≥ 0.08 m     — avoid trivial placements

Usage:
    python scripts/test_randomize_pick_place.py [--episodes 50] [--level full]
"""

import argparse
import mujoco
import numpy as np
from opab.env.base_env import PickPlaceEnv


def scripted_pick_place(env: PickPlaceEnv) -> bool:
    """Run scripted pick-and-place policy. Returns True if successful."""
    ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')

    def ee_pos():
        return env.data.site_xpos[ee_sid].copy()

    def move_to(target, steps=25, grip=0.0):
        for _ in range(steps):
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], 0.0, grip]))

    def grip_close(steps=10):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 1.]))

    def grip_open(steps=8):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 0.]))

    cube_pos = env.get_cube_pos()
    target_pos = env.get_target_pos()

    # Approach above cube
    move_to(cube_pos + [0, 0, 0.05], steps=25)
    # Lower to grasp
    move_to(cube_pos + [0, 0, -0.005], steps=25)
    # Grasp
    grip_close(10)
    # Lift
    move_to(cube_pos + [0, 0, 0.08], steps=15, grip=1.0)
    # Move to target
    move_to(target_pos + [0, 0, 0.06], steps=20, grip=1.0)
    # Lower to place
    move_to(target_pos + [0, 0, 0.013], steps=20, grip=1.0)
    # Release
    grip_open(8)
    # Settle
    for _ in range(20):
        env.step(np.array([0., 0., 0., 0., 0.]))

    # Check success
    cube_final = env.get_cube_pos()
    dist = np.linalg.norm(cube_final[:2] - target_pos[:2])
    success = dist < env.cfg.success_threshold
    return success, dist, cube_pos[:2].copy(), target_pos[:2].copy()


# --- Randomization bounds (Section 6.1) ---
RANDOM_LEVELS = {
    'conservative': {  # ±4cm relative (original)
        'cube_x': (0.26, 0.34),
        'cube_y': (0.06, 0.14),
        'target_x': (0.26, 0.34),
        'target_y': (-0.14, -0.06),
        'min_dist': 0.05,
        'rotate_cube': False,
    },
    'full': {  # Section 6.1 full budget
        'cube_x': (0.18, 0.42),
        'cube_y': (-0.10, 0.22),
        'target_x': (0.18, 0.42),
        'target_y': (-0.22, 0.10),
        'min_dist': 0.08,
        'rotate_cube': True,
    },
}


def euler_z_to_quat(theta_z: float) -> np.ndarray:
    """Convert Z-axis rotation to quaternion [w, x, y, z]."""
    return np.array([
        np.cos(theta_z / 2), 0.0, 0.0, np.sin(theta_z / 2)
    ])


def main():
    parser = argparse.ArgumentParser(description="Test randomized pick_place (Section 6.1)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--level", choices=['conservative', 'full'], default='full',
                        help="Randomization level (conservative=±4cm, full=Section 6.1 budget)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    args = parser.parse_args()

    bounds = RANDOM_LEVELS[args.level]
    min_dist = bounds['min_dist']

    print(f"{'='*60}")
    print(f"  Randomized Pick-and-Place — Level: {args.level}")
    print(f"  Robot: lite6 | Episodes: {args.episodes}")
    print(f"  Cube  X: [{bounds['cube_x'][0]:.2f}, {bounds['cube_x'][1]:.2f}]  "
          f"Y: [{bounds['cube_y'][0]:.2f}, {bounds['cube_y'][1]:.2f}]")
    print(f"  Target X: [{bounds['target_x'][0]:.2f}, {bounds['target_x'][1]:.2f}]  "
          f"Y: [{bounds['target_y'][0]:.2f}, {bounds['target_y'][1]:.2f}]")
    print(f"  Min cube↔target: {min_dist*100:.0f}cm | Rotate cube: {bounds['rotate_cube']}")
    print(f"{'='*60}\n")

    env = PickPlaceEnv(robot='lite6', task='pick_place', seed=args.seed)

    successes = 0
    failures = []
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        # Sample absolute positions within bounds (reject if too close)
        for _attempt in range(5000):
            cube_x = rng.uniform(*bounds['cube_x'])
            cube_y = rng.uniform(*bounds['cube_y'])
            tgt_x = rng.uniform(*bounds['target_x'])
            tgt_y = rng.uniform(*bounds['target_y'])

            new_cube_xy = np.array([cube_x, cube_y])
            new_tgt_xy = np.array([tgt_x, tgt_y])

            if np.linalg.norm(new_cube_xy - new_tgt_xy) >= min_dist:
                break

        # Sample cube Z-rotation
        cube_theta_z = rng.uniform(0, 2 * np.pi) if bounds['rotate_cube'] else 0.0

        # Reset env to nominal
        env.reset(seed=args.seed + ep)

        # --- Apply cube position + rotation ---
        cube_qpos_adr = env.model.jnt_qposadr[env._cube_joint_id]
        env.data.qpos[cube_qpos_adr] = cube_x       # X
        env.data.qpos[cube_qpos_adr + 1] = cube_y   # Y
        # Z stays at table surface (from reset)

        # Apply rotation (freejoint quat is at qpos_adr + 3:7)
        if bounds['rotate_cube']:
            quat = euler_z_to_quat(cube_theta_z)
            env.data.qpos[cube_qpos_adr + 3: cube_qpos_adr + 7] = quat

        # --- Apply target position ---
        tgt_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'target_zone')
        env.model.site_pos[tgt_sid, 0] = tgt_x
        env.model.site_pos[tgt_sid, 1] = tgt_y

        mujoco.mj_forward(env.model, env.data)

        # Run scripted policy
        success, dist, cube_xy, tgt_xy = scripted_pick_place(env)

        status = "✓" if success else "✗"
        if success:
            successes += 1
        else:
            failures.append((ep, cube_xy, tgt_xy, dist, cube_theta_z))

        if (ep + 1) % 10 == 0 or not success:
            rot_str = f" θ={np.degrees(cube_theta_z):5.1f}°" if bounds['rotate_cube'] else ""
            print(f"  [{ep+1:3d}/{args.episodes}] {status}  cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}){rot_str} "
                  f"target=({tgt_xy[0]:.3f},{tgt_xy[1]:.3f}) dist={dist*100:.1f}cm")

    # Summary
    rate = successes / args.episodes * 100
    print(f"\n{'='*60}")
    print(f"  RESULTS: {successes}/{args.episodes} = {rate:.1f}% success")
    print(f"{'='*60}")

    if failures:
        print(f"\n  Failed episodes ({len(failures)}):")
        for ep, cube_xy, tgt_xy, dist, theta in failures[:10]:
            print(f"    ep={ep}: cube=({cube_xy[0]:.3f},{cube_xy[1]:.3f}) θ={np.degrees(theta):.0f}° "
                  f"target=({tgt_xy[0]:.3f},{tgt_xy[1]:.3f}) final_dist={dist*100:.1f}cm")


if __name__ == "__main__":
    main()
