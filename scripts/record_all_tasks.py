#!/usr/bin/env python3
"""Record demo videos for all 7 tasks with lite6 robot.

Generates: videos/lite6_{task}.mp4 for each task.
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opab.env.base_env import PickPlaceEnv

Path("videos").mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Camera configurations
# ---------------------------------------------------------------
# Per-robot camera params are stored in env.cfg.cameras
# Fallback sideview_0 (used if cfg doesn't have cameras dict)
SIDEVIEW_0_FALLBACK = {
    "lookat": [0.15, 0.0, 0.45],
    "distance": 0.70,
    "azimuth": 150,
    "elevation": -22,
}


def make_env(task):
    env = PickPlaceEnv(robot='lite6', task=task, seed=42)
    env.reset(seed=42)
    return env


def make_video(env, task, cam_name="sideview_0"):
    renderer = mujoco.Renderer(env.model, 480, 640)
    cam = mujoco.MjvCamera()

    # Get per-robot camera config
    robot_cameras = getattr(env.cfg, 'cameras', {})

    if cam_name == "sideview_0":
        cam_cfg = robot_cameras.get("sideview_0", SIDEVIEW_0_FALLBACK)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cam_cfg["lookat"]
        cam.distance = cam_cfg["distance"]
        cam.azimuth = cam_cfg["azimuth"]
        cam.elevation = cam_cfg["elevation"]
    else:
        # Use a fixed camera from the XML (agentview_0, topdown_0, etc.)
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam.fixedcamid < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = f'videos/lite6_{cam_name}_{task}.mp4'
    out = cv2.VideoWriter(path, fourcc, 30, (640, 480))

    ee_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')

    def render():
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, cam)
        out.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    def ee_pos():
        return env.data.site_xpos[ee_sid].copy()

    def move_to(target, steps=25, grip=0.0):
        for _ in range(steps):
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], 0.0, grip]))
            render()

    def grip_close(steps=8):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 1.]))
            render()

    def grip_open(steps=8):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 0.]))
            render()

    def idle(steps=5):
        for _ in range(steps):
            env.step(np.array([0., 0., 0., 0., 0.]))
            render()

    # ---------------------------------------------------------------
    # Task-specific scripted policies
    # ---------------------------------------------------------------

    if task == "pick_place":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        # Approach above cube
        move_to(cube_pos + [0, 0, 0.05], steps=25)
        # Lower so finger pads grip cube (palm stays above cube top)
        # EE at cube-5mm keeps palm ~4mm above cube top (palm is ~27mm above EE)
        move_to(cube_pos + [0, 0, -0.005], steps=25)
        # Grasp
        grip_close(10)
        # Lift
        move_to(cube_pos + [0, 0, 0.08], steps=15, grip=1.0)
        # Move to target
        move_to(target_pos + [0, 0, 0.06], steps=20, grip=1.0)
        # Lower to place (stop above table, release, gravity settles)
        # target_pos z is site height (0.405), cube rests at table_top+cube_size (0.418)
        # EE needs to be ~0.013 above target_pos for gentle drop
        move_to(target_pos + [0, 0, 0.013], steps=20, grip=1.0)
        # Release
        grip_open(8)
        # Retract
        move_to(target_pos + [0, 0, 0.06], steps=10)

    elif task == "push":
        cube_pos = env.get_cube_pos()
        target_pos = env.get_target_pos()
        cs = env.cfg.cube_size
        # Push direction: cube -> target (purely -Y for lite6)
        push_dir = (target_pos[:2] - cube_pos[:2])
        push_dist = np.linalg.norm(push_dir)
        push_dir = push_dir / (push_dist + 1e-8)
        # "Behind" = opposite side of push direction
        behind_offset = 0.04  # 4cm behind cube center
        behind_xy = cube_pos[:2] - push_dir * behind_offset
        # Contact height: raise EE above cube center so palm clears cube top
        # Palm visual bottom is ~27mm above EE; cube top is at cube_pos[2]+cs
        # EE at cube_pos[2]+0.008 → palm bottom ~17mm above cube top (clean gap)
        contact_z = cube_pos[2] + 0.008
        # Close gripper first (use closed fingers as pusher)
        grip_close(5)
        # Approach behind cube from above (more steps to ensure convergence)
        move_to(np.array([behind_xy[0], behind_xy[1], contact_z + 0.05]), steps=25, grip=1.0)
        # Lower to push height (more steps for reliable descent)
        move_to(np.array([behind_xy[0], behind_xy[1], contact_z]), steps=25, grip=1.0)
        # Push toward target (compute exact distance to avoid overshoot)
        # From behind position to target = behind_offset + push_dist - cs (stop when cube center reaches target)
        total_push = behind_offset + push_dist - cs
        step_size = 0.005
        n_push_steps = int(total_push / step_size)
        for i in range(n_push_steps):
            env.step(np.array([push_dir[0]*step_size, push_dir[1]*step_size, 0.0, 0.0, 1.0]))
            render()
        # Retract up
        move_to(ee_pos() + [0, 0, 0.05], steps=10, grip=1.0)

    elif task == "stack":
        cube_a_pos = env.get_cube_pos()
        cube_b_pos = env.get_cube_b_pos()
        cs = env.cfg.cube_size
        # Pick cube_A (EE at cube-5mm: palm stays above cube top)
        move_to(cube_a_pos + [0, 0, 0.05], steps=25)
        move_to(cube_a_pos + [0, 0, -0.005], steps=25)
        grip_close(10)
        move_to(cube_a_pos + [0, 0, 0.08], steps=15, grip=1.0)
        # Move above cube_B
        stack_target = cube_b_pos + [0, 0, 2*cs + 0.06]
        move_to(stack_target, steps=20, grip=1.0)
        # Lower cube_A onto cube_B (stop just above so arm doesn't push down)
        move_to(cube_b_pos + [0, 0, 2*cs + 0.003], steps=20, grip=1.0)
        # Release
        grip_open(8)
        # Retract
        move_to(ee_pos() + [0, 0, 0.05], steps=10)

    elif task == "peg_insertion":
        peg_pos = env.get_peg_pos()
        hole_pos = env.get_hole_pos()
        # Approach peg
        move_to(peg_pos + [0, 0, 0.05], steps=20)
        move_to(peg_pos + [0, 0, 0.003], steps=15)
        grip_close(10)
        # Lift peg high (clear of hole height)
        move_to(peg_pos + [0, 0, 0.10], steps=25, grip=1.0)
        # Move XY to above hole at same high Z — slow transport
        high_above_hole = np.array([hole_pos[0], hole_pos[1], ee_pos()[2]])
        for s in range(50):
            d = np.clip((high_above_hole - ee_pos()) * 0.3, -0.005, 0.005)
            env.step(np.array([d[0], d[1], d[2], 0.0, 1.0]))
            render()
        # Descend to just above hole opening — keep fingers clear of socket walls
        # hole_pos is at hole BOTTOM; hole rim top = hole_body_center + hole_depth
        # hole_body_center = hole_pos + hole_depth; rim = hole_pos + 2*hole_depth
        # EE at rim + 0.005 keeps fingertips 5mm above socket walls
        hole_depth = env.cfg.hole_depth
        for s in range(35):
            target = hole_pos + np.array([0, 0, 2*hole_depth + 0.005])
            d = np.clip((target - ee_pos()) * 0.3, -0.003, 0.003)
            env.step(np.array([d[0], d[1], d[2], 0.0, 1.0]))
            render()
        # Release peg — tip is already inside/at hole opening
        grip_open(5)
        # Wait for peg to settle into hole
        idle(30)
        # Retract
        move_to(ee_pos() + [0, 0, 0.05], steps=10)

    elif task == "drawer_open":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'drawer_handle_site')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        # Approach above handle (like door: go to ~40mm above bar)
        move_to(handle_pos + [0, 0, 0.04], steps=25)
        # Slow descent to bar (30 steps for 45mm — gentle to avoid bounce)
        move_to(handle_pos + [0, 0, -0.005], steps=30)
        # Grasp
        grip_close(10)
        # Pull drawer open — track handle position to maintain grip
        drawer_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'drawer_joint')
        for i in range(80):
            cur_handle = env.data.site_xpos[handle_sid].copy()
            error = cur_handle - ee_pos()
            dx = np.clip(error[0] * 0.5, -0.003, 0.003)
            dz = np.clip(error[2] * 0.5, -0.003, 0.003)
            env.step(np.array([dx, -0.003, dz, 0.0, 1.0]))
            render()
            dpos = env.data.qpos[env.model.jnt_qposadr[drawer_jid]]
            if dpos >= 0.045:
                break
        # Release — partial open only (avoid hitting cabinet with inner finger)
        for _ in range(6):
            env.step(np.array([0., 0., 0.001, 0., 0.5]))  # lift slightly + half-open grip
            render()
        # Retract
        move_to(ee_pos() + [0, -0.03, 0.03], steps=10)

    elif task == "turn_faucet":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'faucet_handle_site')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        faucet_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'faucet_joint')
        faucet_qadr = env.model.jnt_qposadr[faucet_jid]
        switch_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'faucet_switch')
        # Compute lever direction to orient gripper perpendicular to it
        switch_pos = env.data.xpos[switch_bid].copy()
        lever_dir = handle_pos[:2] - switch_pos[:2]
        lever_angle = np.arctan2(lever_dir[1], lever_dir[0])
        # Rotate wrist so fingers close ACROSS the lever (perpendicular)
        # Target yaw = lever_angle (fingers align with lever → grip across it)
        target_yaw = lever_angle
        # Approach handle — rotate wrist during approach
        for s in range(20):
            target = handle_pos + np.array([0, 0, 0.03])
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], target_yaw * 0.1, 0.0]))
            render()
        # Descend to handle
        for s in range(15):
            target = handle_pos + np.array([0, 0, -0.003])
            d = np.clip((target - ee_pos()) * 0.3, -0.01, 0.01)
            env.step(np.array([d[0], d[1], d[2], 0.0, 0.0]))
            render()
        grip_close(10)
        # Rotate: track handle arc (same pattern as door_open)
        for i in range(60):
            cur_handle = env.data.site_xpos[handle_sid].copy()
            switch_pos = env.data.xpos[switch_bid].copy()
            # Radius from hinge to handle
            radius_vec = cur_handle[:2] - switch_pos[:2]
            # Tangent for CW rotation (negative angle): rotate radius 90° CW -> (ry, -rx)
            tangent = np.array([radius_vec[1], -radius_vec[0]])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            # Track handle position (match door: high gain + wide clip)
            error = cur_handle - ee_pos()
            error[2] = 0
            pull_str = 0.004
            track_gain = 0.5
            dx = tangent[0] * pull_str + np.clip(error[0] * track_gain, -0.008, 0.008)
            dy = tangent[1] * pull_str + np.clip(error[1] * track_gain, -0.008, 0.008)
            env.step(np.array([dx, dy, 0.0, 0.0, 1.0]))
            render()
            angle = env.data.qpos[faucet_qadr]
            if angle <= -1.0:
                break
        # Release
        grip_open(8)
        move_to(ee_pos() + [0, 0, 0.04], steps=10)

    elif task == "door_open":
        handle_sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
        hinge_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'door_hinge')
        hinge_qadr = env.model.jnt_qposadr[hinge_jid]
        hinge_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'door_panel')
        handle_pos = env.data.site_xpos[handle_sid].copy()
        # Grip target: shift along bar + lower for center grip
        grip_target = handle_pos + np.array([-0.012, 0, -0.005])
        # Approach above
        move_to(grip_target + [0, 0, 0.04], steps=25)
        # Lower to bar
        move_to(grip_target, steps=15)
        # Grasp
        grip_close(10)
        # Pull door open - track handle arc
        prev_angle = 0.0
        stall_count = 0
        for i in range(120):
            angle = env.data.qpos[hinge_qadr]
            if angle >= 1.48:
                break
            if i > 15 and abs(angle - prev_angle) < 0.001:
                stall_count += 1
                if stall_count > 10:
                    break
            else:
                stall_count = 0
            prev_angle = angle
            cur_handle = env.data.site_xpos[handle_sid].copy()
            hinge_pos = env.data.xpos[hinge_bid].copy()
            radius_vec = cur_handle[:2] - hinge_pos[:2]
            tangent = np.array([-radius_vec[1], radius_vec[0]])
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            error = cur_handle - ee_pos()
            error[2] = 0
            pull_str = 0.004
            track_gain = 0.5
            dx = tangent[0] * pull_str + np.clip(error[0] * track_gain, -0.008, 0.008)
            dy = tangent[1] * pull_str + np.clip(error[1] * track_gain, -0.008, 0.008)
            env.step(np.array([dx, dy, 0.0, 0.02, 1.0]))
            render()
        # Release + retract
        grip_open(5)
        move_to(ee_pos() + [0, 0, 0.04], steps=10)

    out.release()
    renderer.close()
    return path


# ---------------------------------------------------------------
# Main: record all tasks
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="lite6")
    parser.add_argument("--task", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--cam", default="sideview_0",
                        help="Camera name: sideview_0 (free), agentview_0, topdown_0")
    args = parser.parse_args()

    tasks = ["pick_place", "push", "stack", "peg_insertion",
             "drawer_open", "turn_faucet", "door_open"]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Recording: {task} [{args.cam}]")
        print(f"{'='*50}")
        env = make_env(task)
        path = make_video(env, task, cam_name=args.cam)
        env.close()
        print(f"  -> {path}")

    print(f"\n{'='*50}")
    print(f"All {len(tasks)} videos recorded in videos/")
