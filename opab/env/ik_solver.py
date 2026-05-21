"""
Damped Least Squares IK solver for task-space control.

Standalone, stateless module — can be used with any robot configuration.
"""
from __future__ import annotations

import mujoco
import numpy as np


def orientation_error(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """
    Compute orientation error as a 3D axis-angle vector.

    Uses the skew-symmetric part of R_err = R_desired @ R_current^T:
        error = 0.5 * [R_err[2,1] - R_err[1,2],
                       R_err[0,2] - R_err[2,0],
                       R_err[1,0] - R_err[0,1]]

    (Siciliano, Robotics: Modelling, Planning and Control, Ch. 3.7)

    Returns:
        (3,) orientation error vector (zero when aligned)
    """
    R_err = R_desired @ R_current.T
    error = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    return error


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    ee_site_id: int,
    ee_body_id: int,
    use_site_for_ee: bool,
    arm_joint_ids: list[int],
    arm_dof_ids: list[int],
    arm_qpos_ids: list[int],
    n_arm_joints: int,
    ik_damping: float = 0.01,
    ik_max_iter: int = 50,
) -> np.ndarray:
    """
    Solve inverse kinematics using Damped Least Squares with orientation.

    Features:
      - 6-DOF position + orientation tracking
      - Adaptive damping (decreases through iterations for precision)
      - Joint limit clamping
      - State preservation (doesn't modify input data permanently)

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be temporarily modified then restored)
        target_pos: (3,) desired EE position
        target_mat: (3,3) desired EE orientation matrix
        ee_site_id: MuJoCo site ID for EE position/orientation
        ee_body_id: MuJoCo body ID for EE fallback
        use_site_for_ee: Whether to use site (True) or body (False)
        arm_joint_ids: List of joint IDs for the arm
        arm_dof_ids: List of DOF addresses for the arm
        arm_qpos_ids: List of qpos addresses for the arm
        n_arm_joints: Number of arm joints
        ik_damping: Base damping factor for DLS
        ik_max_iter: Maximum iteration count

    Returns:
        (n_arm_joints,) target joint positions
    """
    # Save state
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    base_damping = ik_damping
    pos_tol = 1e-4
    ori_tol = 1e-3

    # Orientation weight: full for 6+ DOF, reduced for 5 DOF
    ori_weight = 1.0 if n_arm_joints >= 6 else 0.3

    for it in range(ik_max_iter):
        mujoco.mj_forward(model, data)

        # Current EE pose
        if use_site_for_ee and ee_site_id >= 0:
            ee_pos = data.site_xpos[ee_site_id].copy()
            ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        elif ee_body_id >= 0:
            ee_pos = data.xpos[ee_body_id].copy()
            ee_mat = data.xmat[ee_body_id].reshape(3, 3)
        else:
            break

        # Position error
        pos_err = target_pos - ee_pos
        pos_err_norm = np.linalg.norm(pos_err)

        # Orientation error
        ori_err = orientation_error(ee_mat, target_mat)
        ori_err_norm = np.linalg.norm(ori_err)

        # Check convergence
        if pos_err_norm < pos_tol and ori_err_norm < ori_tol:
            break

        # Compute Jacobian at EE
        if use_site_for_ee and ee_site_id >= 0:
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)

        # Extract arm DOF columns
        J_pos = jacp[:, arm_dof_ids]  # (3, n_arm)
        J_ori = jacr[:, arm_dof_ids]  # (3, n_arm)

        # Stack into 6xN Jacobian with orientation weighting
        J = np.vstack([J_pos, ori_weight * J_ori])  # (6, n_arm)
        err = np.concatenate([pos_err, ori_weight * ori_err])  # (6,)

        # Adaptive damping: decrease over iterations for fine convergence
        lam = base_damping * max(0.1, 1.0 - it / ik_max_iter)

        # DLS: dq = J^T (J J^T + λ²I)^{-1} err
        JJT = J @ J.T + (lam ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Clamp max step
        max_step = 0.5  # rad
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        # Apply delta
        for i, qpos_id in enumerate(arm_qpos_ids):
            data.qpos[qpos_id] += dq[i]

        # Joint limit clamping
        for i, jid in enumerate(arm_joint_ids):
            qpos_id = arm_qpos_ids[i]
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            if lo < hi:  # limits exist
                data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], lo, hi)

    # Extract result
    qpos = np.array([data.qpos[i] for i in arm_qpos_ids])

    # Restore state
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    data.ctrl[:] = saved_ctrl

    return qpos
