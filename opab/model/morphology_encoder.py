"""
Morphology Encoder — URDF → fixed-size descriptor

Converts robot kinematic parameters (DH params, joint limits,
workspace volume, payload) into a compact 32-dim embedding.
Used for FiLM conditioning in the diffusion U-Net.

See docs/planning/02_One_Policy_Any_Body_objective.md § F4 for theory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MorphologyEncoder(nn.Module):
    """
    Encodes robot morphology from kinematic parameters into a
    fixed-size vector for FiLM conditioning.

    Input features per joint (padded to max_joints=7):
        - DH params: a, d, alpha, theta_offset (4 values)
        - Joint limits: lower, upper (2 values)
    Plus global features:
        - workspace_radius, payload_kg, gripper_max_width, total_dof

    Total input dim: 7 * 6 + 4 = 46 (padded)
    Output dim: 32
    """

    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.morphology_encoder.input_dim
        hidden_dims = cfg.morphology_encoder.hidden_dims
        output_dim = cfg.morphology_encoder.output_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim) if cfg.morphology_encoder.use_layer_norm else nn.Identity(),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, morph_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            morph_features: (batch, input_dim) — flattened kinematic params

        Returns:
            morph_embedding: (batch, 32)
        """
        return self.encoder(morph_features)

    @staticmethod
    def from_robot_config(robot_cfg) -> torch.Tensor:
        """
        Extract morphology feature vector from a robot config (YAML).

        Returns a 1D tensor ready for the encoder.
        """
        import numpy as np

        dh = np.array(robot_cfg.dh_params)                     # (n_joints, 4)
        limits_lower = np.array(robot_cfg.joint_limits.lower)   # (n_joints,)
        limits_upper = np.array(robot_cfg.joint_limits.upper)   # (n_joints,)

        n_joints = len(limits_lower)
        max_joints = robot_cfg.get("pad_to_n_joints", 7)

        # Per-joint features: DH (4) + limits (2) = 6 per joint
        per_joint = np.zeros((max_joints, 6))
        per_joint[:n_joints, :4] = dh
        per_joint[:n_joints, 4] = limits_lower
        per_joint[:n_joints, 5] = limits_upper

        # Global features
        global_features = np.array([
            robot_cfg.workspace_radius,
            robot_cfg.payload_kg,
            robot_cfg.gripper_max_width,
            float(n_joints),
        ])

        features = np.concatenate([per_joint.flatten(), global_features])
        return torch.tensor(features, dtype=torch.float32)
