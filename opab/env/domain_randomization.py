"""
Domain randomization module for sim-to-real transfer.

Randomizes physics and visual parameters at each episode reset
to create a distribution of environments that covers real-world variation.

Week 1: Implemented but DISABLED (enabled=False).
Week 3: Enable during data collection.
Week 6: Maximum DR for sim-to-real transfer.
"""
from __future__ import annotations

import mujoco
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DRConfig:
    """Domain randomization configuration."""
    enabled: bool = False

    # Physics
    cube_mass_range: float = 0.3        # ±fraction of nominal
    cube_friction_range: float = 0.3    # ±fraction of nominal
    joint_damping_range: float = 0.2    # ±fraction of nominal
    gravity_range: float = 0.1          # ±m/s² from 9.81

    # Geometry
    cube_pos_range: float = 0.05        # ±m from default position

    # Visual
    table_color_randomize: bool = True
    lighting_randomize: bool = True

    # Camera
    camera_pos_range: float = 0.02      # ±m
    camera_angle_range: float = 0.05    # ±rad


class DomainRandomizer:
    """Apply domain randomization to a MuJoCo model at each episode reset."""

    def __init__(self, model: mujoco.MjModel, config: DRConfig):
        self.model = model
        self.config = config
        self._save_nominal()

    def _save_nominal(self):
        """Save original model parameters for restoration."""
        self._nominal = {
            "body_mass": self.model.body_mass.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "gravity": self.model.opt.gravity.copy(),
            "geom_rgba": self.model.geom_rgba.copy(),
        }

    def randomize(self, rng: np.random.Generator):
        """
        Apply random perturbations to model parameters.
        Call this at the beginning of each episode (during reset).
        """
        if not self.config.enabled:
            return

        cfg = self.config

        # --- Physics randomization ---

        # Cube mass
        cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        if cube_body_id >= 0:
            scale = rng.uniform(1 - cfg.cube_mass_range, 1 + cfg.cube_mass_range)
            self.model.body_mass[cube_body_id] = (
                self._nominal["body_mass"][cube_body_id] * scale
            )

        # Cube friction
        cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        if cube_geom_id >= 0:
            scale = rng.uniform(1 - cfg.cube_friction_range, 1 + cfg.cube_friction_range)
            self.model.geom_friction[cube_geom_id, 0] = (
                self._nominal["geom_friction"][cube_geom_id, 0] * scale
            )

        # Joint damping
        for i in range(self.model.nv):
            scale = rng.uniform(1 - cfg.joint_damping_range, 1 + cfg.joint_damping_range)
            self.model.dof_damping[i] = self._nominal["dof_damping"][i] * scale

        # Gravity
        self.model.opt.gravity[2] = (
            self._nominal["gravity"][2]
            + rng.uniform(-cfg.gravity_range, cfg.gravity_range)
        )

        # --- Visual randomization ---

        # Table color
        if cfg.table_color_randomize:
            table_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_surface"
            )
            if table_geom_id >= 0:
                self.model.geom_rgba[table_geom_id, :3] = rng.uniform(0.2, 0.8, 3)

    def reset_to_nominal(self):
        """Restore all parameters to their original values."""
        self.model.body_mass[:] = self._nominal["body_mass"]
        self.model.geom_friction[:] = self._nominal["geom_friction"]
        self.model.dof_damping[:] = self._nominal["dof_damping"]
        self.model.opt.gravity[:] = self._nominal["gravity"]
        self.model.geom_rgba[:] = self._nominal["geom_rgba"]
