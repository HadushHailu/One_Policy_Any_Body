"""
Multi-robot HDF5 dataset for diffusion policy training.

Loads scripted-demonstration HDF5 files produced in Week 1,
windows observations & actions into chunks, normalizes, and
attaches the morphology descriptor for each robot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from opab.dataset.normalizer import Normalizer
from opab.model.morphology_encoder import MorphologyEncoder

logger = logging.getLogger(__name__)


class MultiRobotDataset(Dataset):
    """
    Iterable of ``(obs_images, obs_proprio, action_chunk, morph_vec, task_id)``
    sampled from one or more robot demonstration files.

    Parameters
    ----------
    hdf5_paths : list[str | Path]
        Paths to HDF5 demo files (one per robot).
    robot_configs : list
        OmegaConf DictConfigs (or plain dicts) with DH params, limits, etc.
    obs_horizon : int
        Number of past observations to stack (default 2).
    action_horizon : int
        Length of the predicted action chunk (default 16).
    max_proprio_dim : int
        Pad proprioception to this many joints (default 7).
    task_ids : list[int] | None
        Task ID per HDF5 file. If None, inferred from filename
        ('pick_place' -> 0, 'stack' -> 1).
    """

    TASK_NAME_TO_ID = {"pick_place": 0, "stack": 1}

    def __init__(
        self,
        hdf5_paths: list[str | Path],
        robot_configs: list,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        max_proprio_dim: int = 7,
        task_ids: list[int] | None = None,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.max_proprio_dim = max_proprio_dim

        # ----------------------------------------------------------
        # Load all episodes into memory
        # ----------------------------------------------------------
        self.episodes: list[dict[str, np.ndarray]] = []
        self.morph_vecs: list[torch.Tensor] = []
        self.task_id_per_episode: list[int] = []

        for file_idx, (path, robot_cfg) in enumerate(zip(hdf5_paths, robot_configs)):
            path = Path(path)
            morph_vec = MorphologyEncoder.from_robot_config(robot_cfg)
            logger.info(f"Loading {path}  (morph_vec shape {morph_vec.shape})")

            # Determine task_id for this file
            if task_ids is not None:
                tid = task_ids[file_idx]
            else:
                tid = self._infer_task_id(path)

            with h5py.File(path, "r") as f:
                n_eps = int(f.attrs["n_episodes"])
                for i in range(n_eps):
                    ep = f[f"episode_{i}"]
                    self.episodes.append(
                        {
                            "images": np.array(ep["images"]),            # (T, H, W, 3) uint8
                            "actions": np.array(ep["actions"]),          # (T, 4)
                            "proprioception": np.array(ep["proprioception"]),  # (T, n_joints)
                        }
                    )
                    self.morph_vecs.append(morph_vec)
                    self.task_id_per_episode.append(tid)

        # ----------------------------------------------------------
        # Build valid-sample index
        # ----------------------------------------------------------
        self.index: list[tuple[int, int]] = []  # (episode_idx, t)
        for ep_idx, ep in enumerate(self.episodes):
            T = ep["actions"].shape[0]
            # t must allow obs_horizon lookback and action_horizon lookahead
            for t in range(self.obs_horizon - 1, T):
                self.index.append((ep_idx, t))

        logger.info(
            f"MultiRobotDataset: {len(self.episodes)} episodes, "
            f"{len(self.index)} samples"
        )

        # ----------------------------------------------------------
        # Compute normalisation statistics
        # ----------------------------------------------------------
        self.normalizer = Normalizer()
        all_actions = np.concatenate(
            [ep["actions"] for ep in self.episodes], axis=0
        )
        all_proprio = np.concatenate(
            [self._pad_proprio(ep["proprioception"]) for ep in self.episodes],
            axis=0,
        )
        self.normalizer.fit({"actions": all_actions, "proprioception": all_proprio})

    # ==============================================================
    # helpers
    # ==============================================================
    @staticmethod
    def _infer_task_id(path: Path) -> int:
        """Infer task_id from filename: 'pick_place' -> 0, 'stack' -> 1."""
        name = path.stem.lower()
        if "stack" in name:
            return 1
        return 0  # default pick_place

    def _pad_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Pad proprioception to ``max_proprio_dim`` joints."""
        if proprio.shape[-1] >= self.max_proprio_dim:
            return proprio[..., : self.max_proprio_dim]
        pad_width = [(0, 0)] * (proprio.ndim - 1) + [
            (0, self.max_proprio_dim - proprio.shape[-1])
        ]
        return np.pad(proprio, pad_width, constant_values=0.0).astype(np.float32)

    # ==============================================================
    # Dataset interface
    # ==============================================================
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        T = ep["actions"].shape[0]

        # ---- Observation images: (obs_horizon, 3, H, W) [0,1] ----
        obs_images = []
        for i in range(self.obs_horizon):
            frame_t = t - (self.obs_horizon - 1) + i
            img = ep["images"][frame_t].astype(np.float32) / 255.0  # HWC
            img = np.transpose(img, (2, 0, 1))  # CHW
            obs_images.append(img)
        obs_images = np.stack(obs_images, axis=0)

        # ---- Proprioception: (obs_horizon, max_proprio_dim) ----
        obs_proprio = []
        for i in range(self.obs_horizon):
            frame_t = t - (self.obs_horizon - 1) + i
            p = self._pad_proprio(ep["proprioception"][frame_t : frame_t + 1])[0]
            obs_proprio.append(p)
        obs_proprio = np.stack(obs_proprio, axis=0)

        # ---- Action chunk: (action_horizon, action_dim) ----
        action_chunk = ep["actions"][t : t + self.action_horizon].copy()
        pad_len = self.action_horizon - action_chunk.shape[0]
        if pad_len > 0:
            # Repeat last action to fill the chunk
            action_chunk = np.concatenate(
                [action_chunk, np.tile(action_chunk[-1:], (pad_len, 1))], axis=0
            )

        # ---- Convert to tensors ----
        obs_images_t = torch.from_numpy(obs_images)
        obs_proprio_t = torch.from_numpy(obs_proprio)
        action_chunk_t = torch.from_numpy(action_chunk)
        morph_vec = self.morph_vecs[ep_idx].clone()
        task_id = torch.tensor(self.task_id_per_episode[ep_idx], dtype=torch.long)

        # ---- Normalise actions & proprio ----
        action_chunk_t = self.normalizer.normalize("actions", action_chunk_t)
        obs_proprio_t = self.normalizer.normalize("proprioception", obs_proprio_t)

        return {
            "obs_images": obs_images_t,    # (obs_horizon, 3, H, W)
            "obs_proprio": obs_proprio_t,  # (obs_horizon, max_proprio_dim)
            "action": action_chunk_t,      # (action_horizon, action_dim)
            "morph_vec": morph_vec,         # (morph_feature_dim,)
            "task_id": task_id,             # scalar long
        }
