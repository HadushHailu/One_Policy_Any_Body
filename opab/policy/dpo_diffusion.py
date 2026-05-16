"""
DPO for Diffusion Policies — Novel technical contribution

Adapts Direct Preference Optimization from autoregressive LLMs
to non-autoregressive diffusion models. Key challenge: computing
log π(a|o) for diffusion requires ELBO approximation.

See docs/planning/02_One_Policy_Any_Body_objective.md § F7 for theory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPODiffusionPolicy(nn.Module):
    """
    Wraps a trained MorphologyConditionedDP and adds DPO fine-tuning.

    DPO loss for diffusion:
        L = -E[ log σ( β · ( r_θ(a_w) - r_θ(a_l) ) ) ]
    where:
        r_θ(a) = log π_θ(a|o,m) - log π_ref(a|o,m)
        log π_θ(a|o,m) ≈ -1/T Σ_t ||ε - ε_θ(a_t, t, o, m)||²  (ELBO)
    """

    def __init__(self, policy: nn.Module, cfg):
        super().__init__()
        self.policy = policy                    # trainable
        self.reference = None                   # frozen copy (set in load)
        self.beta = cfg.dpo.beta
        self.num_likelihood_samples = cfg.dpo.num_likelihood_samples

    def load_reference(self, checkpoint_path: str):
        """Load and freeze reference policy."""
        # TODO: Load checkpoint, deep copy, freeze all params
        raise NotImplementedError

    def compute_log_likelihood(
        self, policy: nn.Module, actions: torch.Tensor, obs: dict, morph: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate log π(a|o,m) via ELBO.

        For each diffusion timestep t:
            1. Add noise to action: a_t = √ᾱ_t · a + √(1-ᾱ_t) · ε
            2. Predict noise: ε̂ = ε_θ(a_t, t, o, m)
            3. Reconstruction error: ||ε - ε̂||²

        Average over T timesteps → approximate negative log-likelihood.
        """
        raise NotImplementedError

    def compute_dpo_loss(self, batch: dict) -> torch.Tensor:
        """
        DPO loss from a batch of preference pairs.

        batch contains:
            - obs: observation at decision point
            - morph: morphology descriptor
            - action_preferred: the human-preferred action trajectory
            - action_rejected: the human-rejected action trajectory
        """
        obs = batch["obs"]
        morph = batch["morph"]
        a_w = batch["action_preferred"]
        a_l = batch["action_rejected"]

        # Compute log-likelihoods under both policies
        log_pi_w = self.compute_log_likelihood(self.policy, a_w, obs, morph)
        log_pi_l = self.compute_log_likelihood(self.policy, a_l, obs, morph)
        log_ref_w = self.compute_log_likelihood(self.reference, a_w, obs, morph)
        log_ref_l = self.compute_log_likelihood(self.reference, a_l, obs, morph)

        # DPO implicit reward
        reward_w = log_pi_w - log_ref_w
        reward_l = log_pi_l - log_ref_l

        # Bradley-Terry loss
        loss = -F.logsigmoid(self.beta * (reward_w - reward_l)).mean()

        return loss
