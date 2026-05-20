"""Evaluation module for OPAB.

Handles rollout execution, success metrics computation, and video recording
across all robot×task combinations.
"""
from opab.evaluation.rollout_runner import RolloutRunner
from opab.evaluation.metrics import compute_success_rate

__all__ = ["RolloutRunner", "compute_success_rate"]
