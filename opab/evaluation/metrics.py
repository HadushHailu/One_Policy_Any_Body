"""Evaluation metrics for cross-embodiment manipulation."""
import numpy as np
from typing import Dict, List


def compute_success_rate(results: List[Dict]) -> Dict[str, float]:
    """Compute per-robot, per-task, and aggregate success rates."""
    if not results:
        return {"overall": 0.0}
    
    metrics = {}
    successes = [r["success"] for r in results]
    metrics["overall"] = float(np.mean(successes))
    
    robots = set(r["robot"] for r in results)
    for robot in sorted(robots):
        robot_results = [r for r in results if r["robot"] == robot]
        metrics[f"per_robot/{robot}"] = float(np.mean([r["success"] for r in robot_results]))
    
    tasks = set(r["task"] for r in results)
    for task in sorted(tasks):
        task_results = [r for r in results if r["task"] == task]
        metrics[f"per_task/{task}"] = float(np.mean([r["success"] for r in task_results]))
    
    for robot in sorted(robots):
        for task in sorted(tasks):
            combo = [r for r in results if r["robot"] == robot and r["task"] == task]
            if combo:
                metrics[f"{robot}/{task}"] = float(np.mean([r["success"] for r in combo]))
    
    return metrics


def compute_episode_stats(results: List[Dict]) -> Dict[str, float]:
    """Compute episode-level statistics."""
    if not results:
        return {}
    return {
        "mean_steps": float(np.mean([r.get("steps", 0) for r in results])),
        "mean_reward": float(np.mean([r.get("total_reward", 0) for r in results])),
        "n_episodes": len(results),
        "n_success": sum(r["success"] for r in results),
    }
