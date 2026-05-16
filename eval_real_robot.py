"""
eval_real_robot.py — Deploy and evaluate on physical SO-101

Usage:
    python eval_real_robot.py checkpoint=data/outputs/.../checkpoints/latest.ckpt
"""

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="opab/config",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    from opab.env.real_env import RealSO101Env
    from opab.workspace import load_policy_from_checkpoint

    policy = load_policy_from_checkpoint(cfg)
    env = RealSO101Env(cfg)

    num_episodes = cfg.training.num_eval_episodes
    successes = 0

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.predict_action(obs)
            obs, reward, done, info = env.step(action)
        if info.get("success", False):
            successes += 1
        print(f"Episode {ep+1}/{num_episodes}: {'SUCCESS' if info.get('success') else 'FAIL'}")

    print(f"\nSuccess rate: {successes}/{num_episodes} = {successes/num_episodes:.1%}")


if __name__ == "__main__":
    main()
