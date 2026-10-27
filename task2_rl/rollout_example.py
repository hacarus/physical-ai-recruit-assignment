"""学習済みエージェントで 1 エピソード実行するサンプル.

使い方:
    uv run python rollout_example.py --ckpt checkpoints/ppo_straight.pt --layout straight
"""

import argparse

import torch

from env import PointNavEnv
from ppo import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--layout", choices=["straight", "wall"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent = ActorCritic()
    agent.load_state_dict(torch.load(args.ckpt))
    agent.eval()

    env = PointNavEnv(args.layout)
    obs = env.reset(seed=args.seed)
    episode_return = 0.0
    while True:
        action = agent.act(torch.from_numpy(obs)).numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        if terminated or truncated:
            break
    print(f"return: {episode_return:.3f}")


if __name__ == "__main__":
    main()
