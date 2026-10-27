"""学習済みモデルで 1 エピソード実行するサンプル.

使い方:
    uv run python rollout_example.py --ckpt checkpoints/cnn_straight.pt --layout straight
"""

import argparse

import numpy as np
import torch

from env import PointNavEnv
from models import N_ACTION_STEPS, load_policy


def run_episode(policy, env, seed=None):
    obs = env.reset(seed=seed)
    trajectory = [obs[:2].copy()]
    while True:
        action_seq = policy.predict_action(torch.from_numpy(obs)[None])[0].numpy()
        for action in action_seq[:N_ACTION_STEPS]:
            obs, done, info = env.step(action)
            trajectory.append(obs[:2].copy())
            if done:
                return np.array(trajectory), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--layout", choices=["straight", "wall"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = load_policy(args.ckpt)
    env = PointNavEnv(args.layout)
    trajectory, info = run_episode(policy, env, seed=args.seed)
    print(f"start: {trajectory[0]}, goal: {env.goal}, final: {trajectory[-1]}")
    print(info)


if __name__ == "__main__":
    main()
