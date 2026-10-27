"""課題 1 のモデルを学習するスクリプト.

使い方:
    uv run python train.py --model cnn --dataset data/wall.npz --out checkpoints/cnn_wall.pt
"""

import argparse

import numpy as np
import torch

from models import HORIZON, MODELS


def load_dataset(path: str):
    """各時刻 t について (観測 o_t, 行動列 a_t, ..., a_{t+HORIZON-1}) の組を作る.

    エピソード末尾を越える部分は、エピソード最後の行動を繰り返して埋める.
    """
    data = np.load(path)
    obs, action, ends = data["obs"], data["action"], data["episode_ends"]
    obs_list, chunk_list = [], []
    start = 0
    for end in ends:
        ep_obs, ep_act = obs[start:end], action[start:end]
        padded = np.concatenate([ep_act, np.repeat(ep_act[-1:], HORIZON - 1, axis=0)])
        for t in range(len(ep_obs)):
            obs_list.append(ep_obs[t])
            chunk_list.append(padded[t : t + HORIZON])
        start = end
    return torch.tensor(np.array(obs_list)), torch.tensor(np.array(chunk_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    obs, actions = load_dataset(args.dataset)
    obs, actions = obs.to(args.device), actions.to(args.device)
    print(f"samples: {len(obs)}")

    model = MODELS[args.model]().to(args.device)
    model.obs_normalizer.fit(obs)
    model.action_normalizer.fit(actions.reshape(-1, actions.shape[-1]))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(obs), (args.batch_size,), device=args.device)
        loss = model.compute_loss(obs[idx], actions[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0:
            print(f"iter {it:6d}  loss {loss.item():.5f}")

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model_type": args.model, "state_dict": state_dict}, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
