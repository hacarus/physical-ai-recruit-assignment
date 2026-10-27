"""PPO による学習スクリプト.

使い方:
    uv run python ppo.py --layout wall --out checkpoints/ppo_wall.pt --log logs/wall.csv
"""

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from env import VecEnv

OBS_DIM = 4
ACTION_DIM = 2


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """方策: 状態に依存しない標準偏差を持つ対角ガウス分布. 価値関数: 別の MLP."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(OBS_DIM, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, ACTION_DIM), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(ACTION_DIM))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        dist = Normal(mean, self.actor_logstd.exp().expand_as(mean))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.get_value(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """評価用. deterministic=True ならガウス分布の平均を返す."""
        mean = self.actor_mean(obs)
        if deterministic:
            return mean
        return Normal(mean, self.actor_logstd.exp()).sample()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["straight", "wall"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True, help="エピソードごとの報酬合計を書き出す CSV")
    parser.add_argument("--total-steps", type=int, default=400_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128, help="1 回の更新で各環境から集めるステップ数")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    envs = VecEnv(args.layout, args.num_envs, seed=args.seed)
    agent = ActorCritic()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    N, T = args.num_envs, args.num_steps
    batch_size = N * T
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_steps // batch_size

    obs_buf = torch.zeros(T, N, OBS_DIM)
    act_buf = torch.zeros(T, N, ACTION_DIM)
    logp_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)

    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "global_step", "return"])
    episode_returns = np.zeros(N)
    episode_count = 0
    global_step = 0
    start_time = time.time()

    next_obs = torch.tensor(envs.reset())
    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.lr

        # ---- ロールアウトの収集 ----
        for step in range(T):
            global_step += N
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            obs, reward, terminated, truncated, infos = envs.step(action.numpy())
            episode_returns += reward
            reward = reward.astype(np.float32)
            # 時間切れによる打ち切りでは、打ち切られた後の価値でブートストラップする
            for i in np.where(truncated)[0]:
                with torch.no_grad():
                    final_value = agent.get_value(torch.tensor(infos[i]["final_obs"])).item()
                reward[i] += args.gamma * final_value
            rew_buf[step] = torch.tensor(reward)
            done_buf[step] = torch.tensor(terminated | truncated, dtype=torch.float32)
            next_obs = torch.tensor(obs)

            for i in np.where(terminated | truncated)[0]:
                episode_count += 1
                writer.writerow([episode_count, global_step, f"{episode_returns[i]:.4f}"])
                episode_returns[i] = 0.0

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
        advantages = torch.zeros(T, N)
        last_gae = torch.zeros(N)
        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * v_next * nonterminal - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # ---- 方策と価値関数の更新 ----
        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1, ACTION_DIM)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        for _ in range(args.update_epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start : start + minibatch_size]
                _, new_logp, entropy, new_value = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((new_value - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 20 == 0:
            print(
                f"update {update}/{num_updates}  step {global_step}  episodes {episode_count}  "
                f"v_loss {v_loss.item():.4f}  std {agent.actor_logstd.exp().detach().numpy()}  "
                f"{time.time() - start_time:.0f}s"
            )

    log_file.close()
    torch.save(agent.state_dict(), args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
