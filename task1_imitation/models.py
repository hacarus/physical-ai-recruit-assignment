"""課題 1 で使用する方策モデル.

どちらのモデルも、観測 (x, y, goal_x, goal_y) を条件として
今後 HORIZON ステップ分の行動列 (HORIZON, 2) を出力する.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 4
ACTION_DIM = 2
HORIZON = 16  # 1 回の推論で出力する行動の長さ
N_ACTION_STEPS = 8  # 出力した行動列のうち、実際に環境で実行する長さ


# --------------------------------------------------------------------------- #
# 正規化
# --------------------------------------------------------------------------- #
class Normalizer(nn.Module):
    """データセットの min / max を用いて各次元を [-1, 1] に線形変換する."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("min", torch.zeros(dim))
        self.register_buffer("max", torch.ones(dim))

    def fit(self, x: torch.Tensor):
        x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
        # 値が一定の次元は 0 に写す
        const = (x_max - x_min) < 1e-6
        self.min.copy_(torch.where(const, x_min - 1.0, x_min))
        self.max.copy_(torch.where(const, x_max + 1.0, x_max))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.min) / (self.max - self.min) - 1.0

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) / 2.0 * (self.max - self.min) + self.min


# --------------------------------------------------------------------------- #
# CNN 方策 (MSE 回帰)
# --------------------------------------------------------------------------- #
class CNNPolicy(nn.Module):
    """観測から行動列を直接回帰する 1D-CNN.

    観測を MLP で (channels, HORIZON // 4) の特徴列に変換し、
    1D 転置畳み込みで時間方向に 4 倍アップサンプリングして (HORIZON, 2) を得る.
    """

    def __init__(self, hidden: int = 256, channels: int = 64):
        super().__init__()
        self.obs_normalizer = Normalizer(OBS_DIM)
        self.action_normalizer = Normalizer(ACTION_DIM)
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, channels * (HORIZON // 4)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.Mish(),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.Mish(),
            nn.Conv1d(channels, ACTION_DIM, kernel_size=3, padding=1),
        )

    def forward(self, nobs: torch.Tensor) -> torch.Tensor:
        """正規化済み観測 (B, OBS_DIM) -> 正規化済み行動列 (B, HORIZON, ACTION_DIM)."""
        h = self.encoder(nobs).view(-1, self.channels, HORIZON // 4)
        return self.decoder(h).transpose(1, 2)

    def compute_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        nobs = self.obs_normalizer.normalize(obs)
        naction = self.action_normalizer.normalize(action)
        return F.mse_loss(self(nobs), naction)

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """観測 (B, OBS_DIM) -> 行動列 (B, HORIZON, ACTION_DIM)."""
        nobs = self.obs_normalizer.normalize(obs)
        return self.action_normalizer.unnormalize(self(nobs))


# --------------------------------------------------------------------------- #
# DiT 方策 (拡散モデル)
# --------------------------------------------------------------------------- #
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """adaLN-Zero で条件付けされた Transformer ブロック."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, g1, a1, s2, g2, a2 = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, g1)
        x = x + a1[:, None] * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s2, g2)
        return x + a2[:, None] * self.mlp(h)


class DiTPolicy(nn.Module):
    """ノイズの乗った行動列からノイズを予測する Diffusion Transformer.

    学習: DDPM (ノイズ予測、MSE)
    推論: 標準正規分布からサンプルした行動列を N_DIFFUSION_STEPS 回デノイズする
    """

    N_DIFFUSION_STEPS = 50

    def __init__(self, dim: int = 128, depth: int = 4, n_heads: int = 4):
        super().__init__()
        self.obs_normalizer = Normalizer(OBS_DIM)
        self.action_normalizer = Normalizer(ACTION_DIM)
        self.dim = dim
        self.action_in = nn.Linear(ACTION_DIM, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, HORIZON, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.obs_mlp = nn.Sequential(nn.Linear(OBS_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.action_out = nn.Linear(dim, ACTION_DIM)
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

        # squared cosine noise schedule
        T = self.N_DIFFUSION_STEPS
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, 0).float())

    def forward(self, noisy_action: torch.Tensor, t: torch.Tensor, nobs: torch.Tensor) -> torch.Tensor:
        """(B, HORIZON, ACTION_DIM), (B,), (B, OBS_DIM) -> 予測ノイズ (B, HORIZON, ACTION_DIM)."""
        x = self.action_in(noisy_action) + self.pos_emb
        c = self.t_mlp(timestep_embedding(t, self.dim)) + self.obs_mlp(nobs)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.action_out(modulate(self.final_norm(x), shift, scale))

    def compute_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        nobs = self.obs_normalizer.normalize(obs)
        x0 = self.action_normalizer.normalize(action)
        B = x0.shape[0]
        t = torch.randint(0, self.N_DIFFUSION_STEPS, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alphas_cumprod[t][:, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(self(xt, t, nobs), noise)

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """観測 (B, OBS_DIM) -> 行動列 (B, HORIZON, ACTION_DIM). 呼ぶたびに異なるサンプルが得られる."""
        nobs = self.obs_normalizer.normalize(obs)
        B = obs.shape[0]
        x = torch.randn(B, HORIZON, ACTION_DIM, device=obs.device)
        for i in reversed(range(self.N_DIFFUSION_STEPS)):
            t = torch.full((B,), i, device=obs.device, dtype=torch.long)
            eps = self(x, t, nobs)
            a, ab, b = self.alphas[i], self.alphas_cumprod[i], self.betas[i]
            ab_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=obs.device)
            # x_0 を推定して [-1, 1] にクリップし、事後分布 q(x_{t-1} | x_t, x_0) の平均を計算する
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = b * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return self.action_normalizer.unnormalize(x)


MODELS = {"cnn": CNNPolicy, "dit": DiTPolicy}


def load_policy(path: str, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = MODELS[ckpt["model_type"]]()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
