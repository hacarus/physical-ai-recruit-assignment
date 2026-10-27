"""学習曲線 (エピソードごとの報酬合計) を描画する.

使い方:
    uv run python plot_learning_curve.py logs/straight.csv logs/wall.csv --out outputs/learning_curve.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["global_step"]) for r in rows])
    returns = np.array([float(r["return"]) for r in rows])
    return steps, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--window", type=int, default=100, help="移動平均の窓幅 (エピソード数)")
    parser.add_argument("--out", default="outputs/learning_curve.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in args.logs:
        steps, returns = load_log(path)
        smoothed = np.convolve(returns, np.ones(args.window) / args.window, mode="valid")
        ax.plot(steps[args.window - 1 :], smoothed, label=Path(path).stem)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return (moving average, {args.window} episodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
