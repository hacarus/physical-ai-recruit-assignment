"""2 次元平面上を点が移動するナビゲーション環境 (課題 1 用)."""

from __future__ import annotations

import numpy as np

# 環境の定数
ARENA_MIN = -1.0  # 外枠 (x, y とも [ARENA_MIN, ARENA_MAX])
ARENA_MAX = 1.0
DT = 0.05  # 1 ステップあたりの時間間隔 [s]
MAX_SPEED = 0.6  # 速度ベクトルのノルムの上限
GOAL_RADIUS = 0.1  # goal からこの距離以内に入れば到達とみなす
MAX_EPISODE_STEPS = 200

START_CENTER = (-0.75, 0.0)
GOAL_CENTER = (0.75, 0.0)
START_GOAL_Y_RANGE = 0.1  # start / goal の y 座標は中心から [-0.1, 0.1] の範囲で一様に選ばれる

# "wall" レイアウトの壁 (軸並行の長方形): (x_min, x_max, y_min, y_max)
WALL_RECT = (-0.3, 0.3, -0.6, 0.6)

LAYOUTS = ("straight", "wall")


class PointNavEnv:
    """点を start から goal まで移動させる環境.

    観測: np.array([x, y, goal_x, goal_y])
    行動: np.array([vx, vy]) (ノルムが MAX_SPEED を超える場合は MAX_SPEED に縮められる)
    """

    def __init__(self, layout: str = "straight", seed: int | None = None):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout}")
        self.layout = layout
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2)
        self.goal = np.zeros(2)
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.array(START_CENTER) + [0.0, self.rng.uniform(-START_GOAL_Y_RANGE, START_GOAL_Y_RANGE)]
        self.goal = np.array(GOAL_CENTER) + [0.0, self.rng.uniform(-START_GOAL_Y_RANGE, START_GOAL_Y_RANGE)]
        self.t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        """1 ステップ進める.

        Returns:
            obs: 次の観測
            done: エピソードが終了したか
            info: {"success": goal に到達したか, "collision": 壁に衝突したか, "t": 経過ステップ数}
        """
        vel = clip_speed(np.asarray(action, dtype=np.float64))
        self.pos = np.clip(self.pos + vel * DT, ARENA_MIN, ARENA_MAX)
        self.t += 1

        collision = self.layout == "wall" and in_wall(self.pos)
        success = (not collision) and np.linalg.norm(self.pos - self.goal) < GOAL_RADIUS
        done = collision or success or self.t >= MAX_EPISODE_STEPS
        info = {"success": bool(success), "collision": bool(collision), "t": self.t}
        return self._obs(), done, info

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal]).astype(np.float32)


def clip_speed(vel: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(vel)
    if speed > MAX_SPEED:
        vel = vel * (MAX_SPEED / speed)
    return vel


def in_wall(pos: np.ndarray) -> bool:
    x_min, x_max, y_min, y_max = WALL_RECT
    return bool(x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max)
