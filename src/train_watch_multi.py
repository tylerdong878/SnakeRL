"""
Train with live multi-screen visualization.

Creates N parallel environments (training) and renders all of them into a grid
at normal game speed while PPO learns. Saves the model on exit (CTRL+C).
"""

import argparse
import math
import os
from typing import List

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=16, help="number of parallel games to show")
    parser.add_argument("--cols", type=int, default=0, help="columns in the grid; 0=auto")
    parser.add_argument("--tile-width", type=int, default=200)
    parser.add_argument("--tile-height", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-path", type=str, default="models/snake_ppo_multi")
    parser.add_argument("--fps", type=int, default=10, help="visualization FPS")
    parser.add_argument("--chunk-steps", type=int, default=50000)
    return parser.parse_args()


class MultiGridRenderCallback(BaseCallback):
    def __init__(self, n_envs: int, cols: int, tile_w: int, tile_h: int, fps: int):
        super().__init__()
        self.n_envs = n_envs
        self.cols = cols if cols > 0 else int(math.ceil(math.sqrt(n_envs)))
        self.rows = int(math.ceil(n_envs / self.cols))
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.fps = fps
        pygame.init()
        self.screen = pygame.display.set_mode((self.cols * tile_w, self.rows * tile_h))
        pygame.display.set_caption("SnakeRL - Live Multi Training")
        self.clock = pygame.time.Clock()

    def _on_step(self) -> bool:
        # Handle window events; allow closing with the X button
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt
        except Exception:
            pass

        # Request frames from all envs (returns list of np arrays HxWx3)
        frames: List[np.ndarray] = self.training_env.env_method("render")  # type: ignore[attr-defined]

        # Draw each frame into the grid
        self.screen.fill((0, 0, 0))
        for idx, frame in enumerate(frames):
            if frame is None:
                continue
            row = idx // self.cols
            col = idx % self.cols
            # Pygame expects array shape (W,H,3)
            surf = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
            if surf.get_width() != self.tile_w or surf.get_height() != self.tile_h:
                surf = pygame.transform.scale(surf, (self.tile_w, self.tile_h))
            self.screen.blit(surf, (col * self.tile_w, row * self.tile_h))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _on_training_end(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    # Build vectorized envs that render rgb arrays (headless windows)
    env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "width": args.tile_width,
            "height": args.tile_height,
            "grid_size": args.grid_size,
            "render_mode": "rgb_array",
            "max_steps": args.max_steps,
        },
        monitor_dir=args.log_dir,
    )

    model = PPO(
        "MlpPolicy",
        env,
        device=args.device,
        tensorboard_log=args.log_dir,
        verbose=1,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
    )

    callback = MultiGridRenderCallback(
        n_envs=args.n_envs,
        cols=args.cols,
        tile_w=args.tile_width,
        tile_h=args.tile_height,
        fps=args.fps,
    )

    try:
        while True:
            model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False, callback=callback)
    except KeyboardInterrupt:
        pass

    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()


