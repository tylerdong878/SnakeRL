"""
Training-with-visualization script for SnakeRL.

Runs a single visible environment at normal game speed while training continuously.
Saves the model on exit (CTRL+C).
"""

import argparse
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/snake_ppo")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="logs")
    # 0 => run indefinitely; otherwise stop after this many timesteps
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--chunk-steps", type=int, default=10_000, help="Chunk size when running indefinitely")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    # Visible env at normal speed; SB3 will drive the env, we render each step via callback
    env = SnakeEnv(render_mode="human", max_steps=args.max_steps)

    class LiveRenderCallback(BaseCallback):
        def _on_step(self) -> bool:
            # Render current frame; env.render ticks to normal FPS
            # Use env_method to avoid VecEnv render signature issues
            self.training_env.env_method("render")
            return True

    model = PPO("MlpPolicy", env, tensorboard_log=args.log_dir, device=args.device, verbose=1)

    try:
        if args.total_timesteps and args.total_timesteps > 0:
            model.learn(total_timesteps=args.total_timesteps, callback=LiveRenderCallback())
        else:
            while True:
                model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False, callback=LiveRenderCallback())
    except KeyboardInterrupt:
        pass

    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()


