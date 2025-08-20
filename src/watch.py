"""
Watch the latest saved Snake PPO model play continuously.

This script reloads the newest checkpoint periodically so you can watch
training progress without slowing training itself.
"""

import argparse
import glob
import os
import time
from typing import Optional

from stable_baselines3 import PPO
from src.snake_env import SnakeEnv


def find_latest_model(path_glob: str) -> Optional[str]:
    candidates = glob.glob(path_glob)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        default="models/checkpoints/snake_ppo_*.zip",
        help="Glob pattern to find checkpoints.",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=1000000)
    # 0 disables truncation so the snake can play indefinitely unless it dies
    parser.add_argument("--max-steps", type=int, default=0)
    # Board size configuration
    parser.add_argument("--width", type=int, default=1000, help="Board width in pixels")
    parser.add_argument("--height", type=int, default=1000, help="Board height in pixels")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid cell size in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SnakeEnv(
        width=args.width,
        height=args.height, 
        grid_size=args.grid_size,
        render_mode="human", 
        max_steps=args.max_steps
    )

    last_loaded: Optional[str] = None
    model: Optional[PPO] = None

    while True:
        latest = find_latest_model(args.pattern)
        if latest and latest != last_loaded:
            print(f"Loading model: {latest}")
            model = PPO.load(latest)
            last_loaded = latest

        if model is None:
            print("Waiting for checkpoints...")
            time.sleep(args.poll_seconds)
            continue

        # Play one episode
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

        # After episode, see if there's a newer model
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()


