"""
Evaluation script for a trained SnakeRL PPO model.
"""

import argparse
import os
from stable_baselines3 import PPO
from src.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/snake_ppo.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    render_mode = "human" if args.render else "none"
    env = SnakeEnv(render_mode=render_mode, max_steps=args.max_steps)

    if not os.path.exists(args.model_path):
        # Try without .zip if user passed folder-like path saved by SB3
        alt = args.model_path.rstrip(".zip")
        model = PPO.load(alt)
    else:
        model = PPO.load(args.model_path)

    total_score = 0
    total_length = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render:
                env.render()
        total_score += info["score"]
        total_length += info["snake_length"]
        print(f"Episode {ep+1}: score={info['score']} length={info['snake_length']}")

    print(f"Average score over {args.episodes} eps: {total_score/args.episodes:.2f}")
    print(f"Average length over {args.episodes} eps: {total_length/args.episodes:.2f}")
    env.close()


if __name__ == "__main__":
    main()


