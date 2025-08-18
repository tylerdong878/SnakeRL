"""
Evaluation script for a trained SnakeRL PPO model.
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    # Wrap env to support VecNormalize stats if present
    def make_env():
        return SnakeEnv(render_mode=render_mode, max_steps=args.max_steps)
    env = DummyVecEnv([make_env])

    if not os.path.exists(args.model_path):
        # Try without .zip if user passed folder-like path saved by SB3
        alt = args.model_path.rstrip(".zip")
        model = PPO.load(alt)
    else:
        model = PPO.load(args.model_path)

    # Try to load VecNormalize stats if they exist alongside the model
    vecnorm_path = os.path.join(os.path.dirname(args.model_path), "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    total_score = 0
    total_length = 0

    for ep in range(args.episodes):
        obs = env.reset()
        terminated = truncated = False
        episode_score = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, infos = env.step(action)
            # Unwrap info for single env
            info = infos[0] if isinstance(infos, list) else infos
            episode_score = info.get("score", episode_score)
            if args.render:
                env.render(mode="human")
        total_score += episode_score
        # snake_length is not easily available through VecNormalize/DummyVecEnv infos at the end; print score only
        print(f"Episode {ep+1}: score={episode_score}")

    print(f"Average score over {args.episodes} eps: {total_score/args.episodes:.2f}")
    print(f"Average length over {args.episodes} eps: {total_length/args.episodes:.2f}")
    env.close()


if __name__ == "__main__":
    main()


