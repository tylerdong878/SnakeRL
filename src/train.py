"""
Fast headless training script for SnakeRL using Stable-Baselines3 PPO.

Runs as fast as possible (no rendering, parallel envs, GPU if available).
Saves the model when training ends (or when you stop it with CTRL+C).
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 0 => run indefinitely until interrupted
    parser.add_argument("--total-timesteps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-path", type=str, default="models/snake_ppo")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help='"cpu", "cuda", or "auto"')
    parser.add_argument("--chunk-steps", type=int, default=100_000, help="Learn in chunks when running indefinitely")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output dirs exist
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    # Ensure parent dir for model path exists

    # Vectorized training envs (parallel across CPU cores), headless for speed
    env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render_mode": "none", "max_steps": args.max_steps},
        monitor_dir=args.log_dir,
    )

    # PPO on GPU/CPU (auto), tune a bit for vectorized envs
    model = PPO(
        "MlpPolicy",
        env,
        device=args.device,
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
        n_steps=256,          # per env → total rollout = n_steps * n_envs
        batch_size=256,       # multiple of n_envs recommended
        n_epochs=10,
    )

    try:
        if args.total_timesteps and args.total_timesteps > 0:
            model.learn(total_timesteps=args.total_timesteps)
        else:
            while True:
                model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False)
    except KeyboardInterrupt:
        pass

    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()


