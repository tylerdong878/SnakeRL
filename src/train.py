"""
Training script for SnakeRL using Stable-Baselines3 PPO.
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from src.snake_env import SnakeEnv


class RenderEvalCallback(BaseCallback):
    """Periodically render a short evaluation episode in a visible window."""

    def __init__(self, eval_freq: int = 20000, n_episodes: int = 1, max_steps: int = 300):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.eval_env = SnakeEnv(render_mode="human", max_steps=self.max_steps)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            for _ in range(self.n_episodes):
                obs, _ = self.eval_env.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    self.eval_env.render()
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-path", type=str, default="models/snake_ppo")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Training env headless for speed
    env = Monitor(SnakeEnv(render_mode="none", max_steps=args.max_steps))

    model = PPO("MlpPolicy", env, tensorboard_log=args.log_dir, verbose=1, seed=args.seed)

    callback = RenderEvalCallback(eval_freq=args.eval_freq, n_episodes=1, max_steps=300)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()


