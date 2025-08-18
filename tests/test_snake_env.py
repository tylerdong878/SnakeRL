import numpy as np
from src.snake_env import SnakeEnv


def test_reset_and_observation_range():
    env = SnakeEnv(width=200, height=200, grid_size=20, render_mode="none", max_steps=50)
    obs, info = env.reset(seed=123)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    env.close()


def test_step_terminated_vs_truncated():
    env = SnakeEnv(width=100, height=100, grid_size=20, render_mode="none", max_steps=3)
    env.reset(seed=0)
    # Take 3 steps: should truncate even if not dead
    for _ in range(2):
        env.step(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated is True
    env.close()


def test_seeding_determinism():
    env1 = SnakeEnv(width=200, height=200, grid_size=20, render_mode="none", max_steps=50)
    env2 = SnakeEnv(width=200, height=200, grid_size=20, render_mode="none", max_steps=50)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.allclose(obs1, obs2)
    env1.close()
    env2.close()


def test_reward_signals():
    env = SnakeEnv(width=200, height=200, grid_size=20, render_mode="none", max_steps=50)
    env.reset(seed=1)
    # Take one step; ensure step penalty applied
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert reward <= 0.0
    env.close()


