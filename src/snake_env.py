"""
Snake Reinforcement Learning Environment
A Gymnasium environment wrapper for the Snake game.
"""

import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# Import our Snake game
from .snake_game import SnakeGame, Direction


class SnakeEnv(gym.Env):
    """
    Snake environment for reinforcement learning.
    
    This wraps our SnakeGame class to provide a standard RL interface.
    """
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 10}
    
    def __init__(self, width: int = 1000, height: int = 1000, grid_size: int = 100, render_mode: Optional[str] = "human", max_steps: int = 1000):
        super().__init__()
        
        # Game setup
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        self.render_mode = render_mode if render_mode in {"human", "rgb_array", "none", None} else "human"
        
        # Create the actual game instance (headless for non-human modes)
        headless = self.render_mode != "human"
        # In rgb_array mode (used by multi-screen), hide HUD for clean tiles
        show_hud = self.render_mode == "human"
        self.game = SnakeGame(width, height, grid_size, headless=headless, show_hud=show_hud)
        
        # Define action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Define normalized observation space (float32 in [0,1])
        # [head_x, head_y, food_x, food_y, direction, length]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = max_steps  # Prevent infinite episodes
    
    def _direction_to_number(self, direction: Direction) -> int:
        """Convert Direction enum to number for the AI."""
        if direction == Direction.UP:
            return 0
        elif direction == Direction.DOWN:
            return 1
        elif direction == Direction.LEFT:
            return 2
        elif direction == Direction.RIGHT:
            return 3
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to normalized float32 observation array for the AI."""
        # Current positions
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        
        # Direction and length
        direction_num = self._direction_to_number(self.game.direction)
        length = len(self.game.snake)
        
        # Normalization denominators (avoid division by zero)
        max_x = max(1, self.grid_width - 1)
        max_y = max(1, self.grid_height - 1)
        max_dir = 3
        max_len = max(1, self.grid_width * self.grid_height)
        
        obs = np.array([
            head_x / max_x,
            head_y / max_y,
            food_x / max_x,
            food_y / max_y,
            direction_num / max_dir,
            length / max_len,
        ], dtype=np.float32)
        return obs
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    
    
    def _number_to_direction(self, action: int) -> Direction:
        """Convert AI action number back to Direction enum."""
        if action == 0:
            return Direction.UP
        elif action == 1:
            return Direction.DOWN
        elif action == 2:
            return Direction.LEFT
        elif action == 3:
            return Direction.RIGHT
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, or 3.")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Seed RNGs for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset the game
        self.game.reset()
        
        # Reset episode tracking
        self.step_count = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Return observation and empty info dict
        return observation, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment based on the AI's action."""
        # Increment step counter
        self.step_count += 1
        
        # Convert AI action to game direction
        direction = self._number_to_direction(action)
        
        # Update game direction
        self.game.direction = direction
        
        # Track score before moving to detect food consumption on this step
        previous_score = self.game.score
        
        # Move the snake (this handles collision detection)
        self.game._move_snake()
        
        # Check episode status per Gymnasium API
        terminated = self.game.game_over
        truncated = self.step_count >= self.max_steps
        
        # Determine if food was eaten this step
        food_eaten = self.game.score > previous_score
        
        # Calculate reward
        reward = self._calculate_reward(food_eaten)
        
        # Get current observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'score': self.game.score,
            'snake_length': len(self.game.snake),
            'steps': self.step_count,
            'distance_to_food': self._calculate_distance(self.game.snake[0], self.game.food)
        }
        
        # Return: observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, food_eaten: bool) -> float:
        """Calculate the reward for the current state."""
        reward = 0.0
        
        # Reward for eating food on this step
        if food_eaten:
            reward += 10.0
        
        # Efficiency penalty (every step)
        reward -= 0.1
        
        # Check if game over
        if self.game.game_over:
            reward -= 10.0
        
        return reward
    
    def render(self):
        """Render the current game state according to render_mode."""
        if self.render_mode in ("human", None):
            # Handle window events; allow closing with the X button
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise KeyboardInterrupt
            except Exception:
                pass
            self.game._draw()
            # Cap to game FPS for normal-speed visualization
            try:
                self.game.clock.tick(self.game.fps)
            except Exception:
                pass
            return None
        
        if self.render_mode == "rgb_array":
            # Draw first to ensure the frame is current and handle events
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        raise KeyboardInterrupt
            except Exception:
                pass
            self.game._draw()
            frame = pygame.surfarray.array3d(self.game.screen)
            # Convert from (W, H, 3) to (H, W, 3)
            return np.transpose(frame, (1, 0, 2))
        
        # "none" mode or unrecognized mode: no rendering
        return None
    
    def close(self):
        """Clean up resources."""
        pygame.quit()
