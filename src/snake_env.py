"""
Snake Reinforcement Learning Environment
A Gymnasium environment wrapper for the Snake game.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

# Import our Snake game
from .snake_game import SnakeGame, Direction


class SnakeEnv(gym.Env):
    """
    Snake environment for reinforcement learning.
    
    This wraps our SnakeGame class to provide a standard RL interface.
    """
    
    def __init__(self, width: int = 800, height: int = 600, grid_size: int = 20):
        super().__init__()
        
        # Game setup
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Create the actual game instance
        self.game = SnakeGame(width, height, grid_size)
        
        # Define action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Define observation space: [head_x, head_y, food_x, food_y, direction, length]
        # head_x, head_y, food_x, food_y: 0 to grid_size-1
        # direction: 0,1,2,3 (UP,DOWN,LEFT,RIGHT)
        # length: 1 to max possible length
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 1]),
            high=np.array([self.grid_width-1, self.grid_height-1, self.grid_width-1, self.grid_height-1, 3, 100]),
            dtype=np.int32
        )
        
        # TODO: Define reward function
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000  # Prevent infinite episodes
        
        # Store previous distance for efficiency calculation
        self.previous_distance = 0
    
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
        """Convert game state to observation array for the AI."""
        # Get current positions
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        
        # Convert direction to number
        direction_num = self._direction_to_number(self.game.direction)
        
        # Get snake length
        length = len(self.game.snake)
        
        # Return observation array: [head_x, head_y, food_x, food_y, direction, length]
        return np.array([head_x, head_y, food_x, food_y, direction_num, length], dtype=np.int32)
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _is_moving_toward_food(self) -> bool:
        """Check if snake is moving toward or away from food."""
        current_distance = self._calculate_distance(self.game.snake[0], self.game.food)
        
        # If this is the first step, we can't compare
        if self.previous_distance == 0:
            self.previous_distance = current_distance
            return False
        
        # Check if distance decreased (moving toward food)
        moving_toward = current_distance < self.previous_distance
        
        # Update previous distance for next step
        self.previous_distance = current_distance
        
        return moving_toward
    
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
        
        # Reset the game
        self.game.reset()
        
        # Reset episode tracking
        self.step_count = 0
        self.previous_distance = 0
        
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
        
        # Move the snake (this handles collision detection)
        self.game._move_snake()
        
        # Check if episode is done
        done = self.game.game_over or self.step_count >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward()
        
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
        return observation, reward, done, False, info
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        reward = 0.0
        
        # Check if food was eaten this step
        if len(self.game.snake) > 1:  # Snake grew (ate food)
            reward += 10.0  # Big reward for eating food
        
        # Efficiency penalty (every step)
        reward -= 0.1
        
        # Check if game over
        if self.game.game_over:
            reward -= 10.0  # Big penalty for dying
        
        return reward
    
    def render(self):
        """Render the current game state."""
        self.game._draw()
    
    def close(self):
        """Clean up resources."""
        pygame.quit()
