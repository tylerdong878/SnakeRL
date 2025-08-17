"""
Snake Reinforcement Learning Environment
A Gymnasium environment wrapper for the Snake game.
"""

import gymnasium as gym
import numpy as np
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
        
        # TODO: Define action space
        # TODO: Define observation space
        # TODO: Define reward function
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000  # Prevent infinite episodes
