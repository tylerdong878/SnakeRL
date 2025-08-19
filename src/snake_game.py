"""
Snake Game Implementation
A classic Snake game built with Pygame for reinforcement learning.
"""

import pygame
import os
import random
import sys
from typing import List, Tuple, Optional
from enum import Enum


class Direction(Enum):
    """Snake movement directions."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    """Main Snake game class."""
    
    def __init__(self, width: int = 1000, height: int = 1000, grid_size: int = 100, headless: bool = False, show_hud: bool = True):
        """
        Initialize the Snake game.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            grid_size: Size of each grid cell in pixels
            headless: If True, do not create a visible window (for training)
        """
        self.headless = headless
        self.show_hud = show_hud
        
        # Use dummy video driver if headless to avoid opening a window
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        
        pygame.init()
        
        # Game dimensions
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Pygame setup
        if self.headless:
            # Offscreen surface for headless rendering
            self.screen = pygame.Surface((width, height))
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Snake Game - SnakeRL")
            self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.DARK_GREEN = (0, 200, 0)
        
        # Game state
        self.reset()
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        # Snake starts in the middle
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = Direction.RIGHT
        self.last_direction = self.direction
        self.input_queue: List[Direction] = []
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False
        self.fps = 10
    
    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food at random location."""
        while True:
            food = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1)
            )
            if food not in self.snake:
                return food
    
    def _handle_input(self) -> None:
        """Handle keyboard input."""
        if self.headless:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._queue_direction(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self._queue_direction(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self._queue_direction(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self._queue_direction(Direction.RIGHT)
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    def _is_opposite(self, d1: Direction, d2: Direction) -> bool:
        """Return True if directions are opposites."""
        dx1, dy1 = d1.value
        dx2, dy2 = d2.value
        return dx1 == -dx2 and dy1 == -dy2

    def _queue_direction(self, new_direction: Direction) -> None:
        """Queue direction changes; apply one per tick; ignore 180° reversals vs current/queued."""
        reference = self.input_queue[-1] if self.input_queue else self.last_direction
        if self._is_opposite(new_direction, reference):
            return
        self.input_queue.append(new_direction)
    
    def _move_snake(self) -> None:
        """Move the snake in the current direction."""
        if self.game_over:
            return
        
        # Apply at most one queued direction per tick (skip invalid reversals)
        while self.input_queue:
            candidate = self.input_queue.pop(0)
            if not self._is_opposite(candidate, self.last_direction):
                self.direction = candidate
                break

        # Disallow external 180° reversals only when snake length > 1
        if len(self.snake) > 1 and self._is_opposite(self.direction, self.last_direction):
            self.direction = self.last_direction

        # Get current head position
        head_x, head_y = self.snake[0]
        
        # Calculate new head position
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Check for collisions
        if self._check_collision(new_head):
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Update last_direction after a successful move
        self.last_direction = self.direction
    
    def _check_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with walls or snake."""
        x, y = position
        
        # Wall collision
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        
        # Self collision
        if position in self.snake:
            return True
        
        return False
    
    def _draw(self) -> None:
        """Draw the game state."""
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = self.GREEN if i == 0 else self.DARK_GREEN
            rect = pygame.Rect(
                segment[0] * self.grid_size,
                segment[1] * self.grid_size,
                self.grid_size,
                self.grid_size
            )
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Draw food
        food_rect = pygame.Rect(
            self.food[0] * self.grid_size,
            self.food[1] * self.grid_size,
            self.grid_size,
            self.grid_size
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)
        
        if self.show_hud:
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, self.WHITE)
            bg_w = score_text.get_width() + 8
            bg_h = score_text.get_height() + 6
            score_bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
            score_bg.fill((0, 0, 0, 150))
            self.screen.blit(score_bg, (6, 6))
            self.screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, self.RED)
            restart_text = font.render("Press R to restart", True, self.WHITE)
            
            # Center the text
            game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        if not self.headless:
            pygame.display.flip()
    
    def run(self) -> None:
        """Main game loop."""
        print("Snake Game Started!")
        print("Controls: Arrow keys to move, R to restart, ESC to quit")
        
        while True:
            self._handle_input()
            self._move_snake()
            self._draw()
            self.clock.tick(self.fps)


def main():
    """Main function to run the game."""
    game = SnakeGame()
    game.run()


if __name__ == "__main__":
    main()
