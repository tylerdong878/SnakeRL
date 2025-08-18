import pytest

from src.snake_game import SnakeGame, Direction


def make_game():
    # Small grid for deterministic tests, headless to avoid window
    return SnakeGame(width=200, height=200, grid_size=20, headless=True)


def test_reset_initial_state():
    game = make_game()
    game.reset()
    assert len(game.snake) == 1
    assert game.game_over is False
    assert game.score == 0
    # Food is in-bounds and not on snake
    fx, fy = game.food
    assert 0 <= fx < game.grid_width
    assert 0 <= fy < game.grid_height
    assert game.food not in game.snake


def test_growth_on_food():
    game = make_game()
    game.reset()
    hx, hy = game.snake[0]
    # Place food directly to the right and move RIGHT
    game.food = (hx + 1, hy)
    game.direction = Direction.RIGHT
    game._move_snake()
    assert len(game.snake) == 2
    assert game.score == 1


def test_wall_collision_sets_game_over():
    game = make_game()
    game.reset()
    # Place head at left wall and move LEFT
    game.snake = [(0, game.grid_height // 2)]
    game.direction = Direction.LEFT
    game._move_snake()
    assert game.game_over is True


def test_self_collision_sets_game_over():
    game = make_game()
    game.reset()
    # Arrange snake so that moving UP would move into its own body at (2,1)
    game.snake = [(2, 2), (2, 1), (2, 0)]
    game.direction = Direction.UP
    game._move_snake()
    assert game.game_over is True


def test_food_never_on_snake():
    game = make_game()
    game.reset()
    # Grow the snake a bit and check new food spawns not on snake
    game.snake = [(2, 2), (1, 2), (1, 1)]
    for _ in range(10):
        food = game._spawn_food()
        assert food not in game.snake


