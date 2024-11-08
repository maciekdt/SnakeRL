import pytest
from environment.snake_logic import SnakeLogic, Direction  # Ensure the import path is correct

def test_initialization():
    game = SnakeLogic()
    
    assert game.snake_list == [(7, 7)]
    assert game.apple_position not in game.snake_list
    assert game.apple_position[0] >= 0 and game.apple_position[0] < game.BOARD_SIZE
    assert game.apple_position[1] >= 0 and game.apple_position[1] < game.BOARD_SIZE
    assert not game.game_over
    

def test_step_up_with_apple():
    game = SnakeLogic()
    game.apple_position = (8,8)
    game.snake_list = [(6,6), (6,7), (7,7), (8,7)]
    
    snake_list, apple_position, game_over, got_apple = game.step(Direction.UP)
    expected_result = [(6,7), (7,7), (8,7), (8,8)]
    
    assert game.snake_list == expected_result
    assert snake_list == expected_result
    assert got_apple
    assert apple_position != (8,8)
    assert game.apple_position != (8,8)
    assert not game_over
    assert not game.game_over
    assert game.current_direction == Direction.UP
    

def test_back_step():
    game = SnakeLogic()
    game.apple_position = (8,8)
    game.snake_list = [(6,6), (6,7)]
    game.current_direction = Direction.UP
    
    snake_list, apple_position, game_over, got_apple = game.step(Direction.DOWN)
    expected_result = [(6,7), (6,8)]
    
    assert game.snake_list == expected_result
    assert snake_list == expected_result
    assert not got_apple
    assert apple_position == (8,8)
    assert game.apple_position == (8,8)
    assert not game_over
    assert not game.game_over
    assert game.current_direction == Direction.UP
    
    
def test_out_of_board_game_over():
    game = SnakeLogic()
    game.apple_position = (8,8)
    game.snake_list = [(13,6), (14,6)]
    
    snake_list, apple_position, game_over, got_apple = game.step(Direction.RIGHT)
    
    assert game_over
    assert game.game_over
    assert snake_list == [(14,6)]
    

def test_snake_self_colision_game_over():
    game = SnakeLogic()
    game.apple_position = (8,8)
    game.snake_list = [(7,6), (6,6), (6,5), (6,4), (6,3), (5,3), (4,3), (4,4), (4,5), (5,5)]
    
    snake_list, apple_position, game_over, got_apple = game.step(Direction.RIGHT)
    
    assert game_over
    assert game.game_over
    assert snake_list == [(6,6), (6,5), (6,4), (6,3), (5,3), (4,3), (4,4), (4,5), (5,5)]
