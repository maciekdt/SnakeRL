import pytest
import numpy as np
from environment.snake_logic import Direction, SnakeLogic
from environment.snake_env import SnakeEnv 

def test_initialization():
    env = SnakeEnv()
    assert env.action_space.n == 4
    assert env.observation_space.shape == (3, 15, 15)
    assert env.steps_in_episode_counter == 0
    
def test_reset():
    env = SnakeEnv()
    observation = env.reset()
    
    assert observation.shape == (3, 15, 15)
    assert not np.array_equal(observation[0], np.zeros((15, 15), dtype=np.int8))
    assert np.array_equal(observation[1], observation[2])
    assert env.steps_in_episode_counter == 0
    

def test_step():
    env = SnakeEnv()
    env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
    
    assert observation.shape == (3, 15, 15)
    assert isinstance(reward, int)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    assert reward in [-1, 10, -10]
    assert not terminated or reward == -10
    

def test_episode_termination():
    env = SnakeEnv()
    env.reset()
    
    terminated = False
    reward = None
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            break
    assert terminated
    assert reward == -10
    
def test_episode_termination():
    env = SnakeEnv()
    env.reset()
    
    observation, reward, terminated, truncated, info = env.step(0)
    assert not terminated
    assert not truncated
    assert reward == -1 or reward == 10