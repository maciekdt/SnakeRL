import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from src.environment.snake_logic import Direction, SnakeLogic

class SnakeEnv(gym.Env):
    int_to_direction = {
        0: Direction.UP,
        1: Direction.DOWN,
        2: Direction.LEFT,
        3: Direction.RIGHT
    }
    
    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 15, 15), dtype=np.int8)
        self.snake_engine = SnakeLogic()
        self.last_snake_array = None
        self.last_apple_array = None
        self.steps_in_episode_counter = 0
        self.render_mode = render_mode
        
    def step(self, action):
        snake_list, apple_position, game_over, got_apple = self.snake_engine.step(self.int_to_direction[action])
        snake_array, apple_array = self._transform_observation(snake_list, apple_position)
        observation = np.stack((apple_array, self.last_snake_array, snake_array), axis=0)
        self.last_snake_array = snake_array
        self.last_apple_array = apple_array
        self.steps_in_episode_counter += 1
        reward = -0.001
        if got_apple:
            reward = 1
        if game_over:
            reward = -1
        terminated = game_over 
        truncated = False
        if self.steps_in_episode_counter > 1000:
            truncated = True
            reward = -1
        info = {}
        
        return observation, reward, terminated, truncated, info
        
    
    def reset(self, seed=None, options=None):
        self.snake_engine = SnakeLogic()
        self.steps_in_episode_counter = 0
        snake_array, apple_array = self._transform_observation(
            self.snake_engine.snake_list,
            self.snake_engine.apple_position
        )
        self.last_snake_array = snake_array
        self.last_apple_array = apple_array
        return np.stack((apple_array, snake_array, snake_array), axis=0), {}
    
    
    def render(self, mode='console'):
        if self.render_mode == "rgb_array":
            print(len(self.snake_engine.snake_list))
            red_channel = (self.last_apple_array * 255).astype(np.uint8)
            green_channel = (self.last_snake_array * 255).astype(np.uint8)
            blue_channel = np.zeros((15, 15), dtype=np.uint8)
            rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
            return rgb_image
        else:
            pass
    
    def close (self):
        pass
    
    def _transform_observation(self, snake_list, apple_position):
        snake_array = np.zeros((15, 15), dtype=np.int8)
        for el in snake_list:
            snake_array[el[1]][el[0]] = 1
        apple_array = np.zeros((15, 15), dtype=np.int8)
        apple_array[apple_position[1]][apple_position[0]] = 1
        return snake_array, apple_array
    

    