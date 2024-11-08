import os
import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.snake_env import SnakeEnv

class VideoCreator:
    def __init__(self, model_path, env_class=SnakeEnv, video_path="output_video.mp4", fps=5):
        self.model_path = model_path
        self.env_class = env_class
        self.video_path = video_path
        self.fps = fps

        self.model = DQN.load(model_path)
        self.env = DummyVecEnv([lambda: env_class(render_mode="rgb_array")])
        
    def create_video(self, num_steps=1000):
        obs = self.env.reset()
        
        frame = self.env.render(mode="rgb_array")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (15*8, 15*8))
        
        for step in range(num_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)
            frame = self.env.render(mode="rgb_array")
            scaled_frame = cv2.resize(frame, (frame.shape[1] * 8, frame.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
            video_writer.write(cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR))    
            if done:
                break

        video_writer.release()
        self.env.close()
        print(f"Video saved to {self.video_path}")

video_creator = VideoCreator(model_path="dqn_snake.zip", video_path="snake_game_video.mp4", fps=10)
video_creator.create_video(num_steps=500)
