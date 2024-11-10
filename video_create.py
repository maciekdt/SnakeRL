from src.video_creator.video_creator import VideoCreator
from src.model.DQN_model import dqn_model
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

video_creator = VideoCreator(
    model_path="saved_models/dqn_snake_40M.zip",
    video_path="videos/snake_game_video.mp4",
    fps=20
)
video_creator.create_video(num_steps=500)