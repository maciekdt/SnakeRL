from src.video_creator.video_creator import VideoCreator
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

video_creator = VideoCreator(
    model_path="logs/saved_models/best_checkpoints/best_model.zip",
    video_path="logs/log_videos/snake_game_video.mp4",
    fps=20
)
video_creator.create_video(num_steps=500)