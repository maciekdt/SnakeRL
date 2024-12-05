from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.environment.snake_env import SnakeEnv
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_eval_callback(eval_freq = 1_000_000):
    return EvalCallback(
        eval_env = Monitor(SnakeEnv()),
        n_eval_episodes = 20,
        deterministic = True,
        render = False,
        verbose = 1,
        eval_freq = eval_freq,
        best_model_save_path = os.path.join(
            base_dir,
            "logs/saved_models/best_checkpoints"
        )
    )