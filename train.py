from src.model.DQN_model import eval_callback, get_dqn_model
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

get_dqn_model().learn(
    total_timesteps = 100_000,
    progress_bar = True,
    callback = eval_callback,
    log_interval = 10_000
)