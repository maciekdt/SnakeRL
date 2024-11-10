from src.model.DQN_model import dqn_model, eval_callback
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

dqn_model.learn(
    total_timesteps = 1_000_000,
    progress_bar = True,
    callback = eval_callback,
    log_interval = 100_000
)
dqn_model.save("saved_models/dqn_snake")