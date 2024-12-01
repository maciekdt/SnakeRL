from src.model.DQN_model import eval_callback, get_dqn_model
import torch
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using CPU")

"params_file = logs/optuna_logs/DQN/best_params_dqn.json"
"""with open(params_file, "r") as f:
    params = json.load(f)
    print("Optune hyperparams loaded")"""
    
get_dqn_model().learn(
    total_timesteps = 100_000_000,
    progress_bar = True,
    callback = eval_callback,
    log_interval = 1_000_000
)