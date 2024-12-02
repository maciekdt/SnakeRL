from src.model.DQN_model import eval_callback, get_dqn_model
import torch
import sys
import os
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type = int, default = 40_000_000)
args = parser.parse_args()

params_file = "logs/optuna_logs/DQN/best_params_dqn.json"
try:
    with open(params_file, "r") as f:
        params = json.load(f)
        print("Optuna hyperparameters loaded")
except FileNotFoundError:
    params = {}
    print("JSON hyperparameter file not found. Using default parameters.")
    
get_dqn_model(**params).learn(
    total_timesteps = args.steps,
    progress_bar = True,
    callback = eval_callback,
    log_interval = 1_000_000
)