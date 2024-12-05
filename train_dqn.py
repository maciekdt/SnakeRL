from src.model.eval_callback import get_eval_callback
from src.environment.snake_env import SnakeEnv
from src.model.DQN_model import get_dqn_model
import torch
import sys
import os
import json
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv


    
def make_env():
    return SnakeEnv()

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 40_000_000)
    parser.add_argument("--logging", type = int, default = 1_000_000)
    parser.add_argument("--vcpu", type = int, default = 8)
    args = parser.parse_args()

    params = None
    params_file = "logs/optuna_logs/DQN/best_params_dqn.json"
    try:
        with open(params_file, "r") as f:
            params = json.load(f)
            print("Optuna hyperparameters loaded")
    except FileNotFoundError:
        params = {}
        print("JSON hyperparameter file not found. Using default parameters.")
    

    

    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    
    
    env_list = [make_env for _ in range(args.vcpu)]
    parallel_snake_env = SubprocVecEnv(env_list)
    print("Created vec-env on", args.vcpu, "vCPUs")
        
    get_dqn_model(
        **params,
        snake_env=parallel_snake_env,
        ).learn(
            total_timesteps = args.steps,
            progress_bar = True,
            callback = get_eval_callback(args.logging / args.vcpu),
            log_interval = args.logging / args.vcpu,
    )