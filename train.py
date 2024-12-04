from src.environment.snake_env import SnakeEnv
from src.model.DQN_model import get_dqn_model
import torch
import sys
import os
import json
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


    
def make_env():
    return SnakeEnv()

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 40_000_000)
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
    
    num_envs = 7
    
    eval_env = Monitor(SnakeEnv())
    eval_callback = EvalCallback(
        eval_env = eval_env,
        n_eval_episodes = 20,
        deterministic = True,
        render = False,
        verbose = 1,
        eval_freq = 1_000_000 / num_envs,
        best_model_save_path = os.path.join(
            "logs/saved_models/best_checkpoints"
        )
    )
    
    tensorboard_log_dir = os.path.join("logs", "tensorboard_logs")
    configure(folder=tensorboard_log_dir)
    
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    
    
    env_list = [make_env for _ in range(num_envs)]
    parallel_snake_env = SubprocVecEnv(env_list)    
        
    get_dqn_model(**params, snake_env=parallel_snake_env, tensorboard_log=tensorboard_log_dir).learn(
        total_timesteps = args.steps,
        progress_bar = True,
        callback = eval_callback,
        log_interval = 1_000_000/num_envs,
    )