import optuna
import json
import torch
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from src.environment.snake_env import SnakeEnv
from src.model.DQN_model import get_dqn_model
from stable_baselines3.common.vec_env import SubprocVecEnv
import gc
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type = int, default = 2_000_000)
parser.add_argument("--trials", type = int, default = 20)
parser.add_argument("--vcpu", type = int, default = 8)
args = parser.parse_args()


def make_env():
    return lambda: SnakeEnv()

def optimize_dqn(trial, parallel_snake_env):
    learning_rate = trial.suggest_float('learning_rate', .00001, .01, log=True)
    gamma = trial.suggest_float('gamma', .9, .999, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.5, 0.9)
    
    features_dim = trial.suggest_categorical('features_dim', [32, 512, 2048])
    batch_size = trial.suggest_categorical('batch_size', [64, 256, 1024])
    buffer_size = trial.suggest_categorical('buffer_size', [100_000, 1_000_000])
    
    model = get_dqn_model(
        learning_rate = learning_rate,
        gamma = gamma,
        features_dim = features_dim,
        batch_size = batch_size,
        exploration_fraction = exploration_fraction,
        buffer_size = buffer_size,
        tensorboard_log = None,
        verbose = 0,
        snake_env = parallel_snake_env
    )

    eval_callback = EvalCallback(
        eval_env = parallel_snake_env,
        n_eval_episodes = 30,
        eval_freq = args.steps // args.vcpu,
        best_model_save_path = None,
        deterministic = True,
        verbose = 0
    )

    model.learn(
        total_timesteps = args.steps + 10_000,
        callback=eval_callback,
        progress_bar = True
    )
    
    mean_reward = eval_callback.last_mean_reward
    
    del model.replay_buffer
    del model
    del eval_callback
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Replay buffer cleaned")
    
    return mean_reward

if __name__ == "__main__":
    print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")
    
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    
    env_list = [make_env() for _ in range(args.vcpu)]
    parallel_snake_env = SubprocVecEnv(env_list)
    print("Created vec-env on", args.vcpu, "vCPUs")
    
    log_path = "logs/optuna_logs/DQN/"
    study = optuna.create_study(
        direction="maximize",
        storage = "sqlite:///" + log_path  + "study.db",
        load_if_exists = True,
        study_name = "no-name-03a6446b-9722-4191-8720-a85a4c775ed0"
    )

    study.optimize(
        lambda trial: optimize_dqn(trial, parallel_snake_env),
        n_trials = args.trials, 
        show_progress_bar = True
    )

    best_params = study.best_params
    best_params_path = log_path + "best_params_dqn.json"

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
        
    parallel_snake_env.close()
    del parallel_snake_env
