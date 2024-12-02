import optuna
import json
import torch
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.environment.snake_env import SnakeEnv
from src.model.DQN_model import get_dqn_model
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type = int, default = 2_000_000)
parser.add_argument("--trials", type = int, default = 20)
args = parser.parse_args()

def make_env():
    return SnakeEnv()

def optimize_dqn(trial):
    learning_rate = trial.suggest_float('learning_rate', .00001, .01, log=True)
    gamma = trial.suggest_float('gamma', .9, .999, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.5, 0.9)
    
    features_dim = trial.suggest_categorical('features_dim', [32, 128, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    num_envs = 4
    env_list = [make_env for _ in range(num_envs)]
    parallel_snake_env = SubprocVecEnv(env_list)
    eval_env = DummyVecEnv([make_env])

    model = get_dqn_model(
        learning_rate = learning_rate,
        gamma = gamma,
        features_dim = features_dim,
        batch_size = batch_size,
        exploration_fraction = exploration_fraction,
        tensorboard_log = None,
        verbose = 0,
        snake_env = parallel_snake_env
    )

    eval_callback = EvalCallback(
        eval_env = parallel_snake_env,
        n_eval_episodes = 30,
        eval_freq = args.steps,
        best_model_save_path = None,
        deterministic = True,
        verbose = 0
    )

    model.learn(
        total_timesteps = args.steps + 1000,
        callback=eval_callback,
        progress_bar = True
    )

    return eval_callback.last_mean_reward

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    
    log_path = "logs/optuna_logs/DQN/"
    study = optuna.create_study(
        direction="maximize",
        storage = "sqlite:///" + log_path  + "study.db"
    )

    study.optimize(
        optimize_dqn,
        n_trials = args.trials, 
        show_progress_bar = True
    )

    best_params = study.best_params
    best_params_path = log_path + "best_params_dqn.json"

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
