import optuna
import json
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environment.snake_env import SnakeEnv
from src.model.DQN_model import get_dqn_model

def optimize_dqn(trial):
    learning_rate = trial.suggest_float('learning_rate', .00001, .01, log=True)
    gamma = trial.suggest_float('gamma', .9, .999, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.5, 0.9)
    
    features_dim = trial.suggest_categorical('features_dim', [32, 128, 512])
    train_freq = trial.suggest_categorical('train_freq', [16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    buffer_size = trial.suggest_categorical('buffer_size', [10_000, 100_000, 1_000_000])

    model = get_dqn_model(
        learning_rate = learning_rate,
        gamma = gamma,
        features_dim = features_dim,
        train_freq = train_freq,
        batch_size = batch_size,
        buffer_size = buffer_size,
        exploration_fraction = exploration_fraction,
        tensorboard_log = None,
        verbose = 0
    )
    
    total_learn_timesteps = 1000
    
    eval_callback = EvalCallback(
        eval_env = Monitor(SnakeEnv()),
        n_eval_episodes = 3,
        eval_freq = total_learn_timesteps,
        best_model_save_path = None,
        deterministic = True,
        verbose = 0
    )

    model.learn(
        total_timesteps = total_learn_timesteps + 1,
        callback=eval_callback,
        progress_bar = False
    )

    return eval_callback.last_mean_reward

log_path = "logs/optuna_logs/DQN/"
study = optuna.create_study(
    direction="maximize",
    storage = "sqlite:///" + log_path  + "study.db"
)

study.optimize(
    optimize_dqn,
    n_trials = 10, 
    show_progress_bar = True
)

best_params = study.best_params
best_params_path = log_path + "best_params_dqn.json"

with open(best_params_path, "w") as f:
    json.dump(best_params, f, indent=4)
