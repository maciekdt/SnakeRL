from src.model.A2C_model import get_a2c_model
from src.model.eval_callback import get_eval_callback
from src.environment.snake_env import SnakeEnv
import torch
import sys
import os
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    print(f"Using device: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Using device: CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 40_000_000)
    parser.add_argument("--logging", type = int, default = 1_000_000)
    parser.add_argument("--vcpu", type = int, default = 8)
    args = parser.parse_args()

    vec_env = make_vec_env(SnakeEnv, n_envs=args.vcpu, vec_env_cls=SubprocVecEnv)

    get_a2c_model(snake_env=vec_env).learn(
            total_timesteps = args.steps,
            progress_bar = True,
            callback = get_eval_callback(args.logging / args.vcpu),
            log_interval = args.logging / args.vcpu,
    )