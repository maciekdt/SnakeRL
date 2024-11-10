import os
from src.environment.snake_env import SnakeEnv
from src.model.CNN_feature_extractor import CNNFeatureExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

custom_policy_kwargs = dict(
    features_extractor_class=CNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch = [32]
)

snake_env = SnakeEnv()

eval_callback = EvalCallback(
    eval_env = snake_env,
    eval_freq = 100_000,
    n_eval_episodes = 20,
    deterministic = True,
    render = False,
    verbose = 2
)

dqn_model = DQN(
    policy = "MlpPolicy",
    env = snake_env,
    policy_kwargs = custom_policy_kwargs,
    verbose = 2,
    
    learning_rate = .0001,
    gamma = .95,
    
    buffer_size = 100_000,
    train_freq = 32,
    batch_size = 64,
    
    exploration_initial_eps = 1,
    exploration_final_eps = .05,
    exploration_fraction = .8,
    
    tensorboard_log = os.path.join(base_dir, "tensorboard_logs")
)