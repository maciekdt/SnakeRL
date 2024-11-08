from environment.snake_env import SnakeEnv
from model.CNN_feature_extractor import CNNFeatureExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN


custom_policy_kwargs = dict(
    features_extractor_class=CNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch = [32]
)

snake_env = SnakeEnv()

eval_callback = EvalCallback(
    eval_env = snake_env,
    eval_freq = 50_000,
    n_eval_episodes = 5,
    deterministic = True,
    render = False,
    verbose = 1
)

dqn_model = DQN(
    policy = "MlpPolicy",
    env = snake_env,
    policy_kwargs = custom_policy_kwargs,
    verbose = 0,
    
    learning_rate = .0001,
    gamma = .95,
    
    buffer_size = 10_000,
    train_freq = 32,
    batch_size = 64,
    
    exploration_initial_eps = 1,
    exploration_final_eps = .05,
    exploration_fraction = .7
)