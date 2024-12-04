import os
from src.environment.snake_env import SnakeEnv
from src.model.CNN_feature_extractor import CNNFeatureExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

eval_callback = EvalCallback(
    eval_env = Monitor(SnakeEnv()),
    n_eval_episodes = 20,
    deterministic = True,
    render = False,
    verbose = 1,
    eval_freq = 1_000_000 // 8,
    best_model_save_path = os.path.join(
        base_dir,
        "logs/saved_models/best_checkpoints"
    )
)
def get_dqn_model(
    learning_rate = .0001,
    gamma = .95,
    features_dim = 128,
    batch_size = 64,
    train_freq = 32,
    buffer_size = 1_000_000,
    exploration_fraction = 0.8,
    tensorboard_log = os.path.join(base_dir, "logs/tensorboard_logs"),
    verbose = 1,
    snake_env = Monitor(SnakeEnv())
    ):
    
    return  DQN(
        policy = "MlpPolicy",
        env = snake_env,
        verbose = verbose,
        
        learning_rate = learning_rate,
        gamma = gamma,
        
        buffer_size = buffer_size,
        train_freq = train_freq,
        batch_size = batch_size,
        
        exploration_initial_eps = 1,
        exploration_final_eps = .05,
        exploration_fraction = exploration_fraction,
        
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch = [32]
        ),
        
        tensorboard_log = tensorboard_log,
    )