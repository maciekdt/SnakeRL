from stable_baselines3 import A2C
import os
from src.environment.snake_env import SnakeEnv
from src.model.CNN_feature_extractor import CNNFeatureExtractor
from stable_baselines3.common.monitor import Monitor

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_a2c_model(
    learning_rate = .001,
    gamma = .95,
    ent_coef = 0.01,
    n_steps = 10,
    verbose = 1,
    features_dim = 32,
    snake_env = Monitor(SnakeEnv()),
    tensorboard_log = os.path.join(base_dir, "logs/tensorboard_logs")
    ):
    
    return  A2C(
        policy = "MlpPolicy",
        env = snake_env,
        verbose = verbose,
        
        learning_rate = learning_rate,
        gamma = gamma,
        ent_coef = ent_coef,
        n_steps = n_steps,
        
        
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch = [32]
        ),
        
        tensorboard_log = tensorboard_log,
    )