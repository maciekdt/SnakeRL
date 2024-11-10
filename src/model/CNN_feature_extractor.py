import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),       
            nn.LazyLinear(out_features=features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observation: dict) -> th.Tensor:
        return self.cnn(observation)