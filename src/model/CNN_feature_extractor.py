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
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),       
        )
        
        with th.no_grad():
            dummy_input = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observation: dict) -> th.Tensor:
        x = self.cnn(observation)
        return self.linear(x)
        