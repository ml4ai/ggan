import torch
from torch import nn

"""
Encodes the uv_layout for a given mesh into z dimensional latent vector 
such that we can concatenate it with some random noise and provide it as an input to the generator 
"""


class Encoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        """
        Encode image into out_dim vector
        :param in_channels: number of input channels [3 for RGB]
        :param out_dim: size of output latent vector
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=32, out_features=out_dim),
            # map to [-1, 1]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)
