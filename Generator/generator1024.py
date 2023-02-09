"""
Generator that takes nz dimensional noise and generates 1024 x 1024 x 3 image
Ref: https://github.com/tymokvo/AEGeAN/blob/master/AEGeAN_1024.py
sigmoid layer at the last maps the output image into floats [0, 1] that can me mapped to [0, 255]
"""

import json
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, config_file):
        super(Generator, self).__init__()

        config = json.load(open(config_file, "r"))
        # size of latent code
        nz = config["nz"]
        self.nz = nz
        # size of feature maps in generator
        ngf = config["ngf"]
        # number of channels in training images
        nc = config["nc"]

        self.main = nn.Sequential(
            # block 0
            nn.ConvTranspose2d(nz, ngf * 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.LeakyReLU(inplace=True),

            # block 1
            nn.ConvTranspose2d(ngf * 64, ngf * 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.LeakyReLU(inplace=True),

            # block 2
            nn.ConvTranspose2d(ngf * 64, ngf * 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.LeakyReLU(inplace=True),

            # block 3
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.LeakyReLU(inplace=True),

            # block 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(inplace=True),

            # block 5
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(inplace=True),

            # block 6
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(inplace=True),

            # block 7
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(inplace=True),

            # block 8
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(inplace=True),

            # block 9
            nn.ConvTranspose2d(ngf * 8, nc, 4, stride=2, padding=1, bias=False),
            # Tanh takes it to [-1, 1]
            # nn.Tanh()
            # our textures are [0, 1] tensors
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.nz, 1, 1)
