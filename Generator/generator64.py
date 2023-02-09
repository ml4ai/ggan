"""
Generator that takes nz dimensional noise and generates 64 x 64 x 3 image
Ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # Tanh takes it to [-1, 1]
            # nn.Tanh()
            # our textures are [0, 1] tensors
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inp):
        return self.main(inp)

    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.nz, 1, 1)
