"""
Discriminator that takes 512 x 512 x 3 image and predicts the probability of real
Ref: https://github.com/tymokvo/AEGeAN/blob/master/AEGeAN_512.py
"""

from torch import nn
import json


class Discriminator(nn.Module):
    def __init__(self, config_file):
        super(Discriminator, self).__init__()

        config = json.load(open(config_file, "r"))
        # size of feature maps in discriminator
        ndf = config["ndf"]
        # number of channels in training images
        nc = config["nc"]

        self.main = nn.Sequential(
            # block 1
            # input: (batch x nc x 512 x 512)
            nn.Conv2d(nc, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # block 2
            # input: (batch x nc x 256 x 256)
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # block 3
            # input: (batch x nc x 128 x 128)
            nn.Conv2d(ndf * 16, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # block 4
            # input: (batch x nc x 64 x 64)
            nn.Conv2d(ndf * 16, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # block 5
            # input: (batch x nc x 32 x 32)
            nn.Conv2d(ndf * 16, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # block 6
            # input: (batch x nc x 16 x 16)
            nn.Conv2d(ndf * 16, ndf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # block 7
            # input: (batch x nc x 8 x 8)
            nn.Conv2d(ndf * 32, ndf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # block 8
            # input: (batch x nc x 4 x 4)
            nn.Conv2d(ndf * 32, ndf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(),

            # block9
            # input: (batch x nc x 2 x 2)
            nn.Conv2d(ndf * 32, 1, 2, stride=2, padding=0, bias=False),
            # Try nn.BCEWithLogitsLoss()
            nn.Sigmoid()
        )

    def forward(self, inp):
        return self.main(inp)
