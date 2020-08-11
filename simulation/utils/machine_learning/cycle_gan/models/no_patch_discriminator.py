import torch
from torch import nn as nn


class NoPatchDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(NoPatchDiscriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            norm_layer(128),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            norm_layer(256),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            norm_layer(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
