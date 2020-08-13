import torch
from torch import nn as nn


class NoPatchDiscriminator(nn.Module):
    def __init__(self, input_nc: int, norm_layer: nn.Module = nn.BatchNorm2d):
        """Construct a no patch gan discriminator :param input_nc: the number of
        channels in input images :type input_nc: int :param norm_layer:
        normalization layer

        Args:
            input_nc (int): the number of channels in input images
            norm_layer (nn.Module): normalization layer
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwarding through network and avg pooling

        Args:
            x (torch.Tensor): the input tensor
        """
        x = self.model(x)
        # Average pooling and flatten
        return torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
