import torch
from torch import nn as nn


class NoPatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        norm_layer: nn.Module = nn.BatchNorm2d,
        n_layers_d: int = 4,
        use_sigmoid: bool = True,
    ):
        """Construct a no patch gan discriminator :param input_nc: the number of
        channels in input images :type input_nc: int :param norm_layer:
        normalization layer

        Args:
            input_nc (int): the number of channels in input images
            norm_layer (nn.Module): normalization layer
            n_layers_d (int): the number of convolution blocks
            use_sigmoid (bool): sigmoid activation at the end
        """
        super(NoPatchDiscriminator, self).__init__()

        self.use_sigmoid = use_sigmoid

        # A bunch of convolutions one after another
        model = [
            nn.Conv2d(input_nc, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        input_nc = 64
        for i in range(n_layers_d):
            model += [
                nn.Conv2d(input_nc, input_nc * 2, 4, stride=2, padding=1),
                norm_layer(input_nc * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            input_nc *= 2

        model += [nn.Conv2d(512, 1, 8, stride=2, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwarding through network and avg pooling

        Args:
            x (torch.Tensor): the input tensor
        """
        x = self.model(x)
        # Average pooling and flatten

        output = torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        return torch.sigmoid(output) if self.use_sigmoid else output
