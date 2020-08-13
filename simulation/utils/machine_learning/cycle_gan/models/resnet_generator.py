import functools
from typing import Optional, List

from torch import nn as nn, Tensor

from simulation.utils.machine_learning.cycle_gan.models.helper import get_activation_layer
from simulation.utils.machine_learning.cycle_gan.models.resnet_block import ResnetBlock


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few
    downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
        activation: str = "TANH",
        conv_layers_in_block: int = 2,
        dilations: Optional[List[int]] = None,
    ):
        """Construct a Resnet-based generator

        Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_layer (nn.Module): normalization layer
            use_dropout (bool): if use dropout layers
            n_blocks (int): the number of ResNet blocks
            padding_type (str): the name of padding layer in conv layers:
                reflect | replicate | zero
            activation (str):
            conv_layers_in_block (int): Number of convolution layers in each
                block.
            dilations: List of dilations for each conv layer.
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            multiplier = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * multiplier,
                    ngf * multiplier * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * multiplier * 2),
                nn.ReLU(True),
            ]

        multiplier = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * multiplier,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    n_conv_layers=conv_layers_in_block,
                    dilations=dilations,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            multiplier = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * multiplier,
                    int(ngf * multiplier / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * multiplier / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [get_activation_layer(activation)]

        self.model = nn.Sequential(*model)

    def forward(self, input: Tensor) -> Tensor:
        """Standard forward.

        Args:
            input (Tensor): the input tensor
        """
        return self.model(input)
