import functools

import torch
from torch import nn as nn


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection. X.

    -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int = None,
        submodule: nn.Module = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        """Construct a Unet submodule with skip connections.

        Args:
            outer_nc (int): the number of filters in the outer conv layer
            inner_nc (int): the number of filters in the inner conv layer
            input_nc (int): the number of channels in input images/features
            submodule (nn.Module): previously defined submodules
            outermost (bool): if this module is the outermost module
            innermost (bool): if this module is the innermost module
            norm_layer (nn.Module): normalization layer
            use_dropout (bool): if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        down_conv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc)
        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_nc)

        if outermost:
            up_conv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_conv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection, if this is not the outermost.

        Args:
            x (torch.Tensor): the input tensor
        """
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)
