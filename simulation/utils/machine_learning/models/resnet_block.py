from typing import List

from torch import Tensor
from torch import nn as nn


class ResnetBlock(nn.Module):
    """Define a Resnet block."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: nn.Module,
        use_dropout: bool,
        use_bias: bool,
        n_conv_layers: int = 2,
        dilations: List[int] = None,
    ):
        """Initialize the Resnet block.

        A resnet block is a conv block with skip connections.
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function. Original Resnet paper:
        https://arxiv.org/pdf/1512.03385.pdf

        Args:
            dim (int): number of channels in the conv layer.
            padding_type (str): name of padding layer: reflect | replicate | zero
            norm_layer (nn.Module): normalization layer
            use_dropout (bool): if use dropout layers.
            use_bias (bool): if the conv layer uses bias or not
            n_conv_layers (int): Number of convolution layers in this block.
            dilations: List of dilations for each conv layer.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, n_conv_layers, dilations
        )

    def build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer,
        use_dropout: bool,
        use_bias: bool,
        n_conv_layers: int = 2,
        dilations: List[int] = None,
    ) -> nn.Sequential:
        """Construct a convolutional block.

        Args:
            dim (int): number of channels in the conv layer.
            padding_type (str): name of padding layer: reflect | replicate | zero
            norm_layer: normalization layer
            use_dropout (bool): if use dropout layers.
            use_bias (bool): if the conv layer uses bias or not
            n_conv_layers (int): Number of convolution layers in this block.
            dilations: List of dilations for each conv layer.

        Returns:
            A conv block (with a conv layer, a normalization layer,
            and a non-linearity layer (ReLU))
        """
        if dilations == "None":
            dilations = [1 for _ in range(n_conv_layers)]

        assert n_conv_layers == len(
            dilations
        ), "There must be exactly one dilation value for each conv layer."

        conv_block = []

        for dilation in dilations:
            padding = 0
            if padding_type == "reflect":
                conv_block += [nn.ReflectionPad2d(dilation)]
            elif padding_type == "replicate":
                conv_block += [nn.ReplicationPad2d(dilation)]
            elif padding_type == "zero":
                padding = dilation
            else:
                raise NotImplementedError("padding [%s] is not implemented" % padding_type)

            conv_block += [
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=padding,
                    dilation=dilation,
                    bias=use_bias,
                ),
                norm_layer(dim),
                nn.ReLU(True),
            ]

            if use_dropout:
                conv_block += [nn.Dropout(0.5)]

        if use_dropout:
            # The last dropout layer should not be there
            del conv_block[-1]

        return nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward with skip connection.

        Args:
            x (Tensor): the input tensor
        """
        out = x + self.conv_block(x)  # add skip connections
        return out
