from typing import List

from torch import nn

from simulation.utils.machine_learning.models.helper import (
    get_norm_layer,
)
from simulation.utils.machine_learning.models.resnet_generator import (
    ResnetGenerator,
)
from simulation.utils.machine_learning.models.unet_generator import UnetGenerator


def create_generator(
    input_nc: int,
    output_nc: int,
    ngf: int,
    netg: str,
    norm: str = "batch",
    use_dropout: bool = False,
    activation: nn.Module = nn.Tanh(),
    conv_layers_in_block: int = 2,
    dilations: List[int] = None,
) -> nn.Module:
    """Create a generator

    Returns a generator

    Our current implementation provides two types of generators: U-Net: [unet_128] (for 128x128 input images) and [
    unet_256] (for 256x256 input images) The original U-Net paper: https://arxiv.org/abs/1505.04597

    Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
    Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations. We
    adapt Torch code from Justin Johnson's neural style transfer project (
    https://github.com/jcjohnson/fast-neural-style).

    It uses RELU for non-linearity.

    Args:
        input_nc (int): # of input image channels: 3 for RGB and 1 for grayscale
        output_nc (int): # of output image channels: 3 for RGB and 1 for grayscale
        ngf (int): # of gen filters in the last conv layer
        netg (str): specify generator architecture [resnet_<ANY_INTEGER>blocks | unet_256 | unet_128]
        norm (str): instance normalization or batch normalization [instance | batch | none]
        use_dropout (bool): enable or disable dropout
        activation (nn.Module): Choose which activation to use.
        conv_layers_in_block (int): specify number of convolution layers per resnet block
        dilations: dilation for individual conv layers in every resnet block
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if "resnet" in netg:
        # Extract number of resnet blocks from name of netg
        blocks = int(netg[7:-6])
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=blocks,
            activation=activation,
            conv_layers_in_block=conv_layers_in_block,
            dilations=dilations,
        )
    elif netg == "unet_128":
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    elif netg == "unet_256":
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netg)
    return net
