from torch import nn

from simulation.utils.machine_learning.cycle_gan.models.n_layer_discriminator import (
    NLayerDiscriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.no_patch_discriminator import (
    NoPatchDiscriminator,
)
from simulation.utils.machine_learning.models.helper import get_norm_layer


def create_discriminator(
    input_nc: int,
    ndf: int,
    netd: str,
    n_layers_d: int = 3,
    norm: str = "batch",
    use_sigmoid: bool = False,
) -> nn.Module:
    """Create a discriminator.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70Ãƒâ€”70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters than
        a full-image discriminator and can work on arbitrarily-sized images in a
        fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in
        the discriminator with the parameter <n_layers_d> (default=3 as used in
        [basic] (PatchGAN).)

    It uses Leaky RELU for non-linearity.

    Args:
        input_nc (int): # of input image channels: 3 for RGB and 1 for grayscale
        ndf (int): # of discriminator filters in the first conv layer
        netd (str): specify discriminator architecture [basic | n_layers |
            no_patch]. The basic model is a 70x70 PatchGAN. n_layers allows you
            to specify the layers in the discriminator
        n_layers_d (int): number of layers in the discriminator network
        norm (str): instance normalization or batch normalization [instance |
            batch | none]
        use_sigmoid (bool): Use sigmoid activation at the end of discriminator
            network
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netd == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif netd == "n_layers":  # more options
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_d, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif netd == "no_patch":  # without any patch gan
        net = NoPatchDiscriminator(
            input_nc, norm_layer=norm_layer, n_layers_d=n_layers_d, use_sigmoid=use_sigmoid
        )
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netd)
    return net
