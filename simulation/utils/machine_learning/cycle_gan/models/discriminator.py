from simulation.utils.machine_learning.cycle_gan.models.helper import (
    get_norm_layer,
    init_net,
)
from simulation.utils.machine_learning.cycle_gan.models.n_layer_discriminator import (
    NLayerDiscriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.no_patch_discriminator import (
    NoPatchDiscriminator,
)


def create_discriminator(
    input_nc,
    ndf,
    netd,
    n_layers_d=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[0],
    use_sigmoid=False,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netd (str)         -- the architecture's name: basic | n_layers | no_patch
        n_layers_d (int)   -- the number of conv layers in the discriminator; effective when netd=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_d> (default=3 as used in [basic] (PatchGAN).)

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
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
        net = NoPatchDiscriminator(input_nc, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netd)
    return init_net(net, init_type, init_gain, gpu_ids)
