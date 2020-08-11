from simulation.utils.machine_learning.cycle_gan.models.helper import (
    get_norm_layer,
    init_net,
)
from simulation.utils.machine_learning.cycle_gan.models.resnet_generator import (
    ResnetGenerator,
)
from simulation.utils.machine_learning.cycle_gan.models.unet_generator import UnetGenerator


def create_generator(
    input_nc,
    output_nc,
    ngf,
    netg,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[0],
    activation="TANH",
    conv_layers_in_block=2,
    dilations=None,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netg (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        activation (string) -- The activation function used at the end
        conv_layers_in_block: Number of convolution layers in each block.
        dilations: List of dilations for each conv layer.

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations. We
        adapt Torch code from Justin Johnson's neural style transfer project (
        https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
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
    return init_net(net, init_type, init_gain, gpu_ids)
