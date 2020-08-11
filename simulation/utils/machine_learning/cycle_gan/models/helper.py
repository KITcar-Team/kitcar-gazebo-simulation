import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################
from simulation.utils.machine_learning.cycle_gan.models.n_layer_discriminator import (
    NLayerDiscriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.no_patch_discriminator import (
    NoPatchDiscriminator,
)
from simulation.utils.machine_learning.cycle_gan.models.resnet_generator import (
    ResnetGenerator,
)
from simulation.utils.machine_learning.cycle_gan.models.unet_generator import UnetGenerator


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer():
            return nn.Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, lr_policy, lr_decay_iters, n_epochs):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network

    For 'linear', we keep the same learning rate for the first <n_epochs> epochs
    and linearly decay the rate to zero over the next <n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == "linear":

        def lambda_rule(epoch, epoch_count=1, n_epochs=100, n_epochs_decay=100):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[0]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def create_generator(
    input_nc,
    output_nc,
    ngf,
    netG,
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
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
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

    if "resnet" in netG:
        # Extract number of resnet blocks from name of netG
        blocks = int(netG[7:-6])
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
    elif netG == "unet_128":
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    elif netG == "unet_256":
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def create_discriminator(
    input_nc,
    ndf,
    netD,
    n_layers_D=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    use_sigmoid=False,
):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | no_patch
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
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
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid
        )
    elif netD == "no_patch":  # without any patch gan
        net = NoPatchDiscriminator(input_nc, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return init_net(net, init_type, init_gain, gpu_ids)
