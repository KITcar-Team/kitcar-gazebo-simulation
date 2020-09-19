import functools
from typing import List, Union

import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, StepLR
from torch.optim.optimizer import Optimizer

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type: str = "instance") -> nn.Module:
    """Return a normalization layer.

    For BatchNorm, we use learnable affine parameters
    and track running statistics (mean/stddev). For InstanceNorm,
    we do not use learnable affine parameters. We do not track running statistics.

    Args:
        norm_type (str): the name of the normalization layer: batch | instance | none
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
        norm_layer = nn.Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(
    optimizer: Optimizer,
    lr_policy: str,
    lr_decay_iters: int,
    n_epochs: int,
    lr_step_factor: float,
) -> Union[LambdaLR, StepLR, ReduceLROnPlateau]:
    """Return a learning rate scheduler.

    For 'linear', we keep the same learning rate for the first <n_epochs> epochs
    and linearly decay the rate to zero over the next <n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default
    PyTorch schedulers. See https://pytorch.org/docs/stable/optim.html for more details.

    Args:
        optimizer (Optimizer): the optimizer of the network
        lr_policy (str): learning rate policy. [linear | step | plateau | cosine]
        lr_decay_iters (int): multiply by a gamma every lr_decay_iters iterations
        n_epochs (int): number of epochs with the initial learning rate
        lr_step_factor (float): Multiplication factor at every step in the step scheduler
    """
    if lr_policy == "linear":

        def lambda_rule(
            epoch: int, epoch_count: int = 1, n_epochs: int = 100, n_epochs_decay: int = 100
        ) -> float:
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=lr_step_factor
        )
    elif lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        raise NotImplementedError("learning rate policy [%s] is not implemented", lr_policy)
    return scheduler


def init_weights(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
) -> None:
    """Initialize network weights.

    We use 'normal' in the original pix2pix and CycleGAN paper.
    But xavier and kaiming might work better for some applications.
    Feel free to try yourself.

    Args:
        net (nn.Module): network to be initialized
        init_type (str): the name of an initialization method:
            normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m: nn.Module) -> None:  # define the initialization function
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


def init_net(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> nn.Module:
    """Initialize a network.

    1. register CPU/GPU device;
    2. initialize the network weights

    Return an initialized network.

    Args:
        net (nn.Module): the network to be initialized
        init_type (str): the name of an initialization method:
            normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
        device: on which device should the net run
    """
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def set_requires_grad(nets: List[nn.Module], requires_grad: bool = False):
    """Set requires_grad=False for all the networks to avoid unnecessary computations.

    Args:
        nets (List[nn.Module]): set require grads for this list of networks
        requires_grad (bool): enable or disable grads
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if isinstance(net, nn.Module):
            for param in net.parameters():
                param.requires_grad = requires_grad
