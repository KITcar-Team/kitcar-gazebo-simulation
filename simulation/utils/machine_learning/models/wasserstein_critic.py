import functools
from typing import List, Optional, Tuple

import torch
from torch import nn as nn
from torch.nn import Flatten

from simulation.utils.basics.init_options import InitOptions

from .helper import get_norm_layer
from .resnet_block import ResnetBlock


class WassersteinCritic(nn.Module, InitOptions):
    def __init__(
        self,
        input_nc: int,
        n_blocks: int = 3,
        norm: str = "instance",
        ndf=32,
        height=256,
        width=256,
        use_dropout: bool = False,
        padding_type: str = "reflect",
        conv_layers_in_block: int = 2,
        dilations: Optional[List[int]] = None,
    ):
        """WGAN Critic.

        Implementation follows https://github.com/martinarjovsky/WassersteinGAN

        Args:
            input_nc: Number of channels in input images
            norm: Normalization layer
            n_blocks: Number of resnet blocks
            ndf: Number of features in conv layers
            height: Height of the input image
            width: Width of the input image
            use_dropout: Indicate usage of dropout in resnet blocks
            padding_type: Type of padding to be used
            conv_layers_in_block: Number of convolution layers in each resnet block
            dilations: Type of dilations within each resnet block
        """
        super().__init__()

        norm_layer = get_norm_layer(norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ndf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ndf),
            nn.ReLU(True),
        ]
        dilations = (
            [1 for _ in range(conv_layers_in_block)] if dilations is None else dilations
        )

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            multiplier = 2 ** i
            model += [
                nn.Conv2d(
                    ndf * multiplier,
                    ndf * multiplier * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ndf * multiplier * 2),
                nn.ReLU(True),
            ]

        multiplier = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ndf * multiplier,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    n_conv_layers=conv_layers_in_block,
                    dilations=dilations,
                )
            ]

        model.append(Flatten())
        model += [
            nn.Linear(
                int(ndf * multiplier * height * width / pow(2, 2 * n_downsampling)), 1
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def _clip_weights(self, bounds: Tuple[float, float] = (-0.01, 0.01)):
        """Clip weights to given bounds."""
        # Clip weights of discriminator
        for p in self.parameters():
            p.data.clamp_(*bounds)

    def perform_optimization_step(
        self,
        generator: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_critic: torch.Tensor,
        batch_generator: torch.Tensor,
        weight_clips: Tuple[float, float] = None,
    ) -> float:
        """Do one iteration to update the parameters.

        Args:
            generator: Generation network
            optimizer: Optimizer for the critic's weights
            batch_critic: A batch of inputs for the critic
            batch_generator: A batch of inputs for the generator
            weight_clips: Optional weight bounds for the critic's weights

        Return:
            Current wasserstein distance estimated by critic.
        """

        """Attempt to use WGAN divergence instead of weight clipping.

        from torch.autograd import Variable
        import torch.autograd as autograd

        p = 1
        batch = Variable(batch_critic.type(torch.Tensor), requires_grad=True)
        grad_out = Variable(
            torch.Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False
        ).to(self.device)
        grad = autograd.grad(
            self(batch),
            batch,
            grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = grad.view(grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        # Loss: (gradient ascent)
        loss_d = (-torch.mean(f_x) + torch.mean(f_g_x)) + 1 / 2 * torch.mean(grad_norm).to(
            self.device
        )
        """

        optimizer.zero_grad()
        # Batch 1 into critic
        f_x = self(batch_critic)

        # Batch 2 in generator
        g_x = generator(batch_generator).detach()

        # Batch from generator in critic
        f_g_x = self(g_x)

        # Loss: (gradient ascent)
        loss_d = -1 * (torch.mean(f_x) - torch.mean(f_g_x))

        loss_d.backward()
        optimizer.step()

        if weight_clips is not None:
            self._clip_weights(weight_clips)

        return -1 * loss_d.detach().item()
