import functools

from torch import nn as nn, Tensor


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_sigmoid: bool = False,
        is_quadratic: bool = True,
    ):
        """Construct a PatchGAN discriminator

        Args:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            n_layers (int): the number of conv layers in the discriminator
            norm_layer (nn.Module): normalization layer
            use_sigmoid (bool): sigmoid activation at the end
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d

        kw = 4
        padding_width = 1
        padding_first_layer = (
            padding_width if is_quadratic else (2 * padding_width, padding_width)
        )
        stride_first_layer = 2 if is_quadratic else (1, 2)

        sequence = [
            nn.Conv2d(
                input_nc,
                ndf,
                kernel_size=kw,
                stride=stride_first_layer,
                padding=padding_first_layer,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        num_filters_multiplier = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            num_filters_multiplier_prev = num_filters_multiplier
            num_filters_multiplier = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * num_filters_multiplier_prev,
                    ndf * num_filters_multiplier,
                    kernel_size=kw,
                    stride=2,
                    padding=padding_width,
                    bias=use_bias,
                ),
                norm_layer(ndf * num_filters_multiplier),
                nn.LeakyReLU(0.2, True),
            ]

        num_filters_multiplier_prev = num_filters_multiplier
        num_filters_multiplier = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * num_filters_multiplier_prev,
                ndf * num_filters_multiplier,
                kernel_size=kw,
                stride=1,
                padding=padding_width,
                bias=use_bias,
            ),
            norm_layer(ndf * num_filters_multiplier),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(
                ndf * num_filters_multiplier,
                1,
                kernel_size=kw,
                stride=1,
                padding=padding_width,
            )
        ]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: Tensor) -> Tensor:
        """Standard forward.

        Args:
            input (Tensor): the input tensor
        """
        return self.model(input)
