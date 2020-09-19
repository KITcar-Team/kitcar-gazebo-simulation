from torch import Tensor
from torch import nn as nn

from .unet_block import UnetSkipConnectionBlock


class UnetGenerator(nn.Module):
    """Create a Unet-based generator."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer: nn.Module = nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        """Construct a Unet generator.

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.

        Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            num_downs (int): the number of downsampling layers in UNet.
                For example, # if |num_downs| == 7,
                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int): the number of filters in the last conv layer
            norm_layer (nn.Module): normalization layer
            use_dropout (bool): Use dropout or not
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input: Tensor) -> Tensor:
        """Standard forward.

        Args:
            input (Tensor): the input tensor
        """
        return self.model(input)
