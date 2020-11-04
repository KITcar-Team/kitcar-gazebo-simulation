from typing import List, Union

from torch import nn


class BaseOptions:
    activation: nn.Module = nn.Tanh()
    """Choose which activation to use."""
    checkpoints_dir: str = "./checkpoints"
    """models are saved here"""
    conv_layers_in_block: int = 2
    """specify number of convolution layers per resnet block"""
    crop_size: int = 256
    """then crop to this size"""
    dilations: List[int] = [
        1,
        2,
    ]
    """dilation for individual conv layers in every resnet block"""
    epoch: Union[int, str] = "latest"
    """which epoch to load? set to latest to use latest cached model"""
    init_gain: float = 0.02
    """scaling factor for normal, xavier and orthogonal."""
    init_type: str = "normal"
    """network initialization [normal | xavier | kaiming | orthogonal]"""
    input_nc: int = 1
    """# of input image channels: 3 for RGB and 1 for grayscale"""
    lambda_idt_a: float = 5
    """weight for loss identity of domain A"""
    lambda_idt_b: float = 5
    """weight for loss identity of domain B"""
    lambda_cycle: float = 10
    """weight for cycle loss"""
    load_size: int = 256
    """scale images to this size"""
    mask: str = "resources/mask.png"
    """Path to a mask overlaid over all images"""
    n_layers_d: int = 3
    """number of layers in the discriminator network"""
    name: str = "dr_drift_256"
    """name of the experiment. It decides where to store samples and models"""
    ndf: int = 32
    """# of discriminator filters in the first conv layer"""
    netd: str = "basic"
    """Specify discriminator architecture. [basic | n_layers | no_patch].
    The basic model is a 70x70 PatchGAN.
    n_layers allows you to specify the layers in the discriminator.
    """
    netg: str = "resnet_9blocks"
    """specify generator architecture [resnet_<ANY_INTEGER>blocks | unet_256 | unet_128]"""
    ngf: int = 32
    """# of gen filters in the last conv layer"""
    no_dropout: bool = True
    """no dropout for the generator"""
    norm: str = "instance"
    """instance normalization or batch normalization [instance | batch | none]"""
    output_nc: int = 1
    """of output image channels: 3 for RGB and 1 for grayscale"""
    preprocess: set = {"resize", "crop"}
    """Scaling and cropping of images at load time.

    [resize | crop | scale_width]
    """
    verbose: bool = False
    """if specified, print more debugging information"""
    cycle_noise_stddev: float = 0
    """Standard deviation of noise added to the cycle input. Mean is 0. """
    pool_size: int = 75
    """the size of image buffer that stores previously generated images"""
    max_dataset_size: int = -1
    """maximum amount of images to load; -1 means infinity"""
    is_wgan: bool = False
    """Decide whether to use wasserstein cycle gan or standard cycle gan"""
    l1_or_l2_loss: str = "l1"
    """"l1" or "l2"; Decide whether to use l1 or l2 as cycle and identity loss functions"""
    use_sigmoid: bool = True
    """Use sigmoid activation at end of discriminator"""

    @classmethod
    def to_dict(cls) -> dict:
        return {
            k: v
            for cls in reversed(cls.__mro__)
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
        }
