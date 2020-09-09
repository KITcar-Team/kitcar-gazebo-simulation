from typing import List


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    dataset_a: List[str] = [
        "./../../../../data/real_images/maschinen_halle",
        "./../../../../data/real_images/maschinen_halle_no_obstacles",
        "./../../../../data/real_images/beg_2019",
    ]
    """Path to images of domain A (real images). Can be a list of folders."""
    dataset_b: List[str] = ["./../../../../data/simulated_images/random_roads"]
    """Path to images of domain B (simulated images). Can be a list of folders"""
    display_env: str = "main"
    """Visdom display environment name (default is "main")"""
    display_freq: int = 5
    """Frequency of showing training results on screen"""
    display_id: int = 1
    """Window id of the web display"""
    display_port: int = 8097
    """Visdom port of the web display"""
    is_train: bool = True
    """Enable or disable training mode"""
    num_threads: int = 8
    """# threads for loading data"""
    print_freq: int = 10
    """Frequency of showing training results on console"""
    save_by_iter: bool = False
    """Whether saves model by iteration"""
    save_epoch_freq: int = 1
    """Frequency of saving checkpoints at the end of epochs"""
    save_latest_freq: int = 1000
    """Frequency of saving the latest results"""
    beta1: float = 0.5
    """Momentum term of adam"""
    batch_size: int = 1
    """Input batch size"""
    lr: float = 0.000005
    """Initial learning rate for adam"""
    lr_decay_iters: int = 1
    """Multiply by a gamma every lr_decay_iters iterations"""
    lr_policy: str = "step"
    """Learning rate policy. [linear | step | plateau | cosine]"""
    lr_step_factor: float = 0.1
    """Multiplication factor at every step in the step scheduler"""
    n_epochs: int = 10
    """Number of epochs with the initial learning rate"""
    n_epochs_decay: int = 0
    """Number of epochs to linearly decay learning rate to zero"""
    no_flip: bool = False
    """Flip 50% of all training images vertically"""
    use_sigmoid: bool = True
    """Use sigmoid activation at end of discriminator"""
    continue_train: bool = False
    """Load checkpoints or start from scratch"""


class WassersteinCycleGANTrainOptions(TrainOptions):
    wgan_initial_n_critic: int = 1
    """Number of iterations of the critic before starting training loop"""
    wgan_clip_upper: float = 0.001
    """Upper bound for weight clipping"""
    wgan_clip_lower: float = -0.001
    """Lower bound for weight clipping"""
    wgan_n_critic: int = 5
    """Number of iterations of the critic per generator iteration"""
    is_wgan: bool = True
    """Decide whether to use wasserstein cycle gan or standard cycle gan"""


class CycleGANTrainOptions(TrainOptions):
    pass
