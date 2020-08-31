import itertools
import os
from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.autograd import Variable

import simulation.utils.machine_learning.cycle_gan.models.discriminator
import simulation.utils.machine_learning.cycle_gan.models.gan_loss
import simulation.utils.machine_learning.cycle_gan.models.generator
from simulation.utils.machine_learning.cycle_gan.models import helper
from simulation.utils.machine_learning.data.image_pool import ImagePool


class CycleGANModel:
    """This class implements the CycleGAN model, for learning image-to-image
    translation without paired data.

    By default, it uses a '--netg resnet_9blocks' ResNet generator, a '--netd
    basic' discriminator (PatchGAN introduced by pix2pix), and a least-square
    GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        gpu_ids: List[int] = [0],
        is_train: bool = True,
        preprocess: str = "resize_and_crop",
        gan_mode: str = "lsgan",
        netg: str = "resnet_9blocks",
        netd: str = "basic",
        ndf: int = 32,
        ngf: int = 32,
        n_layers_d: int = 3,
        norm: str = "batch",
        activation: str = "TANH",
        use_sigmoid: bool = False,
        conv_layers_in_block: int = 2,
        cycle_noise_stddev: int = 0,
        dilations: List[int] = None,
        no_dropout: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
        pool_size: int = 50,
        beta1: float = 0.5,
        lr: float = 0.0002,
        lr_policy: str = "linear",
        lambda_a: int = 10,
        lambda_b: int = 10,
        lambda_identity: float = 0.5,
        checkpoints_dir: str = "./checkpoints",
        name: str = "kitcar",
    ):
        """Initialize the CycleGAN class.

        Args:
            input_nc (int): # of input image channels: 3 for RGB and 1 for
                grayscale
            output_nc (int): # of output image channels: 3 for RGB and 1 for
                grayscale
            gpu_ids: e.g. 0 0,1,2, 0,2. use -1 for CPU
            is_train (bool): enable or disable training mode
            preprocess (str): scaling and cropping of images at load time
                [resize_and_crop | crop | scale_width | scale_width_and_crop |
                none]
            gan_mode (str): the type of GAN objective. [vanilla| lsgan |
                wgangp]. vanilla GAN loss is the cross-entropy objective used in
                the original GAN paper.
            netg (str): specify generator architecture
                [resnet_<ANY_INTEGER>blocks | unet_256 | unet_128]
            netd (str): specify discriminator architecture [basic | n_layers |
                no_patch]. The basic model is a 70x70 PatchGAN. n_layers allows
                you to specify the layers in the discriminator
            ndf (int): # of discriminator filters in the first conv layer
            ngf (int): # of gen filters in the last conv layer
            n_layers_d (int): number of layers in the discriminator network
            norm (str): instance normalization or batch normalization [instance
                | batch | none]
            activation (str): Choose which activation to use. [TANH | HARDTANH |
                SELU | CELU | SOFTSHRINK | SOFTSIGN]
            use_sigmoid (bool): Use sigmoid activation at the end of
                discriminator network
            conv_layers_in_block (int): specify number of convolution layers per
                resnet block
            cycle_noise_stddev (int): Standard deviation of noise added to the
                cycle input. Mean is 0.
            dilations: dilation for individual conv layers in every resnet block
            no_dropout (bool): no dropout for the generator
            init_type (str): network initialization [normal | xavier | kaiming |
                orthogonal]
            init_gain (float): scaling factor for normal, xavier and orthogonal.
            pool_size (int): the size of image buffer that stores previously
                generated images
            beta1 (float): momentum term of adam
            lr (float): initial learning rate for adam
            lr_policy (str): linear #learning rate policy. [linear | step |
                plateau | cosine]
            lambda_a (int): weight for loss of domain A
            lambda_b (int): weight for loss of domain B
            lambda_identity (float): weight for loss identity
            checkpoints_dir (str): models are saved here
            name (str): name of the experiment. It decides where to store
                samples and models
        """
        self.gpu_ids = gpu_ids if torch.cuda.device_count() >= 1 else None
        self.is_train = is_train
        self.lr_policy = lr_policy
        self.lambda_identity = lambda_identity
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )  # get device name: CPU or GPU
        self.save_dir = os.path.join(
            checkpoints_dir, name
        )  # save all the checkpoints to save_dir
        if (
            preprocess != "scale_width"
        ):  # with [scale_width], input images might have different sizes,
            # which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        # specify the training losses you want to print out. The training/test scripts will call
        # <BaseModel.get_current_losses>
        self.loss_names = [
            "d_a",
            "g_a",
            "cycle_a",
            "idt_a",
            "d_b",
            "g_b",
            "cycle_b",
            "idt_b",
        ]
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        visual_names_a = ["real_a", "fake_b", "rec_a"]
        visual_names_b = ["real_b", "fake_a", "rec_b"]
        if (
            self.is_train and self.lambda_identity > 0.0
        ):  # if identity loss is used, we also visualize idt_B=G_A(
            # B) ad idt_A=G_A(B)
            visual_names_a.append("idt_b")
            visual_names_b.append("idt_a")

        self.visual_names = (
            visual_names_a + visual_names_b
        )  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.is_train:
            self.model_names = ["g_a", "g_b", "d_a", "d_b"]
        else:  # during test time, only load Gs
            self.model_names = ["g_a", "g_b"]

        self.cycle_noise_stddev = cycle_noise_stddev if self.is_train else 0

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netg_a = simulation.utils.machine_learning.cycle_gan.models.generator.create_generator(
            input_nc,
            output_nc,
            ngf,
            netg,
            norm,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids,
            activation,
            conv_layers_in_block,
            dilations,
        )
        self.netg_b = simulation.utils.machine_learning.cycle_gan.models.generator.create_generator(
            output_nc,
            input_nc,
            ngf,
            netg,
            norm,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids,
            activation,
            conv_layers_in_block,
            dilations,
        )

        if self.is_train:  # define discriminators
            self.netd_a = simulation.utils.machine_learning.cycle_gan.models.discriminator.create_discriminator(
                output_nc,
                netd=netd,
                ndf=ndf,
                n_layers_d=n_layers_d,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=self.gpu_ids,
                use_sigmoid=use_sigmoid,
            )
            self.netd_b = simulation.utils.machine_learning.cycle_gan.models.discriminator.create_discriminator(
                input_nc,
                netd=netd,
                ndf=ndf,
                n_layers_d=n_layers_d,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=self.gpu_ids,
                use_sigmoid=use_sigmoid,
            )

        if self.is_train:
            if (
                lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert input_nc == output_nc
            self.fake_a_pool = ImagePool(
                pool_size
            )  # create image buffer to store previously generated images
            self.fake_b_pool = ImagePool(
                pool_size
            )  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = simulation.utils.machine_learning.cycle_gan.models.gan_loss.GANLoss(
                gan_mode
            ).to(
                self.device
            )  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_g = torch.optim.Adam(
                itertools.chain(self.netg_a.parameters(), self.netg_b.parameters()),
                lr=lr,
                betas=(beta1, 0.999),
            )
            self.optimizer_d = torch.optim.Adam(
                itertools.chain(self.netd_a.parameters(), self.netd_b.parameters()),
                lr=lr,
                betas=(beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_g)
            self.optimizers.append(self.optimizer_d)

    def set_input(self, input: dict) -> None:
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.
        """
        # BtoA
        self.real_a = input["B"].to(self.device)
        self.real_b = input["A"].to(self.device)
        self.image_paths = input["B_paths"]

    def forward(self) -> None:
        """Run forward pass; called by both functions <optimize_parameters> and
        <test>.
        """
        self.fake_b = self.netg_a(self.real_a)  # G_A(A)
        self.fake_a = self.netg_b(self.real_b)  # G_B(B)

        # Calculate cycle. Add gaussian if self.cycle_noise_stddev is not 0
        # See: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694
        # There are two individual noise terms because fake_A and fake_B may have different dimensions
        # (At end of dataset were one of them is not a full batch for example)

        noise_a = (
            Variable(
                self.fake_a.data.new(self.fake_a.size()).normal_(0, self.cycle_noise_stddev)
            )
            if self.cycle_noise_stddev != 0
            else 0
        )
        self.rec_a = self.netg_b(self.fake_b + noise_a)  # G_B(G_A(A))

        noise_b = (
            Variable(
                self.fake_b.data.new(self.fake_b.size()).normal_(0, self.cycle_noise_stddev)
            )
            if self.cycle_noise_stddev != 0
            else 0
        )
        self.rec_b = self.netg_a(self.fake_a + noise_b)  # G_A(G_B(B))

    def setup(
        self,
        verbose: bool = False,
        load_iter: int = 0,
        epoch: str = "latest",
        lr_policy: str = "linear",
        lr_decay_iters: int = 50,
        lr_step_factor: float = 0.1,
        n_epochs: int = 100,
    ) -> None:
        """Load and print networks; create schedulers

        Args:
            verbose (bool): if specified, print more debugging information
            load_iter (int): which iteration to load? if load_iter > 0, the code
                will load models by iter_[load_iter]; otherwise, the code will
                load models by [epoch]
            epoch (str): which epoch to load? set to latest to use latest cached
                model
            lr_policy (str): learning rate policy. [linear | step | plateau |
                cosine]
            lr_decay_iters (int): multiply by a gamma every lr_decay_iters
                iterations
            n_epochs (int): number of epochs with the initial learning rate
        """
        if self.is_train:
            self.schedulers = [
                helper.get_scheduler(
                    optimizer, lr_policy, lr_decay_iters, n_epochs, lr_step_factor
                )
                for optimizer in self.optimizers
            ]
        load_suffix = "iter_%d" % load_iter if load_iter > 0 else epoch
        self.load_networks(load_suffix)
        self.print_networks(verbose)

    def eval(self) -> None:
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self) -> None:
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save
        intermediate steps for backpropagation It also calls <compute_visuals>
        to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks; called at the end of
        every epoch
        """
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate %.7f -> %.7f" % (old_lr, lr))

    def get_current_visuals(self) -> dict:
        """Return visualization images. train.py will display these images with
        visdom.
        """
        visual_ret = dict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self) -> OrderedDict:
        """Return training losses / errors. train.py will print out these errors
        on console, and save them to a file
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch: int) -> None:
        """Save all the networks to the disk.

        Args:
            epoch (int): current epoch; used in the file name '%s_net_%s.pth' %
                (epoch, name):
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(
        self, state_dict: dict, module: nn.Module, keys: List[str], i: int = 0
    ) -> None:
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)

        Args:
            state_dict (dict): a dict containing parameters from the saved model
                files
            module (nn.Module): the network loaded from a file
            keys (List[int]): the keys inside the save file
            i (int): current index in network structure
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch: int) -> None:
        """Load all the networks from the disk.

        Args:
            epoch (int): current epoch; used in the file name '%s_net_%s.pth' %
                (epoch, name):
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if not os.path.isfile(load_path):
                    if not self.is_train:
                        raise RuntimeError("You have to provide checkpoints at test time!")
                    else:
                        continue

                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                print(f"Loaded: {load_path}")
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(
                    state_dict.keys()
                ):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose: bool) -> None:
        """Print the total number of parameters in the network and (if verbose)
        network architecture

        Args:
            verbose (bool): print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets: List[nn.Module], requires_grad: bool = False) -> None:
        """Set requires_grad=False for all the networks to avoid unnecessary
        computations :param nets: :type nets: network list :param requires_grad:
        :type requires_grad: bool

        Args:
            nets: set require grads for this list of networks
            requires_grad (bool): enable or disable grads
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_d_basic(
        self, netd: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> float:
        """Calculate GAN loss for the discriminator

        Return the discriminator loss. We also call loss_d.backward() to
        calculate the gradients.

        Args:
            netd (nn.Module): the discriminator network
            real (torch.Tensor): the real image
            fake (torch.Tensor): the fake image
        """
        # Real
        pred_real = netd(real)
        loss_d_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netd(fake.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        return loss_d

    def backward_d_a(self) -> None:
        """Calculate GAN loss for discriminator D_A"""
        fake_b = self.fake_b_pool.query(self.fake_b)
        self.loss_d_a = self.backward_d_basic(self.netd_a, self.real_b, fake_b)

    def backward_d_b(self) -> None:
        """Calculate GAN loss for discriminator D_B"""
        fake_a = self.fake_a_pool.query(self.fake_a)
        self.loss_d_b = self.backward_d_basic(self.netd_b, self.real_a, fake_a)

    def backward_g(self) -> None:
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_a = self.lambda_a
        lambda_b = self.lambda_b
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_a = self.netg_a(self.real_b)
            self.loss_idt_a = (
                self.criterionIdt(self.idt_a, self.real_b) * lambda_b * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_b = self.netg_b(self.real_a)
            self.loss_idt_b = (
                self.criterionIdt(self.idt_b, self.real_a) * lambda_a * lambda_idt
            )
        else:
            self.loss_idt_a = 0
            self.loss_idt_b = 0

        # GAN loss D_A(G_A(A))
        self.loss_g_a = self.criterionGAN(self.netd_a(self.fake_b), True)
        # GAN loss D_B(G_B(B))
        self.loss_g_b = self.criterionGAN(self.netd_b(self.fake_a), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_a = self.criterionCycle(self.rec_a, self.real_a) * lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_b = self.criterionCycle(self.rec_b, self.real_b) * lambda_b
        # combined loss and calculate gradients
        self.loss_g = (
            self.loss_g_a
            + self.loss_g_b
            + self.loss_cycle_a
            + self.loss_cycle_b
            + self.loss_idt_a
            + self.loss_idt_b
        )
        self.loss_g.backward()

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights; called in
        every training iteration
        """
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netd_a, self.netd_b], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_g()  # calculate gradients for G_A and G_B
        self.optimizer_g.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netd_a, self.netd_b], True)
        self.optimizer_d.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_d_a()  # calculate gradients for D_A
        self.backward_d_b()  # calculate gradients for D_B
        self.optimizer_d.step()  # update D_A and D_B's weights

    @classmethod
    def from_options(cls, **kwargs: dict):
        """This method applies all parameters from a dict to constructor of
        CycleGanModel and returns the object

        Args:
            **kwargs (dict): the dict with keys matching the constructor
                variables
        """
        init_keys = cls.__init__.__code__.co_varnames  # Access the init functions arguments
        kwargs = {
            key: kwargs[key] for key in init_keys if key in kwargs
        }  # Select all elements in kwargs, that are also arguments of the init function
        return cls(**kwargs)
