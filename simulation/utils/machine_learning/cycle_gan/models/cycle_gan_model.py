import itertools
import os
from collections import OrderedDict

import torch
from torch.autograd import Variable

import simulation.utils.machine_learning.cycle_gan.models.discriminator
import simulation.utils.machine_learning.cycle_gan.models.gan_loss
import simulation.utils.machine_learning.cycle_gan.models.generator
from simulation.utils.machine_learning.cycle_gan.models import helper
from simulation.utils.machine_learning.cycle_gan.util.image_pool import ImagePool


class CycleGANModel:
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    By default, it uses a '--netg resnet_9blocks' ResNet generator,
    a '--netd basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        gpu_ids=[0],
        is_train=True,
        cycle_noise_stddev=0,
        checkpoints_dir="./checkpoints",
        name="kitcar",
        preprocess="resize_and_crop",
        input_nc=1,
        lambda_identity=0.5,
        output_nc=1,
        ngf=32,
        netg="resnet_9blocks",
        norm="batch",
        no_dropout=False,
        init_type="normal",
        init_gain=0.02,
        activation="TANH",
        conv_layers_in_block=2,
        dilations=None,
        netd="basic",
        n_layers_d=3,
        use_sigmoid=False,
        pool_size=50,
        ndf=32,
        gan_mode="lsgan",
        beta1=0.5,
        lr=0.0002,
        lr_policy="linear",
        lambda_a=10,
        lambda_b=10,
    ):
        """Initialize the CycleGAN class."""
        self.gpu_ids = gpu_ids
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

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # BtoA
        self.real_a = input["B"].to(self.device)
        self.real_b = input["A"].to(self.device)
        self.image_paths = input["B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
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
        verbose=False,
        continue_train=False,
        load_iter=0,
        epoch="latest",
        lr_policy="linear",
        lr_decay_iters=50,
        n_epochs=100,
    ):
        """Load and print networks; create schedulers"""
        if self.is_train:
            self.schedulers = [
                helper.get_scheduler(optimizer, lr_policy, lr_decay_iters, n_epochs)
                for optimizer in self.optimizers
            ]
        if not self.is_train or continue_train:
            load_suffix = "iter_%d" % load_iter if load_iter > 0 else epoch
            self.load_networks(load_suffix)
        self.print_networks(verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backpropagation
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate %.7f -> %.7f" % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = dict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
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

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
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

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(
                    state_dict.keys()
                ):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
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

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_d_basic(self, netd, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netd (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_d.backward() to calculate the gradients.
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

    def backward_d_a(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_b = self.fake_b_pool.query(self.fake_b)
        self.loss_d_a = self.backward_d_basic(self.netd_a, self.real_b, fake_b)

    def backward_d_b(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_a = self.fake_a_pool.query(self.fake_a)
        self.loss_d_b = self.backward_d_basic(self.netd_b, self.real_a, fake_a)

    def backward_g(self):
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

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
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
    def from_options(cls, **kwargs):
        init_keys = cls.__init__.__code__.co_varnames  # Access the init functions arguments
        kwargs = {
            key: kwargs[key] for key in init_keys if key in kwargs
        }  # Select all elements in kwargs, that are also arguments of the init function
        return cls(**kwargs)
