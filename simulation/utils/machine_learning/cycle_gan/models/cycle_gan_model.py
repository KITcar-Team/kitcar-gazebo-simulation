import itertools
import os
from collections import OrderedDict

import torch
from torch.autograd import Variable

import simulation.utils.machine_learning.cycle_gan.models.gan_loss
from simulation.utils.machine_learning.cycle_gan.models import helper
from simulation.utils.machine_learning.cycle_gan.util.image_pool import ImagePool


class CycleGANModel:
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        gpu_ids=[0],
        isTrain=True,
        cycle_noise_stddev=0,
        checkpoints_dir="./checkpoints",
        name="kitcar",
        preprocess="resize_and_crop",
        input_nc=1,
        lambda_identity=0.5,
        output_nc=1,
        ngf=32,
        netG="resnet_9blocks",
        norm="batch",
        no_dropout=False,
        init_type="normal",
        init_gain=0.02,
        activation="TANH",
        conv_layers_in_block=2,
        dilations=None,
        netD="basic",
        n_layers_D=3,
        use_sigmoid=False,
        pool_size=50,
        ndf=32,
        gan_mode="lsgan",
        beta1=0.5,
        lr=0.0002,
        lr_policy="linear",
        lambda_A=10,
        lambda_B=10,
    ):
        """Initialize the CycleGAN class.
        """
        self.gpu_ids = gpu_ids
        self.isTrain = isTrain
        self.lr_policy = lr_policy
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
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
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
        ]
        # specify the images you want to save/display. The training/test scripts will call
        # <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if (
            self.isTrain and self.lambda_identity > 0.0
        ):  # if identity loss is used, we also visualize idt_B=G_A(
            # B) ad idt_A=G_A(B)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = (
            visual_names_A + visual_names_B
        )  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        self.cycle_noise_stddev = cycle_noise_stddev if self.isTrain else 0

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = helper.create_generator(
            input_nc,
            output_nc,
            ngf,
            netG,
            norm,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids,
            activation,
            conv_layers_in_block,
            dilations,
        )
        self.netG_B = helper.create_generator(
            output_nc,
            input_nc,
            ngf,
            netG,
            norm,
            not no_dropout,
            init_type,
            init_gain,
            self.gpu_ids,
            activation,
            conv_layers_in_block,
            dilations,
        )

        if self.isTrain:  # define discriminators
            self.netD_A = helper.create_discriminator(
                output_nc,
                netD=netD,
                ndf=ndf,
                n_layers_D=n_layers_D,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=self.gpu_ids,
                use_sigmoid=use_sigmoid,
            )
            self.netD_B = helper.create_discriminator(
                input_nc,
                netD=netD,
                ndf=ndf,
                n_layers_D=n_layers_D,
                norm=norm,
                init_type=init_type,
                init_gain=init_gain,
                gpu_ids=self.gpu_ids,
                use_sigmoid=use_sigmoid,
            )

        if self.isTrain:
            if (
                lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert input_nc == output_nc
            self.fake_A_pool = ImagePool(
                pool_size
            )  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(
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
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=lr,
                betas=(beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=lr,
                betas=(beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # BtoA
        self.real_A = input["B"].to(self.device)
        self.real_B = input["A"].to(self.device)
        self.image_paths = input["B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        # Calculate cycle. Add gaussian if self.cycle_noise_stddev is not 0
        # See: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694
        # There are two individual noise terms because fake_A and fake_B may have different dimensions
        # (At end of dataset were one of them is not a full batch for example)
        noise_B = (
            Variable(
                self.fake_B.data.new(self.fake_B.size()).normal_(0, self.cycle_noise_stddev)
            )
            if self.cycle_noise_stddev != 0
            else 0
        )
        self.rec_A = self.netG_B(self.fake_B + noise_B)  # G_B(G_A(A))

        noise_A = (
            Variable(
                self.fake_A.data.new(self.fake_A.size()).normal_(0, self.cycle_noise_stddev)
            )
            if self.cycle_noise_stddev != 0
            else 0
        )
        self.rec_B = self.netG_A(self.fake_A + noise_A)  # G_A(G_B(B))

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
        if self.isTrain:
            self.schedulers = [
                helper.get_scheduler(optimizer, lr_policy, lr_decay_iters, n_epochs)
                for optimizer in self.optimizers
            ]
        if not self.isTrain or continue_train:
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

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

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
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
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

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    @staticmethod
    def fromOptions(opt):
        return CycleGANModel(
            gpu_ids=opt["gpu_ids"],
            isTrain=opt["isTrain"],
            cycle_noise_stddev=opt["cycle_noise_stddev"],
            checkpoints_dir=opt["checkpoints_dir"],
            name=opt["name"],
            preprocess=opt["preprocess"],
            input_nc=opt["input_nc"],
            lambda_identity=opt["lambda_identity"],
            output_nc=opt["output_nc"],
            ngf=opt["ngf"],
            netG=opt["netG"],
            norm=opt["norm"],
            no_dropout=opt["no_dropout"],
            init_type=opt["init_type"],
            init_gain=opt["init_gain"],
            activation=opt["activation"],
            conv_layers_in_block=opt["conv_layers_in_block"],
            dilations=opt["dilations"],
            netD=opt["netD"],
            n_layers_D=opt["n_layers_D"],
            use_sigmoid=opt["use_sigmoid"],
            pool_size=opt["pool_size"],
            ndf=opt["ndf"],
            gan_mode=opt["gan_mode"],
            beta1=opt["beta1"],
            lr=opt["lr"],
            lr_policy=opt["lr_policy"],
            lambda_A=opt["lambda_A"],
            lambda_B=opt["lambda_B"],
        )
