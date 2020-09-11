import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss

from simulation.utils.machine_learning.cycle_gan.models.base_model import BaseModel
from simulation.utils.machine_learning.data.image_pool import ImagePool
from simulation.utils.machine_learning.models.helper import set_requires_grad
from .cycle_gan_stats import CycleGANStats


class CycleGANModel(BaseModel):
    """This class implements the CycleGAN model, for learning image-to-image
    translation without paired data.

    By default, it uses a '--netg resnet_9blocks' ResNet generator, a '--netd
    basic' discriminator (PatchGAN introduced by pix2pix), and a least-square
    GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        netg_a: nn.Module,
        netg_b: nn.Module,
        netd_a: nn.Module = None,
        netd_b: nn.Module = None,
        is_train: bool = True,
        cycle_noise_stddev: int = 0,
        pool_size: int = 50,
        beta1: float = 0.5,
        lr: float = 0.0002,
        lr_policy: str = "linear",
        lambda_idt_a: int = 10,
        lambda_idt_b: int = 10,
        lambda_cycle: float = 0.5,
        optimizer_type: str = "adam",
        is_l1: bool = False,
    ):
        """Initialize the CycleGAN class.

        Args:
            is_train (bool): enable or disable training mode
            cycle_noise_stddev (int): Standard deviation of noise added to the
                cycle input. Mean is 0.
            pool_size (int): the size of image buffer that stores previously
                generated images
            beta1 (float): momentum term of adam
            lr (float): initial learning rate for adam
            lr_policy (str): linear #learning rate policy. [linear | step |
                plateau | cosine]
            lambda_idt_a (int): weight for loss of domain A
            lambda_idt_b (int): weight for loss of domain B
            lambda_cycle (float): weight for loss identity
        """
        super().__init__(
            netg_a,
            netg_b,
            netd_a,
            netd_b,
            is_train,
            lambda_cycle,
            lambda_idt_a,
            lambda_idt_b,
            is_l1,
            optimizer_type,
            lr_policy,
            beta1,
            lr,
            cycle_noise_stddev,
        )

        if is_train:
            self.fake_a_pool = ImagePool(
                pool_size
            )  # create image buffer to store previously generated images
            self.fake_b_pool = ImagePool(
                pool_size
            )  # create image buffer to store previously generated images

            # define loss functions
            def gan_loss(prediction: torch.Tensor, is_real: bool):
                target = torch.tensor(
                    1.0 if is_real else 0.0, device=prediction.device
                ).expand_as(prediction)
                return mse_loss(prediction, target)

            self.criterionGAN = gan_loss

    def backward_d_basic(
        self, netd: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> Tensor:
        """Calculate GAN loss for the discriminator

        We also call loss_d.backward() to calculate the gradients.

        Return:
            Discriminator loss.

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

    def backward_d_a(self, real_b, fake_b) -> float:
        """Calculate GAN loss for discriminator D_A"""
        fake_b = self.fake_b_pool.query(fake_b)
        loss_d_a = self.backward_d_basic(self.networks.d_a, real_b, fake_b).item()
        return loss_d_a

    def backward_d_b(self, real_a, fake_a) -> float:
        """Calculate GAN loss for discriminator D_B"""
        fake_a = self.fake_a_pool.query(fake_a)
        loss_d_b = self.backward_d_basic(self.networks.d_b, real_a, fake_a).item()
        return loss_d_b

    def do_iteration(self, batch_a: torch.Tensor, batch_b: torch.Tensor):
        """Calculate losses, gradients, and update network weights; called in
        every training iteration
        """
        real_a = batch_a
        real_b = batch_b
        # forward
        fake_a, fake_b, rec_a, rec_b = self.forward(
            real_a, real_b
        )  # compute fake images and reconstruction images.
        # G_A and G_B
        set_requires_grad(
            [self.networks.d_a, self.networks.d_b], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero

        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        idt_a = self.networks.g_a(real_b)
        loss_idt_a = self.criterionIdt(idt_a, real_b) * self.lambda_idt_a
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        idt_b = self.networks.g_b(real_a)
        loss_idt_b = self.criterionIdt(idt_b, real_a) * self.lambda_idt_b

        # GAN loss D_A(G_A(A))
        loss_g_a = self.criterionGAN(self.networks.d_a(fake_b), True)
        # GAN loss D_B(G_B(B))
        loss_g_b = self.criterionGAN(self.networks.d_b(fake_a), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_a = self.criterionCycle(rec_a, real_a) * self.lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_b = self.criterionCycle(rec_b, real_b) * self.lambda_cycle
        # combined loss and calculate gradients
        loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
        loss_g.backward()
        self.optimizer_g.step()  # update G_A and G_B's weights

        # D_A and D_B
        set_requires_grad([self.networks.d_a, self.networks.d_b], True)
        self.optimizer_d.zero_grad()  # set D_A and D_B's gradients to zero
        loss_d_a = self.backward_d_a(real_b, fake_b)  # calculate gradients for D_A
        loss_d_b = self.backward_d_b(real_a, fake_a)  # calculate gradients for D_B
        self.optimizer_d.step()  # update D_A and D_B's weights

        return CycleGANStats(
            real_a,
            real_b,
            fake_a,
            fake_b,
            rec_a,
            rec_b,
            idt_a,
            idt_b,
            loss_g_a.item(),
            loss_g_b.item(),
            loss_idt_a.item(),
            loss_idt_b.item(),
            loss_cycle_a.item(),
            loss_cycle_b.item(),
            loss_d_a,
            loss_d_b,
        )
