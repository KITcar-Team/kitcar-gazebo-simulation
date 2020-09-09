from typing import Tuple, List

import torch
from torch import Tensor, nn

from simulation.utils.machine_learning.models.helper import set_requires_grad
from .base_model import BaseModel, CycleGANNetworks
from .cycle_gan_stats import CycleGANStats


class WassersteinCycleGANModel(BaseModel):
    """This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    By default, it uses a '--netg resnet_9blocks' ResNet generator, a '--netd basic' discriminator (PatchGAN
    introduced by pix2pix), and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(
        self,
        netg_a: nn.Module,
        netg_b: nn.Module,
        netd_a: nn.Module = None,
        netd_b: nn.Module = None,
        is_train: bool = True,
        beta1: float = 0.5,
        lr: float = 0.0002,
        lr_policy: str = "linear",
        lambda_idt_a: int = 10,
        lambda_idt_b: int = 10,
        lambda_cycle: float = 0.5,
        optimizer_type: str = "rms_prop",
        is_l1: bool = False,
        wgan_n_critic: int = 5,
        wgan_initial_n_critic: int = 5,
        wgan_clip_lower=-0.01,
        wgan_clip_upper=0.01,
    ):
        """Initialize the CycleGAN class.

        Args:
            is_train (bool): enable or disable training mode
            beta1 (float): momentum term of adam
            lr (float): initial learning rate for adam
            lr_policy (str): linear #learning rate policy. [linear | step | plateau | cosine]
            lambda_idt_a (int): weight for loss of domain A
            lambda_idt_b (int): weight for loss of domain B
            lambda_cycle (float): weight for loss identity
            is_l1 (bool): Decide whether to use l1 loss or l2 loss as cycle and identity loss functions
        """
        self.wgan_initial_n_critic = wgan_initial_n_critic
        self.clips = (wgan_clip_lower, wgan_clip_upper)

        self.wgan_n_critic = wgan_n_critic

        self.networks = CycleGANNetworks(netg_a, netg_b, netd_a, netd_b)

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
        )

    def update_critic_a(
        self, batch_a: Tensor, batch_b: Tensor, clip_bounds: Tuple[float, float] = None,
    ):
        set_requires_grad([self.networks.d_a], requires_grad=True)
        return self.networks.d_a.perform_optimization_step(
            self.networks.g_a, self.optimizer_d, batch_a, batch_b, clip_bounds,
        )

    def update_critic_b(
        self, batch_a: Tensor, batch_b: Tensor, clip_bounds: Tuple[float, float] = None,
    ):
        set_requires_grad([self.networks.d_b], requires_grad=True)
        return self.networks.d_b.perform_optimization_step(
            self.networks.g_b, self.optimizer_d, batch_b, batch_a, clip_bounds,
        )

    def update_generators(self, batch_a: Tensor, batch_b: Tensor):
        """"""
        self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
        # G_A and G_B
        set_requires_grad(
            [self.networks.d_a, self.networks.d_b], False
        )  # Ds require no gradients when optimizing Gs
        set_requires_grad([self.networks.g_a, self.networks.g_b], True)

        g_a_x = self.networks.g_a(batch_b)
        f_g_a_x = self.networks.d_a(g_a_x)
        loss_g_a = -torch.mean(f_g_a_x)

        g_b_x = self.networks.g_b(batch_a)
        f_g_b_x = self.networks.d_b(g_b_x)
        loss_g_b = -torch.mean(f_g_b_x)

        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        idt_a = self.networks.g_a(batch_a)
        loss_idt_a = self.criterionIdt(idt_a, batch_a) * self.lambda_idt_a
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        idt_b = self.networks.g_b(batch_b)
        loss_idt_b = self.criterionIdt(idt_b, batch_b) * self.lambda_idt_b

        rec_a = self.networks.g_a(g_b_x)
        rec_b = self.networks.g_b(g_a_x)

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_a = self.criterionCycle(rec_a, batch_a) * self.lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_b = self.criterionCycle(rec_b, batch_b) * self.lambda_cycle
        # combined loss and calculate gradients
        loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
        loss_g.backward()

        self.optimizer_g.step()  # update G_A and G_B's weights

        stats = CycleGANStats(
            real_a=batch_a,
            real_b=batch_b,
            fake_a=g_a_x,
            fake_b=g_b_x,
            rec_a=rec_a,
            rec_b=rec_b,
            idt_a=idt_a,
            idt_b=idt_b,
            loss_g_a=loss_g_a.item(),
            loss_g_b=loss_g_b.item(),
            loss_idt_a=loss_idt_a.item(),
            loss_idt_b=loss_idt_b.item(),
            loss_cycle_a=loss_cycle_a.item(),
            loss_cycle_b=loss_cycle_b.item(),
        )
        return stats

    def pre_training(self, critic_batches):
        # Update critic
        for batch_a, batch_b in critic_batches:
            self.update_critic_a(batch_a, batch_b, self.clips)
            self.update_critic_b(batch_a, batch_b, self.clips)

    def do_iteration(
        self,
        batch_a: torch.Tensor,
        batch_b: torch.Tensor,
        critic_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        # Update critic
        for batch_a_d, batch_b_d in critic_batches:
            distance_a = self.update_critic_a(batch_a_d, batch_b_d, self.clips)
            distance_b = self.update_critic_b(batch_a_d, batch_b_d, self.clips)

        update_stats: CycleGANStats = self.update_generators(batch_a, batch_b)
        update_stats.w_distance_a = distance_a
        update_stats.w_distance_b = distance_b

        return update_stats
