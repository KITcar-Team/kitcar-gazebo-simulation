import abc
import itertools
import os
import pickle
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import L1Loss, MSELoss
from torch.optim import RMSprop

from simulation.utils.basics.init_options import InitOptions
from simulation.utils.machine_learning.models import helper

from .cycle_gan_stats import CycleGANStats


@dataclass
class CycleGANNetworks:
    g_a: nn.Module
    g_b: nn.Module
    d_a: nn.Module = None
    d_b: nn.Module = None

    def save(self, prefix_path: str) -> None:
        """Save all the networks to the disk.

        Args:
            prefix_path (str): the path which gets extended by the model name
        """
        for name, net in self.__dict__.items():
            if net is None:
                continue
            net = pickle.loads(pickle.dumps(net))
            save_path = prefix_path + f"{name}.pth"
            torch.save(net.state_dict(), save_path)

    def load(self, prefix_path: str, device: torch.device):
        """Load all the networks from the disk.

        Args:
            prefix_path (str): the path which is extended by the model name
            device (torch.device): The device on which the networks are loaded
        """
        for name, net in self.__dict__.items():
            if net is None:
                continue
            load_path = prefix_path + f"{name}.pth"
            if not os.path.isfile(load_path):
                raise FileNotFoundError(f"No model weights file found at {load_path}")

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on device
            state_dict = torch.load(load_path, map_location=str(device))
            print(f"Loaded: {load_path}")
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(
                state_dict.keys()
            ):  # need to copy keys here because we mutate in loop
                CycleGANNetworks.__patch_instance_norm_state_dict(
                    state_dict, net, key.split(".")
                )
            net.load_state_dict(state_dict)

    @staticmethod
    def __patch_instance_norm_state_dict(
        state_dict: dict, module: nn.Module, keys: List[str], i: int = 0
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
            CycleGANNetworks.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def print(self, verbose: bool) -> None:
        """Print the total number of parameters in the network and (if verbose) network
        architecture.

        Args:
            verbose (bool): print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name, net in self.__dict__.items():
            if net is None:
                continue
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

    def __iter__(self):
        return (net for net in self.__dict__.values() if net is not None)


class BaseModel(ABC, InitOptions):
    def __init__(
        self,
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
        beta1: float = 0.5,
        lr: float = 0.0002,
        cycle_noise_stddev: float = 0,
    ):
        self.is_train = is_train
        self.lambda_cycle = lambda_cycle
        self.lambda_idt_a = lambda_idt_a
        self.lambda_idt_b = lambda_idt_b
        self.is_l1 = is_l1
        self.metric = 0  # used for learning rate policy 'plateau'
        self.lr_policy = lr_policy

        self.cycle_noise_stddev = cycle_noise_stddev if is_train else 0

        self.networks = CycleGANNetworks(netg_a, netg_b, netd_a, netd_b)

        if self.is_train:
            # define loss functions
            self.criterionCycle = L1Loss() if self.is_l1 else MSELoss()
            self.criterionIdt = L1Loss() if self.is_l1 else MSELoss()

            if optimizer_type == "rms_prop":
                self.optimizer_g = RMSprop(
                    itertools.chain(
                        self.networks.g_a.parameters(), self.networks.g_b.parameters()
                    ),
                    lr=lr,
                )
                self.optimizer_d = RMSprop(
                    itertools.chain(
                        self.networks.d_a.parameters(), self.networks.d_b.parameters()
                    ),
                    lr=lr,
                )
            else:
                self.optimizer_g = torch.optim.Adam(
                    itertools.chain(
                        self.networks.g_a.parameters(), self.networks.g_b.parameters()
                    ),
                    lr=lr,
                    betas=(beta1, 0.999),
                )
                self.optimizer_d = torch.optim.Adam(
                    itertools.chain(
                        self.networks.d_a.parameters(), self.networks.d_b.parameters()
                    ),
                    lr=lr,
                    betas=(beta1, 0.999),
                )

    def forward(self, real_a, real_b) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        fake_b = self.networks.g_a(real_a)  # G_A(A)
        fake_a = self.networks.g_b(real_b)  # G_B(B)

        # Calculate cycle. Add gaussian if self.cycle_noise_stddev is not 0
        # See: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694 # noqa: E501
        # There are two individual noise terms because fake_A and fake_B may
        # have different dimensions
        # (At end of dataset were one of them is not a full batch for example)

        if self.cycle_noise_stddev == 0:
            noise_a = 0
            noise_b = 0
        else:
            noise_a = (
                torch.zeros(fake_a.size())
                .normal_(0, self.cycle_noise_stddev)
                .requires_grad_()
            )
            noise_b = (
                torch.zeros(fake_a.size())
                .normal_(0, self.cycle_noise_stddev)
                .requires_grad_()
            )

        rec_a = self.networks.g_b(fake_b + noise_a)  # G_B(G_A(A))
        rec_b = self.networks.g_a(fake_a + noise_b)  # G_A(G_B(B))

        return fake_a, fake_b, rec_a, rec_b

    def test(self, batch_a, batch_b) -> CycleGANStats:
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate
        steps for backpropagation It also calls <compute_visuals> to produce additional
        visualization results
        """
        with torch.no_grad():
            fake_a, fake_b, rec_a, rec_b = self.forward(batch_a, batch_b)

        return CycleGANStats(batch_a, batch_b, fake_a, fake_b, rec_a, rec_b)

    def create_schedulers(
        self,
        epoch: Union[int, str] = "latest",
        lr_policy: str = "linear",
        lr_decay_iters: int = 50,
        lr_step_factor: float = 0.1,
        n_epochs: int = 100,
    ):
        """Create schedulers.

        Args:
            lr_policy: learning rate policy. [linear | step | plateau | cosine]
            lr_decay_iters: multiply by a gamma every lr_decay_iters iterations
            lr_step_factor: multiply lr with this factor every epoch
            n_epochs: number of epochs with the initial learning rate
        """
        self.schedulers = [
            helper.get_scheduler(
                optimizer, lr_policy, lr_decay_iters, n_epochs, lr_step_factor
            )
            for optimizer in [self.optimizer_d, self.optimizer_g]
        ]

    def eval(self) -> None:
        """Make models eval mode during test time."""
        for net in self.networks:
            net.eval()

    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks."""
        old_lr = self.optimizer_g.param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizer_g.param_groups[0]["lr"]
        print("learning rate %.7f -> %.7f" % (old_lr, lr))

    @abc.abstractmethod
    def do_iteration(
        self, batch_a: Tuple[torch.Tensor, str], batch_b: Tuple[torch.Tensor, str]
    ):
        raise NotImplementedError("Abstract method!")

    def pre_training(self):
        pass
