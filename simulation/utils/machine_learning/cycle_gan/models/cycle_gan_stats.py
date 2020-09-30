from dataclasses import dataclass

from torch import Tensor


@dataclass
class CycleGANStats:
    real_a: Tensor = None
    real_b: Tensor = None
    fake_a: Tensor = None
    fake_b: Tensor = None
    rec_a: Tensor = None
    rec_b: Tensor = None
    idt_a: Tensor = None
    idt_b: Tensor = None
    loss_g_a_to_b: float = None
    loss_g_b_to_a: float = None
    loss_idt_a: float = None
    loss_idt_b: float = None
    loss_cycle_a: float = None
    loss_cycle_b: float = None
    loss_d_a: float = None
    loss_d_b: float = None
    w_distance_a: float = None
    w_distance_b: float = None

    def get_visuals(self):
        visuals = {
            "real_a": self.real_a,
            "fake_b": self.fake_b,
            "cycle_a": self.rec_a,
            "idt_a": self.idt_a,
            "real_b": self.real_b,
            "fake_a": self.fake_a,
            "cycle_b": self.rec_b,
            "idt_b": self.idt_b,
        }
        # Filter nones
        visuals = {k: v for k, v in visuals.items() if v is not None}
        return visuals

    def get_losses(self):
        losses = {
            "loss_g_a_to_b": self.loss_g_a_to_b,
            "loss_g_b_to_a": self.loss_g_b_to_a,
            "loss_d_a": self.loss_d_a,
            "loss_d_b": self.loss_d_b,
            "loss_cycle_a": self.loss_cycle_a,
            "loss_cycle_b": self.loss_cycle_b,
            "loss_idt_a": self.loss_idt_a,
            "loss_idt_b": self.loss_idt_b,
            "w_distance_a": self.w_distance_a,
            "w_distance_b": self.w_distance_b,
        }
        # Filter nones
        losses = {k: v for k, v in losses.items() if v is not None}
        return losses
