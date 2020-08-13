import torch
from torch import nn as nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label
    tensor that has the same size as the input.
    """

    def __init__(self, gan_mode: str):
        """Initialize GANLoss.

        Note: Do not use sigmoid as the last layer of Discriminator. LSGAN
        needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.

        Args:
            gan_mode (str): The type of GAN objective. It currently supports
                vanilla, lsgan, and wgangp.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        """Create label tensors with the same size as the input.

        Args:
            prediction (torch.Tensor): typically the prediction from a
                discriminator
            target_is_real (bool): if the ground truth label is for real images
                or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of
            the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> float:
        """Calculate loss given Discriminator's output and ground truth labels.

        Args:
            prediction (torch.Tensor): typically the prediction from a
                discriminator
            target_is_real (bool): if the ground truth label is for real images
                or fake images

        Returns:
            the calculated loss.
        """
        loss = None
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
