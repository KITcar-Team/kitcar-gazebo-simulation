import cv2
import torch
import numpy as np
import scipy.misc as scipy
from tensorboardX import SummaryWriter
import color_classes as cc

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TensorboardVisualizerCallback(Callback):
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to display the result
            of the last validation batch in Tensorboard
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def _apply_mask_overlay(self, image, mask, color=(0, 255, 0)):
        #mask = np.dstack((mask, mask, mask)) * color
        mask = mask.astype(np.uint8)
        return cv2.addWeighted(mask, 0.5, image, 0.5, 0.)  # image * α + mask * β + λ

    def _get_mask_representation(self, image, full_mask):
        """
         Given a mask and an image this method returns
         one image representing 3 patches of the same image.
         These patches represent:
            - The original image
            - The original mask
            - The mask applied to the original image
        Args:
            image (np.ndarray): The original image
            mask (np.ndarray): The predicted mask

        Returns (np.ndarray):
            An image of size (original_image_height, (original_image_width * 3))
            showing 3 patches of the original image
        """

        mask = full_mask[1]

        H, W = image.shape
        results = np.zeros((H, 3 * W), np.uint8)
        p = np.zeros((H * W, 1), np.uint8)

        m = np.zeros((H * W), np.uint8)
        l = mask.reshape(-1)
        masked_img = self._apply_mask_overlay(image, mask)

        a = (2 * l + m)
        miss = np.where(a == 2)
        hit = np.where(a == 3)
        fp = np.where(a == 1)
        p[miss] = 255
        p[hit] = 64
        p[fp] = 128
        p = p.reshape(H, W)

        results[:, 0:W] = image
        results[:, W:2 * W] = p
        results[:, 2 * W:3 * W] = masked_img

        return results

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return

        epoch_id = kwargs['epoch_id']
        last_images, last_targets, last_preds = kwargs['last_val_batch']
        writer = SummaryWriter(self.path_to_files)

        for i, (image, target_mask, pred_mask) in enumerate(zip(last_images, last_targets, last_preds)):

            image = image.data.float().cpu().numpy().astype(np.uint8)
            #image = np.transpose(image, (1, 2, 0))  # Invert c, h, w to h, w, c
            target_mask = target_mask.float().data.cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask.float().data.cpu().numpy().astype(np.uint8)
            if image.shape[0] > 256:  # We don't want the images on tensorboard to be too large
                image = scipy.imresize(image, (256, 256))
                target_mask = scipy.imresize(target_mask, (256, 256))
                pred_mask = scipy.imresize(pred_mask, (256, 256))

            expected_result = self._get_mask_representation(image[0], target_mask)
            pred_result = self._get_mask_representation(image[0], pred_mask)

            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Expected', expected_result, epoch_id, dataformats='HW')
            writer.add_image("Epoch_" + str(epoch_id) + '-Image_' + str(i + 1) + '-Predicted', pred_result, epoch_id, dataformats='HW')
            if i == 1:  # 2 Images are sufficient
                break
        writer.close()


class TensorboardLoggerCallback(Callback):
    def __init__(self, path_to_files):
        """
            Callback intended to be executed at each epoch
            of the training which goal is to add valuable
            information to the tensorboard logs such as the losses
            and accuracies
        Args:
            path_to_files (str): The path where to store the log files
        """
        self.path_to_files = path_to_files

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "epoch":
            return

        epoch_id = kwargs['epoch_id']

        writer = SummaryWriter(self.path_to_files)
        writer.add_scalar('data/train_loss', kwargs['train_loss'], epoch_id)
        writer.add_scalar('data/train_acc', kwargs['train_acc'], epoch_id)
        writer.add_scalar('data/val_loss', kwargs['val_loss'], epoch_id)
        writer.add_scalar('data/val_acc', kwargs['val_acc'], epoch_id)
        writer.close()


class ModelSaverCallback(Callback):
    def __init__(self, path_to_model, verbose=False):
        """
            Callback intended to be executed each time a whole train pass
            get finished. This callback saves the model in the given path
        Args:
            verbose (bool): True or False to make the callback verbose
            path_to_model (str): The path where to store the model
        """
        self.verbose = verbose
        self.path_to_model = path_to_model
        self.suffix = ""

    def set_suffix(self, suffix):
        """

        Args:
            suffix (str): The suffix to append to the model file name
        """
        self.suffix = suffix

    def __call__(self, *args, **kwargs):
        if kwargs['step_name'] != "train":
            return

        pth = self.path_to_model + self.suffix
        net = kwargs['net']
        torch.save(net.state_dict(), pth)

        if self.verbose:
            print("Model saved in {}".format(pth))
