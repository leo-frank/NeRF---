import torch
from kornia.losses import ssim as dssim


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    Computes the Mean Squared Error (MSE) between image_pred and image_gt.

    Args:
        image_pred (torch.Tensor): Predicted image tensor of shape (B, C, H, W).
        image_gt (torch.Tensor): Ground truth image tensor of shape (B, C, H, W).
        valid_mask (torch.Tensor, optional): Validity mask tensor of shape (B, H, W).
        reduction (str, optional): Type of reduction. Options: 'mean' (default), 'none'.

    Returns:
        torch.Tensor: MSE value or tensor of shape (B) if reduction is 'mean' (default),
                     or tensor of shape (B, H, W) if reduction is 'none'.

    """
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def compute_psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between image_pred and image_gt.

    Args: All input images must are normalized !!!
        image_pred (torch.Tensor): Predicted image tensor of shape (B, C, H, W).
        image_gt (torch.Tensor): Ground truth image tensor of shape (B, C, H, W).
        valid_mask (torch.Tensor, optional): Validity mask tensor of shape (B, H, W).
        reduction (str, optional): Type of reduction. Options: 'mean' (default), 'none'.

    Returns:
        torch.Tensor: PSNR value or tensor of shape (B) if reduction is 'mean' (default),
                     or tensor of shape (B, H, W) if reduction is 'none'.

    """
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
