from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio, 

def ssim(im1, im2):
    return structural_similarity(im1, im2)

def mse(im1, im2):
    return mean_squared_error(im1, im2)

import torch

def snr(im):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a 4D torch tensor (sequence of volumes).
    
    Parameters:
    im (torch.Tensor): Input 4D tensor of shape (T, D, H, W)
    
    Returns:
    torch.Tensor: Signal-to-Noise Ratio for each volume in the sequence
    """
    mean_signal = torch.mean(im, dim=(1, 2, 3))
    std_noise = torch.std(im, dim=(1, 2, 3))
    return 20 * torch.log10(mean_signal / std_noise)

def psnr(im1, im2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two 4D torch tensors (sequences of volumes).
    
    Parameters:
    im1 (torch.Tensor): First input 4D tensor of shape (T, D, H, W)
    im2 (torch.Tensor): Second input 4D tensor of shape (T, D, H, W)
    
    Returns:
    torch.Tensor: Peak Signal-to-Noise Ratio for each pair of volumes in the sequences
    """
    mse = torch.mean((im1 - im2) ** 2, dim=(1, 2, 3))
    max_pixel = torch.max(torch.max(im1), torch.max(im2))
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

