import torch
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio

def ssim(im1, im2, data_range=None, multichannel=False):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
    im1 (np.ndarray or torch.Tensor): First input image
    im2 (np.ndarray or torch.Tensor): Second input image
    data_range (float, optional): The data range of the input image (distance between min and max possible values)
    multichannel (bool): Whether the images are multichannel (RGB) or not
    
    Returns:
    float: SSIM value
    """
    if isinstance(im1, torch.Tensor):
        im1 = im1.cpu().numpy()
    if isinstance(im2, torch.Tensor):
        im2 = im2.cpu().numpy()
    
    return structural_similarity(im1, im2, data_range=data_range, multichannel=multichannel)

def mse(im1, im2):
    """
    Calculate the Mean Squared Error (MSE) between two images.
    
    Parameters:
    im1 (np.ndarray or torch.Tensor): First input image
    im2 (np.ndarray or torch.Tensor): Second input image
    
    Returns:
    float: MSE value
    """
    if isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor):
        return torch.mean((im1 - im2) ** 2).item()
    else:
        return mean_squared_error(im1, im2)

def rmse(im1, im2):
    """
    Calculate the Root Mean Squared Error (RMSE) between two images.
    
    Parameters:
    im1 (np.ndarray or torch.Tensor): First input image
    im2 (np.ndarray or torch.Tensor): Second input image
    
    Returns:
    float: RMSE value
    """
    return np.sqrt(mse(im1, im2))

def nrmse(im1, im2):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between two images.
    
    Parameters:
    im1 (np.ndarray or torch.Tensor): First input image
    im2 (np.ndarray or torch.Tensor): Second input image
    
    Returns:
    float: NRMSE value
    """
    rmse_val = rmse(im1, im2)
    if isinstance(im1, torch.Tensor):
        return rmse_val / (torch.max(im1) - torch.min(im1)).item()
    else:
        return rmse_val / (np.max(im1) - np.min(im1))

def snr(im):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of an image or a 4D torch tensor (sequence of volumes).
    
    Parameters:
    im (np.ndarray or torch.Tensor): Input image or 4D tensor of shape (T, D, H, W)
    
    Returns:
    float or torch.Tensor: Signal-to-Noise Ratio
    """
    if isinstance(im, torch.Tensor):
        if im.dim() == 4:
            mean_signal = torch.mean(im, dim=(1, 2, 3))
            std_noise = torch.std(im, dim=(1, 2, 3))
        else:
            mean_signal = torch.mean(im)
            std_noise = torch.std(im)
        return 20 * torch.log10(mean_signal / std_noise)
    else:
        mean_signal = np.mean(im)
        std_noise = np.std(im)
        return 20 * np.log10(mean_signal / std_noise)

def psnr(im1, im2, data_range=None):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images or 4D torch tensors (sequences of volumes).
    
    Parameters:
    im1 (np.ndarray or torch.Tensor): First input image or 4D tensor of shape (T, D, H, W)
    im2 (np.ndarray or torch.Tensor): Second input image or 4D tensor of shape (T, D, H, W)
    data_range (float, optional): The data range of the input image (distance between min and max possible values)
    
    Returns:
    float or torch.Tensor: Peak Signal-to-Noise Ratio
    """
    if isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor):
        if im1.dim() == 4:
            mse = torch.mean((im1 - im2) ** 2, dim=(1, 2, 3))
            max_pixel = torch.max(torch.max(im1), torch.max(im2))
        else:
            mse = torch.mean((im1 - im2) ** 2)
            max_pixel = torch.max(torch.max(im1), torch.max(im2))
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))
    else:
        return peak_signal_noise_ratio(im1, im2, data_range=data_range)

def cnr(im, roi1, roi2):
    """
    Calculate the Contrast-to-Noise Ratio (CNR) between two regions of interest in an image.
    
    Parameters:
    im (np.ndarray or torch.Tensor): Input image
    roi1 (np.ndarray or torch.Tensor): Binary mask for the first region of interest
    roi2 (np.ndarray or torch.Tensor): Binary mask for the second region of interest
    
    Returns:
    float: Contrast-to-Noise Ratio
    """
    if isinstance(im, torch.Tensor):
        mean1 = torch.mean(im[roi1])
        mean2 = torch.mean(im[roi2])
        noise = torch.std(im[roi1 | roi2])
        return torch.abs(mean1 - mean2) / noise
    else:
        mean1 = np.mean(im[roi1])
        mean2 = np.mean(im[roi2])
        noise = np.std(im[roi1 | roi2])
        return np.abs(mean1 - mean2) / noise

def uqi(im1, im2, window_size=8):
    """
    Calculate the Universal Quality Index (UQI) between two images.
    
    Parameters:
    im1 (np.ndarray): First input image
    im2 (np.ndarray): Second input image
    window_size (int): Size of the sliding window
    
    Returns:
    float: Universal Quality Index
    """
    N = window_size ** 2
    sum1 = np.sum(im1, axis=(0, 1))
    sum2 = np.sum(im2, axis=(0, 1))
    sum1_sq = np.sum(im1 ** 2, axis=(0, 1))
    sum2_sq = np.sum(im2 ** 2, axis=(0, 1))
    sum_12 = np.sum(im1 * im2, axis=(0, 1))
    
    mu1 = sum1 / N
    mu2 = sum2 / N
    var1 = (sum1_sq - N * mu1 ** 2) / (N - 1)
    var2 = (sum2_sq - N * mu2 ** 2) / (N - 1)
    cov = (sum_12 - N * mu1 * mu2) / (N - 1)
    
    numerator = 4 * cov * mu1 * mu2
    denominator = (var1 + var2) * (mu1 ** 2 + mu2 ** 2)
    
    return np.mean(numerator / denominator)

def noise_power_spectrum(im):
    """
    Calculate the Noise Power Spectrum (NPS) of an image.
    
    Parameters:
    im (np.ndarray): Input image
    
    Returns:
    np.ndarray: 2D Noise Power Spectrum
    """
    ft = np.fft.fft2(im)
    return np.abs(ft) ** 2 / (im.shape[0] * im.shape[1])

def modulation_transfer_function(im, edge_roi):
    """
    Calculate the Modulation Transfer Function (MTF) of an image using the edge method.
    
    Parameters:
    im (np.ndarray): Input image
    edge_roi (tuple): Region of interest containing the edge (y1, y2, x1, x2)
    
    Returns:
    tuple: (spatial_frequencies, mtf_values)
    """
    y1, y2, x1, x2 = edge_roi
    edge = im[y1:y2, x1:x2]
    
    # Calculate Edge Spread Function (ESF)
    esf = np.mean(edge, axis=0)
    
    # Calculate Line Spread Function (LSF)
    lsf = np.diff(esf)
    
    # Calculate MTF
    mtf = np.abs(np.fft.fft(lsf))
    mtf = mtf[:len(mtf)//2]  # Keep only the positive frequencies
    mtf /= mtf[0]  # Normalize
    
    spatial_frequencies = np.fft.fftfreq(len(lsf))[:len(mtf)]
    
    return spatial_frequencies, mtf

def dice_coefficient(im1, im2):
    """
    Calculate the Dice Coefficient between two binary masks.
    
    Parameters:
    im1 (np.ndarray): First input binary mask
    im2 (np.ndarray): Second input binary mask
    
    Returns:
    float: Dice Coefficient
    """
    return np.sum(im1 & im2) / np.sum(im1 | im2)