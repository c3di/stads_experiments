import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def calculate_psnr(originalImage, noisyImage):
    return psnr(originalImage, noisyImage, data_range = np.max(originalImage)-np.min(originalImage))


def calculate_ssim(noisyImage, originalImage):
    if noisyImage.shape != originalImage.shape:
        raise ValueError("Images must have the same dimensions")

    ssimValue = ssim(noisyImage, originalImage, data_range=np.max(noisyImage)-np.min(noisyImage))

    return ssimValue