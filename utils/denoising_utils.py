import os
from .common_utils import *


def get_noisy_image(img_np: np.ndarray, sigma: float, domain: str = 'xray'):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    noise = np.random.normal(scale=sigma, size=img_np.shape)
    if domain == 'us':
        img_noisy_np = np.exp(np.log(img_np + 1e-6) + noise)
    else:
        img_noisy_np = img_np + noise
    img_noisy_np = np.clip(img_noisy_np, 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
