from .eval_handler import EvalHandler
import numpy as np
from PIL import Image
import cv2


class SSIM(EvalHandler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, original_image: Image, compressed_image: Image) -> float:
        # greyscale
        original = np.array(original_image.convert('L'))
        compressed = np.array(compressed_image.convert('L'))
        original = original.astype(np.float64)
        compressed = compressed.astype(np.float64)
        L = 255
        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.T)
        mu1 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(compressed, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(
            original ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(
            compressed ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(original * compressed, -1,
                               window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
            ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()
