from .eval_handler import EvalHandler
import numpy as np
from PIL import Image


class PSNR(EvalHandler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, original_image: Image, compressed_image: Image) -> float:
        original = np.array(original_image)
        compressed = np.array(compressed_image)
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
