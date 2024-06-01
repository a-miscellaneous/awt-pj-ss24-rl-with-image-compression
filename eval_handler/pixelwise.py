from .eval_handler import EvalHandler
import numpy as np
from PIL import Image


class Pixelwise(EvalHandler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, original_image: Image, compressed_image: Image) -> float:
        original = np.array(original_image)
        compressed = np.array(compressed_image)
        difference = np.abs(original - compressed)
        sum_difference = np.sum(difference)
        dif = sum_difference / \
            (original.shape[0] * original.shape[1] * original.shape[2])
        inverted_dif = 255 - dif
        return inverted_dif
