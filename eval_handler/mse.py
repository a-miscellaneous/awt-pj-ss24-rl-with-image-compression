from .eval_handler import EvalHandler
import numpy as np
from PIL import Image


class MSE(EvalHandler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, original_image: Image, compressed_image: Image) -> float:
        original = np.array(original_image)
        compressed = np.array(compressed_image)
        mse = np.mean((original - compressed) ** 2)
        return mse
