import abc
from PIL import Image
import io


class CompressionHandler(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.parameter_range = range(1, 101)

    def binary_search_size_optimizer(self, image, max_size) -> Image.Image:
        search_space = list(self.parameter_range)
        left = 0
        right = len(search_space) - 1
        middle = left + (right - left) // 2
        while left < right:
            compressed_image = self.compress(image, search_space[middle])
            if len(compressed_image) > max_size:
                right = middle - 1
            else:
                left = middle + 1
            middle = left + (right - left) // 2
        compressed_image = self.compress(image, search_space[middle])
        if len(compressed_image) > max_size:
            return None
        return Image.open(io.BytesIO(compressed_image))

    @abc.abstractmethod
    def compress(self, image, parameter) -> bytes:
        pass

    def __call__(self, image, max_size) -> Image.Image:
        return self.binary_search_size_optimizer(image, max_size)
