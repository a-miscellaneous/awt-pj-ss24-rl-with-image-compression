import abc
from PIL import Image
import io


class CompressionHandler(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.parameter_range = range(1, 101)

    def parameter_range_handler(self, image, eval_functions) -> Image.Image:
        results = {}
        for parameter in self.parameter_range:
            print(
                f"Processing parameter {parameter} for {self.__class__.__name__}...")
            compressed = self.compress(image, parameter)
            size = len(compressed)
            compressed = self.decompress(compressed)
            scores = []
            for eval_func in eval_functions:
                scores.append(eval_func(image, compressed))
            results[parameter] = (size, scores)

        return results

    @abc.abstractmethod
    def compress(self, image, parameter) -> bytes:
        pass

    def decompress(self, compressed: bytes) -> Image.Image:
        return Image.open(io.BytesIO(compressed))

    def __call__(self, image, eval_func) -> Image.Image:
        return self.parameter_range_handler(image, eval_func)
