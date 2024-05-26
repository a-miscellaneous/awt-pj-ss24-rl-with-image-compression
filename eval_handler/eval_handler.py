import abc


class EvalHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, original_image, compressed_image) -> int:
        pass
