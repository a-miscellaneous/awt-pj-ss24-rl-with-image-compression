import abc


class EvalHandler(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, original_image, compressed_image) -> float:
        pass
