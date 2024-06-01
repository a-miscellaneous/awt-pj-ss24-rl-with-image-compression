import abc


class EvalHandler(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.set_name()

    def set_name(self):
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def __call__(self, original_image, compressed_image) -> float:
        pass
