import abc
import pandas as pd
from PIL import Image

# TODO: Typing and comments


class DatasetHandler(metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device
        self.set_name()

    def set_name(self):
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def create_dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_image_tensor(self, row, size=224):
        pass

    @abc.abstractmethod
    def get_PIL_image(self, row, size=224) -> Image.Image:
        pass

    @abc.abstractmethod
    def get_image_type(self, row) -> str:
        pass

    def get_samples(self, sample_size):
        return self.dataframe.sample(sample_size)
