import abc
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# TODO: Typing and comments


class DatasetHandler(metaclass=abc.ABCMeta):
    def __init__(self):
        self.set_name()

    def set_name(self):
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def create_dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_PIL_image(self, row) -> Image.Image:
        pass

    def get_cv2_image(self, row):
        pil_image = self.get_PIL_image(row)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    @abc.abstractmethod
    def get_image_class(self, row) -> str:
        pass

    def get_samples(self, sample_size):
        return self.dataframe.sample(sample_size)

    def get_all_classes(self):
        return self.dataframe.category.unique()
