
from .dataset_handler import DatasetHandler
import pandas as pd
import os
from PIL import Image


class Kodak(DatasetHandler):
    def __init__(self):
        super().__init__()
        dataset_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.dirname(dataset_path)
        dataset_path = os.path.join(dataset_path, "datasets")
        self.dataset_path = os.path.join(dataset_path, "kaggle_Kodak")
        self.create_dataframe()

    def create_dataframe(self) -> pd.DataFrame:
        imgs = os.listdir(self.dataset_path)
        classes = ["realism" for img in imgs]
        self.dataframe = pd.DataFrame({"img": imgs, "category": classes})
        return self.dataframe

    def get_PIL_image(self, row) -> Image.Image:
        img_path = os.path.join(self.dataset_path, row.img)
        return Image.open(img_path)

    def get_image_class(self, row) -> str:
        return row.category
