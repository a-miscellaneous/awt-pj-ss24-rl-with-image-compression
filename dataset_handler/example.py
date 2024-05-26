from .dataset_handler import DatasetHandler
import torchvision
import numpy as np
import pandas as pd
from PIL import Image

pascal_labels = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv/monitor",
}


class PascalDataset(DatasetHandler):
    def __init__(self, device, transform, inverse_transform, target_transform):
        super().__init__(device)
        self.labels = pascal_labels
        self.transform = transform
        self.target_transform = target_transform
        self.inverse_transform = inverse_transform
        self.create_dataframe()

    def create_dataframe(self):
        self.dataset = torchvision.datasets.VOCSegmentation(
            root="datasets/VOCSegmentation",
            year="2012",
            image_set="val",
            download=False,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        data = []

        for idx, (_, mask) in enumerate(self.dataset):
            mask = np.array(mask)
            labels = np.unique(mask)
            labels = labels[(labels != 0) & (labels != 255)]

            data.append(
                {
                    "dataset_idx": idx,
                    "labels": labels,
                    "label_count": len(labels),
                    "label_names": [pascal_labels[label] for label in labels],
                }
            )

        # Create a pandas dataframe
        self.dataframe = pd.DataFrame(data)
        self.dataframe = self.dataframe[self.dataframe["label_count"] == 1].copy(
        )

        exclude_classes = [
            "person",
            "tv/monitor",
            "sofa",
            "potted plant",
            "diningtable",
            "chair",
            "bottle",
        ]
        self.dataframe = self.dataframe[
            ~self.dataframe["label_names"].apply(
                lambda x: any(cls in x for cls in exclude_classes))
        ].copy()

    def get_image_tensor(self, row, size=224):
        return self.dataset[row.dataset_idx][0].unsqueeze(0).to(self.device)

    def get_PIL_image(self, row, size=224):
        img = self.inverse_transform(
            self.dataset[row.dataset_idx][0]).permute(1, 2, 0).numpy()
        return img

    def get_mask(self, row, size=224):
        seg = np.array(self.dataset[row.dataset_idx][1])
        mask = np.zeros_like(seg).astype(np.float32)
        mask[seg == 255] = 1
        return mask

    def get_true_label(self, row):
        seg = np.array(self.dataset[row.dataset_idx][1])
        seg[seg == 255] = 0
        seg_classes = np.unique(seg)
        seg_classes = seg_classes[seg_classes != 0]
        seg_labels = [self.labels[i] for i in seg_classes]
        seg_labels = ", ".join(seg_labels)
        return seg_labels

    def get_image_type(self, row) -> str:
        return "TODO"
