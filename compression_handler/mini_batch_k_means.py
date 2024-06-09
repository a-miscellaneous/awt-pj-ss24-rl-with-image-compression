from .compression_handler import CompressionHandler
import io
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import numpy as np


class KMeans(CompressionHandler):
    def __init__(self):
        super().__init__()
        self.parameter_range = range(1, 101, 10)

    def compress(self, image: Image, parameter) -> bytes:
        clusterer = MiniBatchKMeans(n_clusters=parameter, n_init='auto')
        img_array = np.array(image)
        size = img_array.shape
        reshaped = img_array.reshape(size[0] * size[1], size[2])
        clusterer.fit(reshaped)
        output = clusterer.cluster_centers_[clusterer.labels_]
        output = output.reshape(size)
        output = Image.fromarray(output.astype(np.uint8))
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
