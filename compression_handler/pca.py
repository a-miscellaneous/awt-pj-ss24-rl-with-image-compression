from .compression_handler import CompressionHandler
import io
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np


class cPCA(CompressionHandler):
    def __init__(self):
        super().__init__()
        self.parameter_range = range(1, 201, 10)

    def compress(self, image: Image, parameter) -> bytes:
        img_array = np.array(image)
        size = img_array.shape
        components = min(size[0], size[1], parameter)
        pca = PCA(n_components=components)

        def compress_chanell(pca, array):
            pca.fit(array)
            n_dimensional = pca.transform(array)
            return pca.inverse_transform(n_dimensional)

        colors = []
        for i in range(3):
            colors.append(compress_chanell(pca, img_array[:, :, i]))
        output = np.dstack(colors)
        output = Image.fromarray(output.astype(np.uint8))
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
