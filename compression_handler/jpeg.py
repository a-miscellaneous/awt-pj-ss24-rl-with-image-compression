from .compression_handler import CompressionHandler
import io
from PIL import Image


class JPEG(CompressionHandler):
    def __init__(self):
        super().__init__()
        self.parameter_range = range(1, 101, 5)

    def compress(self, image: Image, parameter) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=parameter)
        buffer.seek(0)
        return buffer.getvalue()
