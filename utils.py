import cv2
import numpy as np
from PIL import Image
import io


def cv2_to_PIL(cv2_image):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)


def PIL_to_cv2(PIL_image):
    return cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)


def score_image_compression(PIL_image, max_size, compression_handler, evaluation_metric):
    compressed_image = compression_handler.binary_search_size_optimizer(
        PIL_image, max_size)
    if compressed_image is None:
        return -1
    compressed_image.save("compressed_image.jpg", "JPEG", quality=100)
    return evaluation_metric(PIL_image, compressed_image)
