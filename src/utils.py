import cv2
import numpy as np
from PIL import Image


def cv2_to_PIL(cv2_image):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)


def PIL_to_cv2(PIL_image):
    return cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
