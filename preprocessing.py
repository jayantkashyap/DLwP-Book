import cv2
import numpy as np

class Preprocessing:
    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        self._height = height
        self._width = width
        self._interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(image, (self._height, self._width), interpolation=self._interpolation)

def image2vector(im):
    if len(im.shape) == 2:
        return im.reshape((im.shape[0]*im.shape[1], 1))
    else:
        return im.reshape((im.shape[0]*im.shape[1]*im.shape[2], 1))


def normalize_rows(x, axis=1):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x/x_norm