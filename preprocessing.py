import cv2

class Preprocessing:
    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        self._height = height
        self._width = width
        self._interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(image, (self._height, self._width), interpolation=self._interpolation)
