# modifing the image values
import cv2
import numpy as np

class SpatialProcessor:
    def __init__(self, size=84):
        self.size=size
    def process(self, frame):
        gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#Y = 0.299R + 0.587G + 0.114B$
        cropped=gray[34:194, :160]
        resized=cv2.resize(cropped, (self.size, self.size), interpolation=cv2.INTER_AREA)
        normalized=resized.astype(np.float32)/255.0
        return normalized
    