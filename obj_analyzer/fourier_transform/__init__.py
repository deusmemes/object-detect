import cv2 as cv
import numpy as np
from obj_analyzer.fourier_transform.catchbuilder import CatchBuilder


class FourierTransform:
    def __init__(self, img_original, img_pattern):
        self.img_orig = img_original
        self.img_pat = img_pattern

    def check_part(self):
        image = np.asarray(self.img_orig, float) / 255
        pattern = np.asarray(self.img_pat, float) / 255
        builder = CatchBuilder(pattern)
        data = builder.Catch(image)
        y, x = builder.ArgMax(data)
        height, width, channels = pattern.shape
        img, pat = self.img_orig, self.img_pat
        cv.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 1)
        return img, pat
