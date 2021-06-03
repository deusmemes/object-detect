import cv2 as cv
import numpy as np


def generate_random(count: int):
    imgs = []
    masks = []

    for i in range(count):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        w = np.random.randint(10, 100, 1)[0]
        h = np.random.randint(10, 100, 1)[0]
        x = np.random.randint(10, 100, 1)[0]
        y = np.random.randint(10, 100, 1)[0]

        mask = black.copy()
        black[x:(x + w), y:(y + h), :] = (0, 255, 255)
        mask[x:(x + w), y:(y + h), :] = (255, 255, 255)
        imgs.append(black)
        masks.append(mask)

    return imgs, masks
