import glob
import cv2 as cv
from datetime import datetime

from obj_analyzer import segmentation


def func(i):
    model = segmentation.Unet(f"test{i}")
    model.fit('data')


def test(count):
    model = segmentation.Unet("test")
    model.get_model()

    list_imgs = glob.glob('data/images/*')
    start_time = datetime.now()
    count = int(count / len(list_imgs))

    for i in range(count):
        images = [cv.imread(file) for file in list_imgs]
        masks = [model.predict(img) for img in images]

    print(datetime.now() - start_time)
