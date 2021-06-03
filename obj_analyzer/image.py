import cv2 as cv
import numpy as np
import base64


def normalize(image, d_size):
    norm_image = np.zeros(d_size)
    final_image = cv.normalize(image, norm_image, 0, 255, cv.NORM_MINMAX)
    return final_image


def resize(image, d_size):
    output = cv.resize(image, (d_size[0], d_size[1]), interpolation=cv.INTER_AREA)
    return output


def get_mask_area(mask):
    count = cv.countNonZero(mask)
    return count


def add_mask(image, mask):
    img_with_mask = image.copy()
    img_with_mask[mask == 1] = (0, 0, 255)
    return img_with_mask


def convert_to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def convert_to_base64(image):
    return base64.b64encode(cv.imencode('.jpg', image)[1]).decode()


def perspective_transform(img_src, img_dst, corners):
    pts_src = np.array([[0, 0], [255, 0], [255, 255], [0, 255]])
    pts_dst = np.array(corners)
    h, status = cv.findHomography(pts_src, pts_dst)
    img_out = cv.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
    return img_out
