from __future__ import print_function

import cv2 as cv
import numpy as np
import itertools as it
from multiprocessing.pool import ThreadPool

from obj_analyzer.keypoints.common import Timer
from obj_analyzer.keypoints.find_obj import init_feature, filter_matches, explore_match
import obj_analyzer.image as iu


def __affine_skew(tilt, phi, img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def __affine_detect(detector, img, mask=None, pool=None):
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = __affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i + 1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


def draw_key_points(img1, img2):
    img1 = iu.convert_to_gray(img1)
    img2 = iu.convert_to_gray(img2)
    detector, matcher = init_feature()
    pool = ThreadPool(processes=cv.getNumberOfCPUs())
    kp1, desc1 = __affine_detect(detector, img1, pool=pool)
    kp2, desc2 = __affine_detect(detector, img2, pool=pool)

    with Timer('matching'):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))
    final_img, corners = explore_match(img1, img2, kp_pairs, None, H)
    return final_img, corners
