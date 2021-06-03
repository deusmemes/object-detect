from __future__ import print_function
import sys

PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import cv2 as cv

# built-in modules
from contextlib import contextmanager


def clock():
    return cv.getTickCount() / cv.getTickFrequency()


@contextmanager
def Timer(msg):
    print(msg, '...', )
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock() - start) * 1000))
