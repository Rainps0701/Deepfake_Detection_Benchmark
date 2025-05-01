#!/usr/bin/env mdl
import imageio
import numpy as np
from io import BytesIO

def jpeg_encode(img: np.array, quality=80):
    img_byte_arr = BytesIO()
    imageio.imwrite(img_byte_arr, img, format='JPEG', quality=quality)
    return img_byte_arr.getvalue()

def jpeg_decode(code: bytes):
    img = imageio.imread(code)
    return img

# vim: ts=4 sw=4 sts=4 expandtab
