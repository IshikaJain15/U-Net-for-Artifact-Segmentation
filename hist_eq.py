import cv2
import argparse
import numpy as pb
import glob
import cv2

def hist_equalization(img):
    """ Normal Histogram Equalization

    Args:
        img : image input with single channel

    Returns:
        : Equalized Image
    """
    array = pb.asarray(img)
    array = array.astype(pb.uint8)
    bin_cont = pb.bincount(array.flatten(), minlength=256)
    pixels = pb.sum(bin_cont)
    bin_cont = bin_cont / pixels
    cumulative_sumhist = pb.cumsum(bin_cont)
    map = pb.floor(255 * cumulative_sumhist).astype(pb.uint8)
    arr_list = list(array.flatten())
    eq_arr = [map[p] for p in arr_list]
    arr_back = pb.reshape(pb.asarray(eq_arr), array.shape)
    return arr_back


def ahe(img, rx=85, ry=85):
    """ Adaptive Histogram Equalization

    Args:
        img : image input with single channel
        rx (int, optional): to divide horizontal regions, Note: Should be divisible by image size in x . Defaults to 136.
        ry (int, optional): to divide vertical regions, Note: Should be divisible by image size in y. Defaults to 185.

    Returns:
        : Equalized Image
    """
    v = img
    img_eq = pb.empty((v.shape[0], v.shape[1]), dtype=pb.uint8)
    for i in range(0, v.shape[1], rx):
        for j in range(0, v.shape[0], ry):
            t = v[j:j + ry, i:i + rx]
            c = hist_equalization(t)
            img_eq[j:j + ry, i:i + rx] = c
    return img_eq

def clahe(img, clip_limit=2.0, grid_size=(17, 17)):
    """ Contrast Limited Adaptive Histogram Equalization (CLAHE)

    Args:
        img : image input with single channel
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        grid_size (tuple, optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        : Equalized Image
    """
    array = pb.asarray(img)
    array = array.astype(pb.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img_eq = clahe.apply(img)
    return img_eq