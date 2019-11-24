import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import cm
import pandas as pd
import warnings

########################################################
#
#
#     This file contains all the helper functions
#
#
########################################################

SIFT = cv2.xfeatures2d.SIFT_create()
def get_pad(I, h, mode):
    I_hight, I_width = I.shape[:2]
    h_hight, h_width = h.shape[:2]
    if mode == "valid":
        top = 0
        bottom = 0
        left = 0
        right = 0
    elif mode == "same":
        d_hight = h_hight - 1
        d_width = h_width - 1
        top = d_hight//2
        bottom = d_hight - d_hight//2
        left = d_hight//2
        right = d_hight - d_hight//2
    elif mode == "full":
        top = h_hight - 1
        bottom = h_hight - 1
        left = h_width - 1
        right = h_width - 1
    else:
        print("Invalid mode option:{}".format(mode))
        return
    return (top, bottom, left, right)

def get_block(I, h, ver, hor):
    h_w, h_h = h.shape[0:2]
    return I[ver:ver+h_h, hor:hor+h_h]

def zero_padding(I, top, bottom, left, right):
    hight = I.shape[0]+top+bottom
    width = I.shape[1]+left+right
    if len(I.shape) == 3:
        depth = I.shape[2]
        result = np.zeros([hight, width, depth])
    elif len(I.shape) == 2:
        result = np.zeros([hight, width])
    else:
        raise ValueError('dimension mismatch') 
    v_end = -bottom
    h_end = -right
    if not bottom:
        v_end = None
    if not right:
        h_end = None
    result[top:v_end, left:h_end] = I
    return result

def correlate(I, h, pad):
    top, bottom, left, right = pad
    tmp = zero_padding(I, top, bottom, left, right)
    new_hight = tmp.shape[0] - h.shape[0] + 1
    new_width = tmp.shape[1] - h.shape[1] + 1
    result = np.zeros((new_hight, new_width))
#     print("result shape:", result.shape )
    for ver in range(new_hight):
        for hor in range(new_width):
            block = get_block(tmp, h, ver, hor)
            mix = np.sum(block*h)
            result[ver, hor] = mix
    return result

def rotateImage(image, angle):
    # creddit to Tomasz Gandor
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def correlatefft(I, h, pad):
    # Credits to zhangqianhui 
    top, bottom, left, right = pad
    tmp = zero_padding(I, top, bottom, left, right)
    if I.ndim == h.ndim:
        if I.ndim == 2:
            I = I[..., np.newaxis]
            h = h[..., np.newaxis]
    else:
        raise ValueError('dimension mismatch') 
    size = (np.array(I.shape) - np.array(h.shape) + 1)[0:2]
    result = []
    assert I.ndim == 3
    for depth in range(I.shape[-1]):
        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
        fslice = tuple([slice(0, int(s)) for s in size])
        new_I = np.fft.fft2(I[...,depth] , fsize)
        new_h = np.fft.fft2(h[...,depth] , fsize)
        channel = np.fft.ifft2(new_I*new_h)[fslice].copy()
        result.append(channel)
    result = np.dstack(result)
    return result

def MyConvolution(I, h, mode="same"):
    if I.shape[0] < h.shape[0] and I.shape[1] < h.shape[1]:
        I, h = h, I
    h = np.rot90(h, 2)
    padding = get_pad(I, h, mode)
    I = correlatefft(I, h, padding)
    if I.max() > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I = I.astype(np.uint8)
    else:
        I = np.clip(I, 0, 1)
    return I

def downsize(image, i):
    result = image.copy()
    for _ in range(i):
        result = cv2.pyrDown(result)
    return result

def upsize(image, i):
    result = image.copy()
    for _ in range(i):
        result = cv2.pyrUp(result)
    return result

def apply_thres(matched_points):
    results = []
    for thres in np.linspace(1, 0, 500):
        results.append((thres, np.sum(matched_points[...,2]<thres)))
    return np.array(results)

def sort_mp(mp):
    sort_idx = mp[...,2].argsort()
    return mp[sort_idx]

def get_unique(mp):
    sort = sort_mp(mp)
    match = np.concatenate((cv2.KeyPoint_convert(sort[...,0]), cv2.KeyPoint_convert(sort[...,1])), axis=1)
    match = pd.DataFrame(match).reset_index()
    max01 = set(match.groupby([0,1]).max()["index"])
    max23 = set(match.groupby([2,3]).max()["index"])
    unique = match.loc[(max01 & max23)]
    return unique.values[:, 1:]
    
def drawKeypoints(img, kp, color=(255, 0, 0), radius=False):
    img = img.copy()
    if img.ndim == 2:
        img = np.dstack((img, img, img))
    if not radius:
        radius = min(img.shape)// 50
    for p in kp:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness=-1)
    return img
        