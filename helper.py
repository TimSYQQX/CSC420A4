import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import cm
import pandas as pd
import warnings
from numba import jit

########################################################
#
#
#     This file contains all the helper functions
#
#
########################################################

SIFT = cv2.xfeatures2d.SIFT_create()

def apply_thres(matched_points):
    results = []
    for thres in np.linspace(1, 0, 500):
        results.append((thres, np.sum(matched_points[...,2]<thres)))
    return np.array(results)

@jit
def sort_mp(mp):
    sort_idx = mp[...,2].argsort()
    return mp[sort_idx]

@jit
def get_unique(mp):
    sort = sort_mp(mp)
    match = np.concatenate((cv2.KeyPoint_convert(sort[...,0]), cv2.KeyPoint_convert(sort[...,1])), axis=1)
    match = pd.DataFrame(match).reset_index()
    max01 = set(match.groupby([0,1]).max()["index"])
    max23 = set(match.groupby([2,3]).max()["index"])
    unique = match.loc[(max01 & max23)]
    return unique.values[:, 1:]

@jit    
def drawKeypoints(img, kp, color=(255, 0, 0), radius=False):
    img = img.copy()
    if img.ndim == 2:
        img = np.dstack((img, img, img))
    if not radius:
        radius = min(img.shape)// 50
    for p in kp:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness=-1)
    return img

@jit
def normalize_des(des):
    norm_des = des / np.sum(des, axis=1)[...,np.newaxis]
    norm_des = np.clip(norm_des, 0, 0.2)
    norm_des = norm_des / np.sum(norm_des, axis=1)[...,np.newaxis]
    return norm_des

@jit
def match(des1, des2, kp1, kp2, L=2):
    norm1 = normalize_des(des1)
    norm2 = normalize_des(des2)
    matched_points = []
    for v in range(norm1.shape[0]):
        dist = np.sum(np.abs(norm1[v] - norm2)**L, axis=1)**(1/L)
        min_idx = dist.argmin()
        min_value = dist[min_idx]
        dist[min_idx] = float("inf")
        min2_idx = dist.argmin()
        ratio = min_value/dist[min2_idx]
        matched_points.append((kp1[v], kp2[min_idx], ratio))
    return np.array(matched_points)

@jit
def draw_matching_lines(img1, img2, match, color=(255,0,0), size=5):
    h = img1.shape[1]
    img = np.hstack((img1, img2))
    for i in range(match.shape[0]):
        pt1 = match[i, 0:2].astype(int)
        pt2 = match[i, 2:4].astype(int)
        pt2[0] += h
        img = cv2.line(img, tuple(pt1), tuple(pt2), color, size)
    return img