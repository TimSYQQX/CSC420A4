import numpy as np
import cv2
import matplotlib.pyplot as plt
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
        img = cv2.circle(img, (int(p.pt[0]), int(p.pt[1])), radius, color, thickness=-1)
    return img

 
def normalize_des(des):
    norm_des = des / np.sum(des, axis=1)[...,np.newaxis]
    norm_des = np.clip(norm_des, 0, 0.2)
    norm_des = norm_des / np.sum(norm_des, axis=1)[...,np.newaxis]
    return norm_des

 
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

 
def draw_matching_lines(img1, img2, match, color=(255,0,0), size=5):
    h = img1.shape[1]
    img = np.hstack((img1, img2))
    for i in range(match.shape[0]):
        pt1 = np.array(match[i, 0].pt).astype(int)
        pt2 = np.array(match[i, 1].pt).astype(int)
        pt2[0] += h
        img = cv2.line(img, tuple(pt1), tuple(pt2), color, size)
    return img


def get_sparse(mp, num=8, min_dist=400, best=True):
    if best:
        mp = sort_mp(mp)
    pairs = []
    for p in mp:
        if len(pairs) == num:
            return np.array(pairs)
        if not pairs:
            pairs.append(p)
        else:
            far = True
            for pair in pairs:
                if np.sqrt(np.sum((np.array(pair[0].pt)-np.array(p[0].pt))**2)) < min_dist:
                    far = False
                    break
                if np.sqrt(np.sum((np.array(pair[1].pt)-np.array(p[1].pt))**2)) < min_dist:
                    far = False
                    break
            if far:
                pairs.append(p)  
    print("No solution")

def get_match(img1, img2):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    mp = match(des1, des2, kp1, kp2, L=2)
    return mp

def draw_match(img1, img2, mp, file):
    pt1, pt2 = drawKeypoints(img1, mp[...,0], radius=50), drawKeypoints(img2, mp[...,1], radius=50)
    line = draw_matching_lines(pt1, pt2, mp)
    fig = figure(dpi=400)
    imshow(line)
    savefig(file)
    
def get_F(sparse):
    pts1 = [i[0].pt for i in sparse]
    pts2 = [i[1].pt for i in sparse]
    F, _ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
    return F*2