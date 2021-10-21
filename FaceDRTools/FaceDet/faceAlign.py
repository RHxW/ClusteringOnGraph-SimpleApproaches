#! -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import os
import copy
import struct

coord5points = [[30.2946, 51.6963],
                [65.5218, 51.6963],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7399, 92.3655]]

def warp_im(im, points1, dshape):
    points2 = np.array(coord5points)

    tform = cv2.estimateAffinePartial2D(points2, points1, ransacReprojThreshold=np.Inf)[0]
    output_im = cv2.warpAffine(im, tform, (dshape[0], dshape[1]), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0])
    return output_im


FLT_EPSILON = 1e-5


def saveNumpy(fname, data):
    sdata = copy.deepcopy(data)
    sdata = sdata.flatten()
    num_len = sdata.shape[0]
    with open(fname, 'wb') as nStream:
        bytei = struct.pack('i', num_len)
        nStream.write(bytei)
        for i in range(num_len):
            bytef = struct.pack('f', sdata[i])
            nStream.write(bytef)
        nStream.flush()
    return


def getTransMatrix(fpoints, std_points):
# def getTransMatrix(std_points, fpoints):
    trans = np.zeros((2, 3))
    points_num = 5.
    sum_x, sum_y = np.sum(std_points, axis=0)
    sum_u, sum_v = np.sum(fpoints, axis=0)
    sum_xx_yy = np.sum(std_points**2)
    sum_ux_vy = np.sum(std_points*fpoints)
    vx_uy = fpoints[:, ::-1]*std_points
    sum_vx__uy = np.sum(vx_uy[:, 0] - vx_uy[:, 1])
    # print("sum_x: ", sum_x)
    # print("sum_y: ", sum_y)
    # print("sum_u: ", sum_u)
    # print("sum_v: ", sum_v)
    # print("sum_xx_yy: ", sum_xx_yy)
    # print("sum_ux_vy: ", sum_ux_vy)
    # print("sum_vx__uy: ", sum_vx__uy)

    if sum_xx_yy <= FLT_EPSILON:
        return None

    q = sum_u - sum_x * sum_ux_vy / sum_xx_yy + sum_y * sum_vx__uy / sum_xx_yy
    p = sum_v - sum_y * sum_ux_vy / sum_xx_yy - sum_x * sum_vx__uy / sum_xx_yy
    r = points_num - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy
    if not (r > FLT_EPSILON or r < -FLT_EPSILON):
        return None
    a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy
    b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy
    c = q / r
    d = p / r

    trans[0, 0] = trans[1, 1] = a
    trans[0, 1] = -b
    trans[1, 0] = b
    trans[0, 2] = c
    trans[1, 2] = d
    # trans = trans[::-1,:]

    return trans


def alignFace(fimage, fmarks, imgSize, scale=1.0):
    assert fimage.ndim >= 2
    assert fmarks.shape[0] == 5 and fmarks.shape[1] == 2
    assert len(imgSize) == 2
    h, w = fimage.shape[:2]
    scale_x = float(w)/float(96)
    scale_y = float(h)/float(112)
    std_marks = np.array([
    30.2946, 51.6963,
    65.5318, 51.5014,
    48.0252, 71.7366,
    33.5493, 92.3655,
    62.7299, 92.2041

    ], dtype=np.float32).reshape(5, 2)
    # print(std_marks)

    std_marks *= [scale_x, scale_y]
    tranMatrix = getTransMatrix(std_marks, fmarks)
    tranMatrix = tranMatrix.astype(np.float32)

    if tranMatrix is not None:
        if scale == 1.0:
            res_image = cv2.warpAffine(fimage, tranMatrix, dsize=(w, h))
        else:
            nw, nh = int(scale*float(w)), int(scale*float(h))
            mw, mh = int((scale-1.)*float(w/2.)), int((scale-1.)*float(h/2.))
            tranMatrix[0, 2] += mw  # move to right
            tranMatrix[1, 2] += mh
            res_image = cv2.warpAffine(fimage, tranMatrix, dsize=(nw, nh))
    else:
        return None, None

    res_image = cv2.resize(res_image, dsize=tuple(imgSize), interpolation=cv2.INTER_CUBIC)
    return res_image


if __name__ == '__main__':
    imgSize = [112, 96]
    # imgSize = [64, 64]

    face_landmarks = np.array([[149.12937927, 244.24391174],
                               [267.15737915, 237.55075073],
                               [204.28997803, 316.4914856],
                               [177.3709259, 325.87362671],
                               [278.78399658, 319.85385132]])

    coord5points = np.array(coord5points, dtype=np.float64)

    face_path = r'F:\facepic\0.jpg'
    face_img = cv2.imread(face_path, cv2.IMREAD_COLOR)
    aligned_face1 = warp_im(face_img, face_landmarks, imgSize)
    aligned_face = alignFace(face_img, face_landmarks, imgSize, scale=1.0)

    cat_align = np.ones([196, 112, 3], dtype=np.uint8) * 0
    cat_align[:96, :, :] = aligned_face1
    cat_align[100:196, :, :] = aligned_face
    cv2.imshow('a', cat_align)
    cv2.waitKey()
