import warnings
import os
import time

import cv2
aruco = cv2.aruco

import csv
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import glob
import argparse
from datetime import datetime
from termcolor import colored

from detector.detector import ArucoDetector

aruco_detector = ArucoDetector()


# used when the default drawing function is not used
def drawdetected(img, ids, corners):
    cv2.line(img, tuple(corners[0][0]), tuple(corners[0][1]), (0, 255, 0), 5)
    cv2.line(img, tuple(corners[0][1]), tuple(corners[0][2]), (0, 255, 0), 5)
    cv2.line(img, tuple(corners[0][2]), tuple(corners[0][3]), (0, 255, 0), 5)
    cv2.line(img, tuple(corners[0][3]), tuple(corners[0][0]), (0, 255, 0), 5)

    t_cood = tuple(corners[0][2])
    t_cood = int(t_cood[0]) + 10, int(t_cood[1])
    marker_id = 'ID:{0:03d}'.format(ids[0])
    detected_img = cv2.putText(img, marker_id, t_cood, cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)

    # perimeter = cv2.arcLength(corners, True)
    # t_cood_2= tuple(corners[0][2])
    # t_cood_2 = int(t_cood_2[0]) + 10, int(t_cood_2[1])
    # marker_id = str(round(perimeter, 2))
    # detected_image = cv2.putText(detected_image, marker_id, t_cood_2, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)

    return detected_img


# # Calculate homography matrix with feature points
# def calc_homography_matrix(baseimg, img):
#     detector = cv2.AKAZE_create()
#     # detector = cv2.ORB_create(500)
#     b = time.time()
#     kp_base, des_base = detector.detectAndCompute(baseimg, None)
#     kp_tag, des_tag = detector.detectAndCompute(img, None)
#     print("detect keypoint：", time.time() - b)

#     ### flann ###
#     c = time.time()
#     FLANN_INDEX_LSH = 6
#     index_params = dict(algorithm=FLANN_INDEX_LSH,
#                         table_number=6,
#                         key_size=12,
#                         multi_probe_level=1)

#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des_base, des_tag, k=2)

#     ### bf ###
#     # bf = cv2.BFMatcher()
#     # matches = bf.knnMatch(des_base, des_tag, k=2)

#     ratio = 0.1
#     good = [[m] for m, n in matches if m.distance < ratio * n.distance]
#     print("good match：", time.time() - c)

#     if len(good) > 10:
#         base_pt = np.float32([kp_base[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         tag_pt = np.float32([kp_tag[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

#         M, _ = cv2.findHomography(tag_pt, base_pt, cv2.RANSAC, 5.0)

#         return M


# Coordinate transformation with homography matrix
def transform_cood(corners, M):
    rotated = []
    for c in corners:
        rotated0 = np.dot(M, np.append(c[0][0], 1))
        rotated1 = np.dot(M, np.append(c[0][1], 1))
        rotated2 = np.dot(M, np.append(c[0][2], 1))
        rotated3 = np.dot(M, np.append(c[0][3], 1))

        rotated0 = (rotated0[:2] / rotated0[2]).astype(np.int64)
        rotated1 = (rotated1[:2] / rotated1[2]).astype(np.int64)
        rotated2 = (rotated2[:2] / rotated2[2]).astype(np.int64)
        rotated3 = (rotated3[:2] / rotated3[2]).astype(np.int64)

        rotated.extend(np.array([[[rotated0, rotated1, rotated2, rotated3]]], dtype=np.float32))

    return rotated


# Delete to remove duplicate coordinates and ids
def deduplication_filer(d_corners, d_ids):
    reject = []
    for i in range(len(d_corners)):
        for j in range(i + 1, len(d_corners)):
            d = d_corners[j][0][0] - d_corners[i][0][0]
            e = d_corners[i][0][3] - d_corners[i][0][0]

            if np.linalg.norm(d) < np.linalg.norm(e):
                reject.append(j)

    for i in set(reject):
        d_corners[i] = " "
        d_ids[i] = " "

    corners = list(filter(lambda s: s != " ", d_corners))
    ids = list(filter(lambda t: t != " ", d_ids))

    return corners, ids


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    s = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--tagpath', type=str, help='PATH of the target directory')
    # parser.add_argument('--createhist', action='store_true', help='Create a histogram')
    parser.add_argument('--createrejectimg', action='store_true', help='Output reject image')
    args = parser.parse_args()

    imgdirpath = args.tagpath

    if os.path.isdir(imgdirpath):
        print('tagpath is directry')
        savedir = imgdirpath + '/result/' + timestamp
        print('savedir:', savedir)
        imglst = glob.glob(args.tagpath + '/*.jpg')
        imglst.sort()

    else:
        print('tagpath is imagefile')
        savedir = os.path.dirname(imgdirpath) + '/result/' + timestamp
        print('savedir:', savedir)
        imglst = [imgdirpath]

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    # dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h10)

    parameters = aruco.DetectorParameters_create()
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 1.0

    
    d_corners, d_ids = aruco_detector.detect(imglst, dictionary, parameters=parameters)
    # d_corners = []
    # d_ids = []
    # for idx, path in enumerate(imglst):
    #     print(colored(os.path.basename(path), 'green'))
    #     img = cv2.imread(path)
    #     if idx == 0:
    #         baseimg = img

    #     corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary, parameters=parameters)

    #     if len(corners) == 0:
    #         print("*** mark is not found ***")
    #         continue
    #     else:
    #         d_ids.extend(ids)

    #         if idx == 0:
    #             d_corners.extend(corners)
    #         else:
    #             M = calc_homography_matrix(baseimg, img)
    #             rotated = transform_cood(corners, M)
    #             d_corners.extend(rotated)

    corners, ids = deduplication_filer(d_corners, d_ids)

    for id, corner in zip(ids, corners):
        dstimg = drawdetected(baseimg, id, corner)

        cv2.imwrite(savedir + "/result.jpg", dstimg)

    print("All time：", time.time() - s)