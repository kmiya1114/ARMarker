#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
import argparse
import sys

sys.path.append('../')
import aruco_dict

aruco = cv2.aruco

print(ArucoDict)

# aruco_dictionary = dict(
#     dic00=aruco.DICT_4X4_50,
#     dic01=aruco.DICT_4X4_100,
#     dic02=aruco.DICT_4X4_250,
#     dic03=aruco.DICT_4X4_1000,
#     dic04=aruco.DICT_5X5_50,
#     dic05=aruco.DICT_5X5_100,
#     dic06=aruco.DICT_5X5_250,
#     dic07=aruco.DICT_5X5_1000,
#     dic08=aruco.DICT_6X6_50,
#     dic09=aruco.DICT_6X6_100,
#     dic10=aruco.DICT_6X6_250,
#     dic11=aruco.DICT_6X6_1000,
#     dic12=aruco.DICT_7X7_50,
#     dic13=aruco.DICT_7X7_100,
#     dic14=aruco.DICT_7X7_250,
#     dic15=aruco.DICT_7X7_1000,
#     dic16=aruco.DICT_ARUCO_ORIGINAL,
#     dic17=aruco.DICT_APRILTAG_16h5,
#     dic18=aruco.DICT_APRILTAG_25h9,
#     dic19=aruco.DICT_APRILTAG_36h10,
#     dic20=aruco.DICT_APRILTAG_36h11
#     )

## create ar marker
def marker_generator(code_id, output_dir):
    fileName = output_dir + "ar_{0:04d}.png".format(code_id)
    generator = aruco.drawMarker(dictionary, code_id, 300)

    if args.addmargin:

        height = (generator.shape[0] // 6) * 8
        width = (generator.shape[0] // 6) * 8
        img_white = np.ones((height, width), np.uint8) * 255
        dst = cv2.bitwise_and(img_white, generator)
        cv2.imwrite(fileName, dst)

    else:
        cv2.imwrite(fileName, generator)
            

if __name__ == '__main__':
    # binary = "4X4_50"
    dictionary = aruco.getPredefinedDictionary(aruco_dictionary['dic00'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='Marker ID')
    parser.add_argument('--outdir', type=str, default='./', help='Marker save directry')
    parser.add_argument('--addmargin', action='store_true', help='Generate markers with margins')
    args = parser.parse_args()

    marker_generator(args.id, args.outdir)