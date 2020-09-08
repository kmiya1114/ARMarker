#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
import argparse

aruco = cv2.aruco

# def creatDic(num):
#     if
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h10)
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
# dictionary = aruco.Dictionary_create_from(50, 6, aruco.getPredefinedDictionary(aruco.DICT_6X6_50))
# dictionary = aruco.Dictionary_create_from(1050, 6, aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h10))
print(dictionary.markerSize)
    # return


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, help='Marker ID')
parser.add_argument('--outdir', type=str, default='/Users/katsuhiro/Desktop/mark', help='Marker save directry')
# parser.add_argument('--createhist', action='store_true', help='Create a histogram')
parser.add_argument('--addmargin', action='store_true', help='Generate markers with margins')
args = parser.parse_args()

## create ar marker
def arGenerator(code_id, output_dir):
    # print(code_id)
    fileName = output_dir + "/ar_{0:04d}.png".format(code_id)
    generator = aruco.drawMarker(dictionary, code_id, 300)

    if args.addmargin:

        height = (generator.shape[0] // 6) * 8
        width = (generator.shape[0] // 6) * 8
        img_white = np.ones((height, width), np.uint8) * 255

        # src2 = cv2.imread('data/src/horse_r.png')
        # generator = cv2.resize(generator, img_white.shape[1::-1])
        # print(generator.shape)

        dst = cv2.bitwise_and(img_white, generator)
        cv2.imwrite(fileName, dst)

    else:
        cv2.imwrite(fileName, generator)


def main():
    # arGenerator(args.id, args.outdir)

    outdir = '/Users/katsuhiro/Work/task/CartPossioning/mark2'

    for i in range(35, 36):
        # code_id = random.randint(0, 1050)
        code_id = i
        # print('code_id', code_id)
        arGenerator(code_id, outdir)
            


if __name__ == '__main__':
    main()