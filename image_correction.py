import numpy as np
import glob
import cv2


def undistort_image(image, mtx, dist):
    # 画像の歪みを補正する関数
    h, w = image.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, mtx, dist, None, new_mtx)
    return undistorted_image


def undistort_images(image_paths, mtx, dist):
    # 複数の画像の歪みを補正する関数
    undistorted_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        undistorted_image = undistort_image(image, mtx, dist)
        undistorted_images.append(undistorted_image)

    show(undistorted_images)

    return undistorted_images


def show(undistorted_images):
    # 表示
    for undistorted_image in undistorted_images:
        cv2.imshow("Undistorted Image", undistorted_image)
        cv2.waitKey(1000)
