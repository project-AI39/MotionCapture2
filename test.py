import cv2
import glob
import os
import numpy as np

import calibration
import image_correction
import coler_marker_detection
import get_calibration_image_paths_list


# calibration画像ファイルのパスを取得
base_path = "./bord"
calibration_image_paths_list = get_calibration_image_paths_list.get_image_paths_list(
    base_path
)

# point画像ファイルのパスを取得
point_images = glob.glob("./point_images/*.png")

# キャリブレーションに使用するパラメータ
pattern_size = (9, 8)  # チェスボードの内部コーナーの数

# カメラ内部パラメータを計算
(
    camera_matrix_list,
    distortion_coefficients_list,
) = calibration.calculate_camera_parameters_list(
    calibration_image_paths_list, pattern_size
)


# 画像の歪みを補正
undistorted_images = image_correction.undistort_images(
    point_images, camera_matrix_list, distortion_coefficients_list
)


# マーカーの中心座標を取得
center_coordinates_list = coler_marker_detection.get_marker_center_coordinates(
    undistorted_images
)
