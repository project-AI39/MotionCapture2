import cv2
import glob
import numpy as np

import calibration
import image_correction
import coler_marker_detection

# calibration画像ファイルのパスを取得
calibration_image_paths = glob.glob("./bord/*.png")

# point画像ファイルのパスを取得
point_images = glob.glob("./point_images/*.png")

# キャリブレーションに使用するパラメータ
pattern_size = (9, 8)  # チェスボードの内部コーナーの数

# カメラ内部パラメータを計算
camera_matrix, distortion_coefficients = calibration.calculate_camera_parameters(
    calibration_image_paths, pattern_size
)

# 画像の歪みを補正
undistorted_images = image_correction.undistort_images(
    point_images, camera_matrix, distortion_coefficients
)

# マーカーの中心座標を取得
center_coordinates_list = coler_marker_detection.get_marker_center_coordinates(
    undistorted_images
)

# print(center_coordinates_list)
for i, center_coordinates in enumerate(center_coordinates_list):
    # i番目のカメラの中心座標のリストを取得
    for j, center_coordinate in enumerate(center_coordinates):
        # j番目のマーカーの中心座標を取得
        print(f"{i}番目のカメラの{j}番目のマーカーの中心座標: {center_coordinate}")
