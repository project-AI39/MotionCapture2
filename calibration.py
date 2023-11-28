import cv2
import glob
import numpy as np


def calculate_camera_parameters(image_paths, pattern_size):
    # オブジェクト座標と画像座標の対応付けを格納するリスト
    obj_points = []  # オブジェクト座標
    img_points = []  # 画像座標

    # チェスボードのコーナーを検出して対応付ける
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
                -1, 2
            )
            obj_points.append(objp)
            img_points.append(corners)

            # チェスボードのコーナーを描写
            # show1(img, pattern_size, corners, ret)

    # カメラ内部パラメータを算出
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # カメラ内部パラメータを表示
    show2(mtx, dist)

    return mtx, dist


def show1(img, pattern_size, corners, ret):
    # チェスボードのコーナーを描写
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    cv2.imshow("Chessboard Corners", img)
    cv2.waitKey(1000)


def show2(camera_matrix, distortion_coefficients):
    # カメラ内部パラメータを表示
    print("カメラ内部パラメータ:")
    print("カメラ行列:")
    print(camera_matrix)
    print("歪み係数:")
    print(distortion_coefficients)


def calculate_camera_parameters_list(calibration_image_paths_list, pattern_size):
    # カメラ行列と歪み係数のリストを格納する変数
    camera_matrix_list = []
    distortion_coefficients_list = []

    # 各キャリブレーション画像パスに対してカメラパラメータを計算し、リストに追加する
    for calibration_image_paths in calibration_image_paths_list:
        camera_matrix, distortion_coefficients = calculate_camera_parameters(
            calibration_image_paths, pattern_size
        )
        camera_matrix_list.append(camera_matrix)
        distortion_coefficients_list.append(distortion_coefficients)

    return camera_matrix_list, distortion_coefficients_list
