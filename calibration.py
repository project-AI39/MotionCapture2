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
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow("Chessboard Corners", img)
            cv2.waitKey(1000)

    # カメラ内部パラメータを算出
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    show(mtx, dist)

    return mtx, dist


def show(camera_matrix, distortion_coefficients):
    # カメラ内部パラメータを表示
    print("カメラ内部パラメータ:")
    print("カメラ行列:")
    print(camera_matrix)
    print("歪み係数:")
    print(distortion_coefficients)
