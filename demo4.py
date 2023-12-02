import math
import cv2
import numpy as np


def triangulate_points(A1, R1, t1, points1, A2, R2, t2, points2):
    # カメラ1の射影行列P1を計算
    P1 = np.hstack((R1, t1))
    print(P1, "\n")
    P1 = np.dot(A1, P1)
    print(P1, "\n")

    # カメラ2の射影行列P2を計算
    P2 = np.hstack((R2, t2))
    print(P2, "\n")
    P2 = np.dot(A2, P2)
    print(P2, "\n")

    # 特徴点を正規化座標に変換
    points1 = np.array(points1).T
    points2 = np.array(points2).T

    # 三角測量を実行
    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)

    # 四次元座標を三次元座標に変換
    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d.T, P1, P2


#########################################################################
## 入力値

# 相対的な回転行列
R = np.array(
    [
        [7.06885281e-01, 3.21060402e-03, 7.07320925e-01],
        [-5.47419565e-03, 9.99984582e-01, 9.31788934e-04],
        [-7.07307028e-01, -4.53068102e-03, 7.06891958e-01],
    ]
)
# 相対的な並進ベクトル
t = np.array(
    [
        [-0.71426682],
        [0.00507821],
        [0.69985508],
    ]
)
# カメラ1内部パラメータ行列
A1 = np.array(
    [
        [2.67126877e03, 0.00000000e00, 9.57917092e02],
        [0.00000000e00, 2.67168557e03, 5.19849867e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# カメラ2内部パラメータ行列
A2 = np.array(
    [
        [2.67126877e03, 0.00000000e00, 9.57917092e02],
        [0.00000000e00, 2.67168557e03, 5.19849867e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
# カメラ1の特徴点座標
points1 = np.array(
    [
        [547.1563275434244, 129.1439205955335],
        [602.1808873720137, 183.5551763367463],
        [547.1986809563067, 949.5375103050288],
    ]
)
# カメラ2の特徴点座標
points2 = np.array(
    [
        [682.0972222222222, 278.9375],
        [873.8262032085562, 295.7486631016043],
        [682.1353211009174, 800.0206422018349],
    ]
)
#########################################################################
## 世界座標系におけるカメラの回転行列と並進ベクトル

# カメラ1の回転行列と並進ベクトルを世界座標系とする
R1 = np.eye(3)
t1 = np.array([[0], [0], [0]])

# カメラ2の回転行列と並進ベクトル
R2 = R
t2 = t

# 三角測量を実行
triangulated_points, P1, P2 = triangulate_points(
    A1, R1, t1, points1, A2, R2, t2, points2
)

# 結果を表示
for i, point in enumerate(triangulated_points):
    print(f"Point {i + 1}: {point}")


# カメラ1位置ベクトル
cam1_pos = [0, 0, 0]

# カメラ2位置ベクトル
cam2_pos = t2.ravel()

# カメラ1回転ベクトル
cam1_vecter = [0, 0, 0]

# カメラ2回転ベクトル
cam2_vecter, _ = cv2.Rodrigues(R2)

# カメラ2回転角度
x = cam2_vecter[0]
y = cam2_vecter[1]
z = cam2_vecter[2]
angle_x = math.degrees(math.atan2(y, x))
angle_y = math.degrees(math.atan2(z, x))
angle_z = math.degrees(math.atan2(y, z))


# 結果表示
print("Camera 1:")
print(" Positionベクトル:", cam1_pos)
print(" Rotationベクトル:", cam1_vecter)

print("Camera 2:")
print(" Positionベクトル:", cam2_pos)
print(" Rotationベクトル:", cam2_vecter)
print(" Rotation角度:", angle_x, angle_y, angle_z)
