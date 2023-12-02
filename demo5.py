import cv2
import numpy as np
import dd


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
        [602.1073446327683, 895.425988700565],
        [1373.1917577796467, 129.72918418839362],
        [1318.5394144144145, 184.00225225225225],
        [1373.235841081995, 949.2831783601015],
        [1318.4018161180477, 894.9704880817253],
        [960.1367690782953, 158.16352824578792],
        [960.3856722276741, 348.6182531894014],
        [960.3470588235294, 539.1539215686274],
        [1152.3682132280355, 539.3899308983218],
        [1344.000975609756, 539.3990243902439],
    ]
)
# カメラ2の特徴点座標
points2 = np.array(
    [
        [682.0972222222222, 278.9375],
        [873.8262032085562, 295.7486631016043],
        [682.1353211009174, 800.0206422018349],
        [873.9312169312169, 783.2222222222222],
        [1067.6677631578948, 237.3371710526316],
        [1258.7410881801127, 259.72232645403375],
        [1067.7107438016528, 841.5619834710744],
        [1258.813909774436, 819.2669172932331],
        [960.1551362683438, 269.6687631027254],
        [960.2348008385744, 404.50314465408803],
        [960.1721991701245, 539.1597510373444],
        [1059.8174757281554, 539.1825242718446],
        [1166.9876543209878, 539.2257495590829],
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
points_3d, P1, P2 = triangulate_points(A1, R1, t1, points1, A2, R2, t2, points2)
print(points_3d)
# 結果を表示
for i, point in enumerate(points_3d):
    print(f"Point {i + 1}: {point}")

dd.plot_3d_points_with_arrows(
    points_3d,
    dd.rotation_matrix_to_rotation_vector(R1),
    t1,
    dd.rotation_matrix_to_rotation_vector(R2),
    t2,
)

R2_ = np.array([[1], [0], [0]])

dd.plot_3d_points_with_arrows(
    points_3d,
    dd.rotation_matrix_to_rotation_vector(R1),
    t1,
    R2_,
    t2,
)

#########################################################################
# PnP

success, rvec, tvec = cv2.solvePnP(points_3d, points1, A1, None)
rvec_1 = rvec
tvec_1 = tvec
success, rvec, tvec = cv2.solvePnP(points_3d, points2, A2, None)
rvec_2 = rvec
tvec_2 = tvec

print("rvec_1", rvec_1)
print("tvec_1", tvec_1)
print("rvec_2", rvec_2)
print("tvec_2", tvec_2)


dd.plot_3d_points_with_arrows(points_3d, tvec_1, rvec_1, tvec_2, rvec_2)
dd.plot_3d_points_with_arrows(points_3d, rvec_1, tvec_1, rvec_2, tvec_2)

rvec = np.array([[1], [0], [0]])

dd.plot_3d_points_with_arrows(points_3d, rvec_1, tvec_1, rvec, tvec_2)
