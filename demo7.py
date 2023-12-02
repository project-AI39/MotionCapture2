# import matplotlib.pyplot as plt
# import numpy as np
# import cv2


# def plot_camera_pose(A1, A2, R1, t1, R2, t2, points_3d):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     # 三角測量で点群を計算
#     points_3d = cv2.triangulatePoints(A1 @ R1, A2 @ R2, points1, points2)

#     # PnP解法で姿勢パラメータ計算
#     ret, rvec1, tvec1 = cv2.solvePnP(points_3d, points1, A1)
#     ret, rvec2, tvec2 = cv2.solvePnP(points_3d, points2, A2)

#     # 点群描画
#     ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

#     # カメラ1の位置と姿勢描画
#     ax.scatter(tvec1[0], tvec1[1], tvec1[2], c="r")
#     ax.quiver(tvec1[0], tvec1[1], tvec1[2], rvec1[0], rvec1[1], rvec1[2], length=1)

#     # カメラ2の位置と姿勢描画
#     ax.scatter(tvec2[0], tvec2[1], tvec2[2], c="g")
#     ax.quiver(tvec2[0], tvec2[1], tvec2[2], rvec2[0], rvec2[1], rvec2[2], length=1)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Z")
#     ax.set_zlabel("Y")

#     ax.view_init(elev=45, azim=-45)

#     plt.show()


# return points_3d
