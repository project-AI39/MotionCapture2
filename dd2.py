import matplotlib.pyplot as plt
import numpy as np


def plot_cameras_and_points(points_3d, R1, t1, R2, t2):
    # 3D描画の設定
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 点群をプロット
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

    # カメラ1の位置と方向を可視化
    x1, y1, z1 = t1[0], t1[1], t1[2]
    ax.scatter(x1, y1, z1, c="r")
    ax.quiver(x1, y1, z1, R1[0, 0], R1[1, 0], R1[2, 0], length=1, normalize=True)

    # カメラ2の位置と方向を可視化
    x2, y2, z2 = t2[0], t2[1], t2[2]
    ax.scatter(x2, y2, z2, c="g")
    ax.quiver(x2, y2, z2, R2[0, 0], R2[1, 0], R2[2, 0], length=1, normalize=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # ax.view_init(elev=45, azim=-45)

    plt.show()


# サンプルデータ
# points_3d = np.random.randint(0, 10, size=(10, 3))
# R1 = np.eye(3)
# t1 = np.zeros(3)
# R2 = np.eye(3)
# t2 = np.array([1, 0, 0])

# plot_cameras_and_points(points_3d, R1, t1, R2, t2)
