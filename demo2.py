import numpy as np
from scipy.linalg import svd


def _triangulate_one_point(P1, P2, pt1, pt2):
    """一つの点に対する三角測量"""

    x1, y1 = pt1
    x2, y2 = pt2

    X = np.vstack(
        [
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :],
        ]
    )

    _, _, Vt = svd(X)
    x_w = Vt[-1]
    x_w /= x_w[-1]  # 正規化

    return x_w


def triangulate(P1, P2, points1, points2):
    """複数の点に対する三角測量"""

    assert points1.shape == points2.shape

    X_w = []
    for pt1, pt2 in zip(points1, points2):
        x_w = _triangulate_one_point(P1, P2, pt1, pt2)
        X_w.append(x_w)

    X_w = np.vstack(X_w)
    return X_w


def recover_pose(E, pts1, pts2):
    # EのSVD
    U, _, Vt = svd(E)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    t1 = U[:, 2]
    t2 = -U[:, 2]

    pose_candidates = [[R1, t1], [R1, t2], [R2, t1], [R2, t2]]

    # 最適姿勢の選択
    front_count = []
    for R, t in pose_candidates:
        count = 0
        for pt1, pt2 in zip(pts1, pts2):
            P1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            P2 = np.column_stack([R, t])
            x_w = triangulate(P1, P2, pt1, pt2)

            if x_w[-1] > 0:  # 前方判定
                count += 1
        front_count.append(count)

    # 最も前方点が多い姿勢を選択
    idx = np.argmax(front_count)
    return pose_candidates[idx]
