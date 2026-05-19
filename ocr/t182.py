import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import numpy as np

def compute_single_group_max(points):
    # points 形状: (N, 2)
    # 利用显式广播计算所有点对的 x 和 y 差值
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # 平方和 (N, N)
    sq_dist = np.sum(diff ** 2, axis=-1)
    # 取最大值并开方
    return np.sqrt(np.max(sq_dist))
#
# # 使用列表推导式遍历所有组
# # 虽然外层是 Python 循环，但因为单组计算极快，总体速度依然可观
# results = [compute_single_group_max(g) for g in groups_list]
def fit_circle_lstsq(points):
    """
    最小二乘拟合圆 (Kasa method)
    返回: (cx, cy, r, error)
    """
    x = points[:, 0]
    y = points[:, 1]

    A = np.c_[2*x, 2*y, np.ones(len(points))]
    b = x**2 + y**2

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        r = np.sqrt(cx**2 + cy**2 + c)

        # 拟合误差
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        error = np.mean(np.abs(dist - r))

        return cx, cy, r, error

    except:
        return None, None, None, float("inf")
def is_line_segment(points, ratio_thresh=0.98):
    """
    PCA 判断是否为直线
    """
    pts = points.astype(np.float32)

    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    cov = np.cov(pts_centered.T)

    eigvals, _ = np.linalg.eig(cov)

    eigvals = np.sort(eigvals)[::-1]  # 大 -> 小

    if eigvals[0] + eigvals[1] == 0:
        return True

    ratio = eigvals[0] / (eigvals[0] + eigvals[1])

    return ratio > ratio_thresh

def is_closed_contour(cnt, distance_threshold=5.0):
    """
    检查轮廓是否闭合（首尾点距离小于阈值）
    """
    if len(cnt) < 3:
        return False
    first_point = cnt[0][0]
    last_point = cnt[-1][0]
    distance = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
    dist = compute_single_group_max(cnt)
    return dist < distance_threshold

def detect_arc_by_curvature(
        img,
        max_arc_length=100,
        max_fit_error=1.5,
        min_points=25,
        debug=False,
        radis = 6,
        min_threshold = 160,
        black_pixel_count_threshold=80
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, min_threshold, 255, cv2.THRESH_BINARY_INV
    )

    skeleton = skeletonize(binary > 0)
    skeleton = (skeleton.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    vis = img.copy()
    result = []

    for cnt in contours:

        if len(cnt) < min_points:
            continue

        # ========= 1. 弧长约束 =========
        arc_len = cv2.arcLength(cnt, False)
        if arc_len > max_arc_length:
            continue
        h_img, w_img = img.shape[:2]

        x, y, w, h = cv2.boundingRect(cnt)

        # ========= 边界过滤 =========
        margin = 10  # 可调：越大越严格

        if x <= margin or y <= margin or (x + w) >= (w_img - margin) or (y + h) >= (h_img - margin):
            continue
        # ========= 闭合轮廓判断 =========
        if is_closed_contour(cnt, distance_threshold=5.0):
            continue
        points = cnt[:, 0, :].astype(np.float32)
        # ========= 去除直线 =========
        if is_line_segment(points):
            continue
        # ========= 检查圆心区域黑色像素数量 =========
        cx, cy, r, err = fit_circle_lstsq(points)
        if r is None or r < radis:
            continue
        mask_circle = np.zeros(binary.shape, dtype=np.uint8)
        cv2.circle(mask_circle, (int(cx), int(cy)), int(r), 255, -1)
        black_pixel_count = np.sum((binary > 0) & (mask_circle > 0))
        print(black_pixel_count)
        if black_pixel_count > black_pixel_count_threshold:
            continue
        if err > max_fit_error:
            continue

        # ========= 3. 判定为圆弧 =========
        x, y, w, h = cv2.boundingRect(cnt)

        result.append({
            "bbox": [x, y, w, h],
            "arc_length": float(arc_len),
            "fit_error": float(err),
            "radius": float(r)
        })

        if debug:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 1)

    return result, vis, binary


if __name__ == "__main__":
    import os
    image_path = "micro_img/micro_0040_3_D11OO.jpg"
    img = cv2.imread(image_path)

    boxes, vis, binary2 = detect_arc_by_curvature(img, debug=True, max_arc_length=140,
                                                     max_fit_error=1,
                                                     min_points=15,min_threshold=100,radis=17,
                                                  black_pixel_count_threshold=55)
    print(boxes)

    name, ext = os.path.splitext(os.path.basename(image_path))
    output_path = f"{name}_arc_result.jpg"
    cv2.imwrite(output_path, vis)
    output_path_bin = f"{name}_binary.jpg"
    cv2.imwrite(output_path_bin, binary2)

    print(f"结果已保存到: {output_path}")
    print(f"二值化结果已保存到: {output_path_bin}")
