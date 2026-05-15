import numpy as np


def find_nearest_point_to_poly(poly, color_points):
    """
    计算文本框中心点最近的白色点，如果白色点集为空则使用其他所有颜色点。
    如果最近点是白灯且有多个白灯，选择与其他白灯距离最近的作为最终白灯。

    Args:
        poly: 文本框四边形坐标，格式为 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        color_points: 颜色点字典，格式为 {'blue': [...], 'green': [...], 'white': [...], ...}

    Returns:
        nearest_point: 最近的点坐标 (x, y)
        min_dist: 最近距离
        poly_center: 文本框中心点
        nearest_white_points: 被计算过的最近点相关白灯列表
    """
    poly = np.array(poly, dtype=np.int32)
    poly_center = (int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1])))

    if color_points.get('white'):
        target_points = color_points['white']
    else:
        target_points = []
        for color_name, points in color_points.items():
            if color_name != 'white':
                target_points.extend(points)

    if not target_points:
        return None, float('inf'), poly_center, []

    nearest_point, min_dist = find_nearest_point_from_set(poly_center, target_points)
    nearest_white_points = [nearest_point] if len(color_points['white']) >= 1 else []

    if nearest_point and color_points.get('white') and len(color_points['white']) > 1:
        white_points = color_points['white']
        nearest_array = np.array(nearest_point)
        for wp in white_points:
            wp_array = np.array(wp)
            dist_between = np.sqrt(np.sum((wp_array - nearest_array)**2))
            if 0 < dist_between < 80:
                nearest_point = wp
                dist_to_center = np.sqrt((wp[0] - poly_center[0]) ** 2 + (wp[1] - poly_center[1]) ** 2)
                min_dist = dist_to_center
                nearest_white_points.append((int(wp[0]), int(wp[1])))
        # if len(nearest_white_points) > 1:
        #     current_min_dist = min_dist
        #     for wp in nearest_white_points:
        #         dist_to_center = np.sqrt((wp[0] - poly_center[0])**2 + (wp[1] - poly_center[1])**2)
        #         if dist_to_center < current_min_dist:
        #             nearest_point = wp
        #             min_dist = dist_to_center

    return nearest_point, min_dist, poly_center, nearest_white_points


def find_nearest_point_from_set(target_point, point_set):
    """
    计算目标点到点集中最近的一点。

    Args:
        target_point: 目标点 (x, y)
        point_set: 点集，格式为 [(x1, y1), (x2, y2), ...] 或 [[x1, y1], [x2, y2], ...]

    Returns:
        nearest_point: 最近的点坐标 (x, y)
        min_dist: 最近距离
    """
    if not point_set:
        return None, float('inf')

    target_point = np.array(target_point)
    point_set = np.array(point_set)

    distances = np.sqrt(np.sum((point_set - target_point)**2, axis=1))
    min_idx = np.argmin(distances)

    return (int(point_set[min_idx][0]), int(point_set[min_idx][1])), float(distances[min_idx])


def calculate_distances_to_all(nearest_point, point_set, threshold=340):
    """
    计算最近点与其他所有点的距离，并剔除距离大于阈值的点。

    Args:
        nearest_point: 最近点 (x, y)
        point_set: 点集，格式为 [(x1, y1), (x2, y2), ...] 或 [[x1, y1], [x2, y2], ...]
        threshold: 距离阈值，默认340

    Returns:
        distances_dict: 字典，键为点坐标，值为距离
        filtered_points: 距离在阈值内的点列表
    """
    if not point_set:
        return {}, []

    nearest_point = np.array(nearest_point)
    point_set = np.array(point_set)

    distances = np.sqrt(np.sum((point_set - nearest_point)**2, axis=1))

    distances_dict = {}
    filtered_points = []
    for i, point in enumerate(point_set):
        point_tuple = (int(point[0]), int(point[1]))
        dist = float(distances[i])
        distances_dict[point_tuple] = dist
        if dist <= threshold:
            filtered_points.append((point_tuple, dist))

    return distances_dict, filtered_points


def filter_color_points_by_distance(color_points, threshold=340, reference_point=None, remove_nearest_if_large=True):
    """
    剔除color_points中距离大于阈值的点，保持原始格式。

    Args:
        color_points: 颜色点字典，格式为 {'blue': [...], 'green': [...], 'white': [...], ...}
        threshold: 距离阈值，默认340
        reference_point: 参考点 (x, y)，如果为None则使用所有点的几何中心
        remove_nearest_if_large: 是否剔除最近点本身（如果其到参考点距离>100）

    Returns:
        filtered_color_points: 过滤后的颜色点字典，格式与输入一致
    """
    all_points = []
    for color_name, points in color_points.items():
        for point in points:
            all_points.append(point)

    if not all_points:
        return color_points

    all_points_array = np.array(all_points)

    if reference_point is None:
        reference_point = np.mean(all_points_array, axis=0)

    distances = np.sqrt(np.sum((all_points_array - reference_point)**2, axis=1))
    nearest_idx = np.argmin(distances)
    nearest_point = tuple(all_points_array[nearest_idx])
    nearest_dist = distances[nearest_idx]

    filtered_color_points = {}
    for color_name, points in color_points.items():
        filtered_color_points[color_name] = []
        for point in points:
            point_tuple = (point[0], point[1]) if isinstance(point, (list, tuple)) else tuple(point)
            if point_tuple == nearest_point and nearest_dist > 100:
                continue
            dist = np.sqrt((point[0] - nearest_point[0])**2 + (point[1] - nearest_point[1])**2)
            if dist <= threshold:
                filtered_color_points[color_name].append(point)

    return filtered_color_points


if __name__ == '__main__':
    poly = [[300, 44], [345, 44], [345, 69], [300, 69]]
    color_points = {'blue': [(602, 4), (285, 47)], 'green': [], 'red': [], 'white': [(260, 47)], 'yellow': []}

    nearest_point, min_dist, poly_center = find_nearest_point_to_poly(poly, color_points)
    print(f"文本框中心: {poly_center}")
    print(f"最近点: {nearest_point}, 距离: {min_dist:.1f}")

    all_points = []
    for color_name, points in color_points.items():
        all_points.extend(points)
    distances_dict, filtered_points = calculate_distances_to_all(nearest_point, all_points)
    print(f"\n最近点 {nearest_point} 到其他点的距离:")
    for point, dist in distances_dict.items():
        print(f"  {point}: {dist:.1f}")
    print(f"\n距离<=340的点:")
    for point, dist in filtered_points:
        print(f"  {point}: {dist:.1f}")

    print(f"\n原始color_points:")
    print(f"  {color_points}")

    filtered_color_points = filter_color_points_by_distance(color_points, threshold=340, reference_point=nearest_point)
    print(f"\n过滤后color_points (距离<=340, 以{nearest_point}为参考点):")
    print(f"  {filtered_color_points}")