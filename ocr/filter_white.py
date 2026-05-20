import math

import cv2
import numpy as np
import json
import os

def find_farthest_point(poly, centers):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    left_x = min(poly[:, 0])
    right_x = max(poly[:, 0])

    if isinstance(centers, dict):
        filtered = {k: v for k, v in centers.items() if k not in ['blue', 'white']}
        all_points = []
        for color, pts in filtered.items():
            for p in pts:
                dist_left = abs(p[0] - left_x)
                dist_right = abs(right_x - p[0])
                all_points.append((color, p[0], p[1], dist_left, dist_right))
    else:
        all_points = []
        for p in centers:
            dist_left = abs(p[0] - left_x)
            dist_right = abs(right_x - p[0])
            all_points.append(('point', p[0], p[1], dist_left, dist_right))

    if not all_points:
        return None, ('none', left_x)

    farthest = max(all_points, key=lambda x: max(x[3], x[4]))
    color, x, y, dist_left, dist_right = farthest

    if dist_left >= dist_right:
        side_x = left_x
        side = 'left'
    else:
        side_x = right_x
        side = 'right'

    return (color, x, y), (side, side_x)


def find_nearest_point(poly, centers):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    left_x = min(poly[:, 0])
    right_x = max(poly[:, 0])

    if isinstance(centers, dict):
        filtered = {k: v for k, v in centers.items() if k not in ['blue', 'white']}
        all_points = []
        for color, pts in filtered.items():
            for p in pts:
                dist_left = abs(p[0] - left_x)
                dist_right = abs(right_x - p[0])
                all_points.append((color, p[0], p[1], dist_left, dist_right))
    else:
        all_points = []
        for p in centers:
            dist_left = abs(p[0] - left_x)
            dist_right = abs(right_x - p[0])
            all_points.append(('point', p[0], p[1], dist_left, dist_right))

    if not all_points:
        return None, ('none', left_x)

    nearest = min(all_points, key=lambda x: min(x[3], x[4]))
    color, x, y, dist_left, dist_right = nearest

    if dist_left <= dist_right:
        side_x = left_x
        side = 'left'
    else:
        side_x = right_x
        side = 'right'

    return (color, x, y), (side, side_x)


def get_rect(rect_x1, rect_y1, rect_x2, rect_y2):
    left = min(rect_x1, rect_x2)
    right = max(rect_x1, rect_x2)
    top = min(rect_y1, rect_y2)
    bottom = max(rect_y1, rect_y2)
    return left, top, right, bottom

def expand_polygon_vertical(poly, offset=20):

    pts = np.array(poly, dtype=np.float32)

    # 上边方向
    p1 = pts[0]
    p2 = pts[1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    angle = math.atan2(dy, dx)

    # 法线方向
    nx = -math.sin(angle)
    ny = math.cos(angle)

    expanded = []

    # =========================
    # 上边 -> 法线反方向
    # =========================
    expanded.append((
        int(pts[0][0] - nx * offset),
        int(pts[0][1] - ny * offset)
    ))

    expanded.append((
        int(pts[1][0] - nx * offset),
        int(pts[1][1] - ny * offset)
    ))

    # =========================
    # 下边 -> 法线正方向
    # =========================
    expanded.append((
        int(pts[2][0] + nx * offset),
        int(pts[2][1] + ny * offset)
    ))

    expanded.append((
        int(pts[3][0] + nx * offset),
        int(pts[3][1] + ny * offset)
    ))

    return np.array(expanded, dtype=np.int32)
def point_in_rect(px, py, rect):
    left, top, right, bottom = rect
    return left <= px <= right and top <= py <= bottom


def get_polygon(poly, centers, white_points0=None, is_linear=None):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)

    point_info, side_info = find_nearest_point(poly, centers)

    color, farthest_x, farthest_y = point_info
    side, side_x = side_info

    # =========================
    # 1. 获取整体 bbox
    # =========================
    top_y = min(poly[:, 1])
    bottom_y = max(poly[:, 1])

    rect_x1, rect_y1 = side_x, top_y
    rect_x2, rect_y2 = farthest_x, bottom_y

    rect = get_rect(rect_x1, rect_y1, rect_x2, rect_y2)

    left, top, right, bottom = rect
    rect_poly = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ], dtype=np.int32)
    polygon = expand_polygon_vertical(rect_poly, offset=20)
    polygon_np = np.array(polygon)

    # 最终外扩并对齐后的边界
    xmin = int(np.min(polygon_np[:, 0]))
    ymin = int(np.min(polygon_np[:, 1]))
    xmax = int(np.max(polygon_np[:, 0]))
    ymax = int(np.max(polygon_np[:, 1]))

    rect = (xmin, ymin, xmax, ymax)

    # =========================
    # 【新增】提取平移外扩后的核心两点
    # =========================
    p_tl = (xmin, ymin)  # 左上角点 (Top-Left)
    p_br = (xmax, ymax)  # 右下角点 (Bottom-Right)

    # 在返回值中，追加输出这两个点
    return polygon, rect, point_info, [p_tl, p_br]


def shift_side_and_move_up(poly, angle, centers, debug_img=None, output_path=None):
    """
    沿倾斜角度平移左侧边至目标点，并从目标点开始沿平移后的边向上移动 55 像素。

    参数:
        poly: 文本框四边形顶点 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        angle: 文本框倾斜角度（弧度）
        centers: 外部参考点集

    返回:
        point: 原始定位到的圆弧核心点 (farthest_x, farthest_y)
        p_moved_up: 从 point 出发沿边向上移动 55px 后的新点坐标 (x, y)
    """
    poly = np.array(poly, dtype=np.float64)
    tilt_angle = -angle

    # 1. 调用定位函数获取目标点 point
    point_info, side_info = find_nearest_point(poly, centers)
    color, farthest_x, farthest_y = point_info
    point = (int(farthest_x), int(farthest_y))  # 核心点 point

    # 2. 计算原文本框左侧中点与自适应平移距离
    left_x = poly[0, 0]
    left_y_center = (poly[0, 1] + poly[3, 1]) / 2

    vec_target_x = farthest_x - left_x
    vec_target_y = farthest_y - left_y_center
    dir_x = math.cos(tilt_angle)
    dir_y = math.sin(tilt_angle)

    dynamic_distance = vec_target_x * dir_x + vec_target_y * dir_y

    # 3. 计算左侧边平移后的两个端点坐标
    shift_dx = dynamic_distance * dir_x
    shift_dy = dynamic_distance * dir_y

    p_shifted_top = np.array([poly[0, 0] + shift_dx, poly[0, 1] + shift_dy])
    p_shifted_bottom = np.array([poly[3, 0] + shift_dx, poly[3, 1] + shift_dy])

    # 4. 【核心改进】计算“向上”的单位方向向量
    # 向量从下端点指向上端点，代表“图纸上的向上”
    vec_up = p_shifted_top - p_shifted_bottom
    norm = np.linalg.norm(vec_up)

    if norm == 0:
        # 防止分母为 0 异常崩溃，若高为 0 则退化为朝正上移动
        unit_up_x, unit_up_y = 0.0, -1.0
    else:
        unit_up_x = vec_up[0] / norm
        unit_up_y = vec_up[1] / norm

    # 5. 从输入的 point 开始，沿着这个“向上”的方向滑行 55 像素
    move_distance = 55.0
    p_moved_up = (
        int(farthest_x + move_distance * unit_up_x),
        int(farthest_y + move_distance * unit_up_y)
    )

    # =========================
    # 可视化与 Debug 逻辑
    # =========================
    if debug_img is not None:
        vis_img = debug_img.copy()
        p_top_int = (int(p_shifted_top[0]), int(p_shifted_top[1]))
        p_bot_int = (int(p_shifted_bottom[0]), int(p_shifted_bottom[1]))

        # 蓝线：平移后的整条边
        cv2.line(vis_img, p_top_int, p_bot_int, (255, 0, 0), 2)
        # 红点：原始点 point
        cv2.circle(vis_img, point, 5, (0, 0, 255), -1)
        # 紫点：向上滑行 55px 后的新点
        cv2.circle(vis_img, p_moved_up, 5, (255, 0, 255), -1)
        # 黄色箭头线：展示向上移动的物理轨迹
        cv2.arrowedLine(vis_img, point, p_moved_up, (0, 255, 255), 2, tipLength=0.3)

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_img)

    # 严格按照需求输出这两个点
    return point, p_moved_up

def filter_white_points_in_rect(poly, centers, white_points0=None, is_linear=None):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    point_info, side_info = find_farthest_point(poly, centers)

    if point_info is None:
        filtered_centers = centers.copy()
        return filtered_centers, None, []

    color, farthest_x, farthest_y = point_info
    side, side_x = side_info

    top_y = min(poly[:, 1])
    bottom_y = max(poly[:, 1])

    rect_x1, rect_y1 = side_x, top_y
    rect_x2, rect_y2 = farthest_x, bottom_y
    rect = get_rect(rect_x1, rect_y1, rect_x2, rect_y2)

    if is_linear is False:
        left, top, right, bottom = rect
        rect = (left, top - 15, right, bottom + 15)

    if white_points0:
        white_points = white_points0
    else:
        white_points = centers.get('white', [])

    filtered_white = []
    removed_white = []
    for i, (wx, wy) in enumerate(white_points):
        inside = point_in_rect(wx, wy, rect)
        if inside:
            filtered_white.append((wx, wy))
        else:
            removed_white.append((i + 1, wx, wy))
            print(f"Removed White point {i+1} ({wx}, {wy}): OUTSIDE -> excluded")

    filtered_centers = centers.copy()
    filtered_centers['white'] = filtered_white

    return filtered_centers, rect, removed_white


def draw_rectangle(img, poly, centers):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    point_info, side_info = find_farthest_point(poly, centers)
    color, x, y = point_info
    side, side_x = side_info

    top_y = min(poly[:, 1])
    bottom_y = max(poly[:, 1])

    cv2.rectangle(img, (side_x, top_y), (x, bottom_y), (0, 255, 255), 2)

    cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
    cv2.putText(img, f"{color}({x},{y})", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    print(f"Original side: {side} x={side_x}, Farthest point x={x}")
    print(f"Rectangle: ({side_x}, {top_y}) -> ({x}, {bottom_y})")

    return img
def calculate_textbox_angle(poly):
    poly = np.array(poly, dtype=np.float32)

    if poly.shape[0] != 4:
        return 0.0, None

    # 点顺序：左上、右上、右下、左下
    p0, p1, p2, p3 = poly

    # 上边中点、下边中点
    top_mid = (p0 + p1) / 2
    bottom_mid = (p2 + p3) / 2

    # 从上到下的方向向量
    dx = bottom_mid[0] - top_mid[0]
    dy = bottom_mid[1] - top_mid[1]

    # 角度（相对于竖直方向）
    angle = np.arctan2(dx, dy)

    return angle, (tuple(top_mid.astype(int)), tuple(bottom_mid.astype(int)))

if __name__ == '__main__':
    json_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test19/micro_0052_XN.json"
    img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test19/micro_0052_XN.jpg"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img = cv2.imread(img_path)

    detail = data['re_ocr_details'][0]
    micro_poly = np.array(detail['micro_poly'], dtype=np.int32)
    color_centers = detail['color_centers_separate']
    filtered_centers, rect, removed = filter_white_points_in_rect(micro_poly, color_centers,
                                                                  white_points0=[(275, 40), (259, 40)], is_linear=False)
    _,rect,point_info,points = get_polygon(micro_poly, color_centers)
    textbox_angle, _ = calculate_textbox_angle(micro_poly)
    points = shift_side_and_move_up(micro_poly, textbox_angle, color_centers)
    print(f"\nFiltered white centers: {filtered_centers['white']}")

    # result_img = draw_rectangle(img.copy(), micro_poly, color_centers)

    # cv2.polylines(result_img, [micro_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    if rect:
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (122, 255, 122), 2)
    #
    # point_info, _ = find_nearest_point(micro_poly, color_centers)
    if point_info:
        color, x, y = point_info
        cv2.circle(img, (x, y), 8, (0, 255, 122), -1)
        cv2.putText(img, f"{color}({x},{y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    #
    for i, (wx, wy) in enumerate(points):
        cv2.circle(img, (wx, wy), 6, (255, 122, 255), -1)


    output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test19/micro_0052_XN_annotated.jpg"
    cv2.imwrite(output_path, img)
    print(f"\nSaved to {output_path}")