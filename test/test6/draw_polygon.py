import cv2
import json
import numpy as np
import os


def calculate_shift_params(micro_poly, input_angle=None,extend_length = 55):
    """计算平移参数

    Args:
        micro_poly: micro_poly 坐标列表
        input_angle: 输入的倾斜角度（弧度），如果为None则从micro_poly计算

    Returns:
        dict: 包含计算结果的字典
    """
    micro_poly_arr = np.array(micro_poly)
    sorted_by_x = micro_poly_arr[micro_poly_arr[:, 0].argsort()]
    left_points = sorted_by_x[:2]
    right_points = sorted_by_x[-2:]

    left_center = np.mean(left_points, axis=0)
    right_center = np.mean(right_points, axis=0)
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]

    if input_angle is not None:
        angle = input_angle
        dx = 50 * np.cos(angle)
        dy = 50 * np.sin(angle)
    else:
        angle = np.arctan2(dy, dx)


    length = np.sqrt(dx ** 2 + dy ** 2)
    if length > 0:
        ux = dx / length
        uy = dy / length
    else:
        ux, uy = 1, 0

    p1_shifted = (int(left_points[0][0] + ux * extend_length), int(left_points[0][1] + uy * extend_length))
    p2_shifted = (int(left_points[1][0] + ux * extend_length), int(left_points[1][1] + uy * extend_length))

    left_poly_format = [
        [int(left_points[0][0]), int(left_points[0][1])],
        [int(left_points[1][0]), int(left_points[1][1])],
        [int(p2_shifted[0]), int(p2_shifted[1])],
        [int(p1_shifted[0]), int(p1_shifted[1])]
    ]

    return {
        'left_points': left_points,
        'right_points': right_points,
        'left_center': left_center,
        'right_center': right_center,
        'angle': angle,
        'dx': dx,
        'dy': dy,
        'ux': ux,
        'uy': uy,
        'extend_length': extend_length,
        'p1_shifted': p1_shifted,
        'p2_shifted': p2_shifted,
    },left_poly_format


def visualize_polygon_shift(img, micro_poly, global_poly, split_poly, params):
    """可视化多边形和点平移

    Args:
        img: 原始图像
        micro_poly: micro_poly 坐标列表
        global_poly: global_poly 坐标列表
        split_poly: split_poly 坐标列表
        params: calculate_shift_params 返回的参数字典
    """
    # 绘制 global_poly (红色)
    if global_poly:
        global_poly_np = np.array(global_poly, dtype=np.int32)
        cv2.polylines(img, [global_poly_np], isClosed=True, color=(0, 0, 255), thickness=3)

    # 绘制 micro_poly (绿色)
    if micro_poly:
        micro_poly_np = np.array(micro_poly, dtype=np.int32)
        cv2.polylines(img, [micro_poly_np], isClosed=True, color=(0, 255, 0), thickness=3)

    # 绘制 split_poly (蓝色)
    if split_poly:
        split_poly_np = np.array(split_poly, dtype=np.int32)
        cv2.polylines(img, [split_poly_np], isClosed=True, color=(255, 0, 0), thickness=3)

    # 获取参数
    left_points = params['left_points']
    right_points = params['right_points']
    left_center = params['left_center']
    ux = params['ux']
    uy = params['uy']
    extend_length = params['extend_length']
    p1_shifted = params['p1_shifted']
    p2_shifted = params['p2_shifted']

    # 绘制原始左侧点（黄色）
    cv2.circle(img, (int(left_points[0][0]), int(left_points[0][1])), 10, (0, 255, 255), -1)
    cv2.circle(img, (int(left_points[1][0]), int(left_points[1][1])), 10, (0, 255, 255), -1)

    # 绘制平移后的点（紫色）
    cv2.circle(img, p1_shifted, 10, (255, 0, 255), -1)
    cv2.circle(img, p2_shifted, 10, (255, 0, 255), -1)

    # 绘制连接线（黄色为原始左侧连线，紫色为平移后连线）
    cv2.line(img, (int(left_points[0][0]), int(left_points[0][1])),
              (int(left_points[1][0]), int(left_points[1][1])), (0, 255, 255), 2)
    cv2.line(img, p1_shifted, p2_shifted, (255, 0, 255), 2)

    # 绘制箭头表示平移方向（从左到右）
    center_orig = (int(left_center[0]), int(left_center[1]))
    center_shifted = (center_orig[0] + int(ux * extend_length), center_orig[1] + int(uy * extend_length))
    cv2.arrowedLine(img, center_orig, center_shifted, (0, 255, 0), 3, tipLength=0.3)

    # 绘制右侧点（白色）
    cv2.circle(img, (int(right_points[0][0]), int(right_points[0][1])), 10, (255, 255, 255), -1)
    cv2.circle(img, (int(right_points[1][0]), int(right_points[1][1])), 10, (255, 255, 255), -1)

    # 绘制左侧平移后的四边形（橙色）
    if 'left_poly_format' in params:
        left_poly = params['left_poly_format']
        left_poly_np = np.array(left_poly, dtype=np.int32)
        cv2.polylines(img, [left_poly_np], isClosed=True, color=(0, 165, 255), thickness=2)



def draw_polygon_with_shift(image_path, json_path, input_angle=None):
    """绘制多边形并平移

    Args:
        image_path: 图像路径
        json_path: JSON路径
        input_angle: 输入的倾斜角度（弧度），如果为None则从micro_poly计算
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        exit()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    global_poly = data.get('global_poly', [])
    micro_poly = data.get('micro_poly', [])
    split_poly = data.get('split_poly', [])

    print(f"global_poly: {global_poly}")
    print(f"micro_poly: {micro_poly}")
    print(f"split_poly: {split_poly}")

    if not micro_poly:
        print("没有找到 micro_poly")
        return img

    params,_ = calculate_shift_params(micro_poly, input_angle,130)

    print(f"左侧两个点: {params['left_points']}")
    print(f"右侧两个点: {params['right_points']}")
    print(f"计算得到倾斜角度: {np.degrees(params['angle']):.2f} 度")
    print(f"左侧点平移后的位置: {params['p1_shifted']}, {params['p2_shifted']}")

    visualize_polygon_shift(img, micro_poly, global_poly, split_poly, params)

    output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, "poly_visualized.jpg")
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")

    return img


if __name__ == "__main__":
    image_path = r"./test/test6/micro_0005_S15.jpg"
    json_path = r"./test/test6/micro_0005_S15.json"

    draw_polygon_with_shift(image_path, json_path)
