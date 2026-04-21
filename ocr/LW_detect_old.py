import copy
import time
import cv2
import numpy as np
import json
import os

from paddleocr import PaddleOCR


def calculate_textbox_angle(poly):
    """
    计算文本框的倾斜角度

    参数:
        poly: 文本框的四个顶点坐标列表

    返回:
        float: 倾斜角度（弧度），正值表示顺时针倾斜
        tuple: (上边中点, 下边中点) 用于确定文本方向
    """
    poly = np.array(poly)

    if len(poly) != 4:
        return 0.0, None

    center = np.mean(poly, axis=0)

    distances = [np.linalg.norm(poly[i] - center) for i in range(4)]
    sorted_indices = np.argsort(distances)

    edges = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edge_length = np.linalg.norm(p2 - p1)
        edges.append((i, edge_length, p1, p2))

    edges.sort(key=lambda x: x[1], reverse=True)

    long_edge1 = edges[0]
    long_edge2 = edges[1]

    mid1 = (long_edge1[2] + long_edge1[3]) / 2
    mid2 = (long_edge2[2] + long_edge2[3]) / 2

    if mid1[1] < mid2[1]:
        top_mid = mid1
        bottom_mid = mid2
    else:
        top_mid = mid2
        bottom_mid = mid1

    dx = bottom_mid[0] - top_mid[0]
    dy = bottom_mid[1] - top_mid[1]

    angle = np.arctan2(dx, dy)

    return angle, (tuple(top_mid.astype(int)), tuple(bottom_mid.astype(int)))


def get_rotated_rectangle(center, width, height, angle):
    """
    根据中心点、宽高和角度生成旋转矩形的四个顶点

    参数:
        center: 矩形中心点 (x, y)
        width: 矩形宽度
        height: 矩形高度
        angle: 倾斜角度（弧度）

    返回:
        list: 四个顶点坐标列表
    """
    cx, cy = center
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    hw = width / 2
    hh = height / 2

    corners = [
        (cx - hw * cos_a - hh * sin_a, cy - hh * cos_a + hw * sin_a),
        (cx + hw * cos_a - hh * sin_a, cy - hh * cos_a - hw * sin_a),
        (cx + hw * cos_a + hh * sin_a, cy + hh * cos_a - hw * sin_a),
        (cx - hw * cos_a + hh * sin_a, cy + hh * cos_a + hw * sin_a),
    ]

    return [tuple(map(int, corner)) for corner in corners]


# 初始化OCR
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)  # 更换 PP-OCRv5_mobile 模型


def draw_target_char_left_edge(img, json_path, target_char='S5', color_line_info=None, is_linear=False, debug=True,
                               textbox_angle=0.0):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break
    target_index = 0
    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None

    dt_polys = data.get('micro_poly', [])
    if len(dt_polys[0]) == 2:
        dt_polys = [dt_polys]
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None

    poly = dt_polys[target_index]
    print(f"{target_char}对应的文本框轮廓: {poly}")

    if debug:
        cv2.polylines(img, [np.array(poly)], isClosed=True, color=(147, 20, 255), thickness=2)

    angle = textbox_angle
    print(f"文本框倾斜角度: {np.degrees(angle):.2f} 度")

    use_yellow_left_edge = False
    yellow_left_edge = None

    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]

    edges = []
    for i in range(len(poly)):
        p1 = np.array(poly[i])
        p2 = np.array(poly[(i + 1) % len(poly)])
        edge_length = np.linalg.norm(p2 - p1)
        edges.append((i, edge_length, poly[i], poly[(i + 1) % len(poly)]))

    edges.sort(key=lambda x: x[1], reverse=True)
    long_edge1 = edges[0]
    long_edge2 = edges[1]

    textbox_length = max(long_edge1[1], long_edge2[1])
    print(f"文本框长度: {textbox_length} 像素")

    min_y = min(y_coords)
    max_y = max(y_coords)

    if textbox_length > 300:
        rightmost_x = max(x_coords)
        yellow_box_left = rightmost_x - 40

        if abs(angle) > 0.02:
            poly_arr = np.array(poly)
            center = np.mean(poly_arr, axis=0)

            yellow_contour = get_rotated_rectangle(
                (rightmost_x - 20, center[1]),
                40,
                max_y - min_y,
                angle
            )

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            yellow_left_edge_top = (
            int(yellow_box_left * cos_a + center[1] * sin_a - center[0] * sin_a + yellow_box_left * (1 - cos_a)),
            int(-yellow_box_left * sin_a + center[1] * cos_a + center[0] * sin_a - center[1] * sin_a))
            yellow_left_edge_bottom = (
            int(yellow_box_left * cos_a + center[1] * sin_a - center[0] * sin_a + yellow_box_left * (1 - cos_a)),
            int(-yellow_box_left * sin_a + center[1] * cos_a + center[0] * sin_a - center[1] * sin_a + (max_y - min_y)))

            right_edge_top = poly_arr[poly_arr[:, 0].argmax()].copy()
            right_edge_bottom = poly_arr[poly_arr[:, 0].argmax()].copy()

            rightmost_points = poly_arr[poly_arr[:, 0] >= rightmost_x - 5]
            if len(rightmost_points) >= 2:
                rightmost_points_sorted = rightmost_points[rightmost_points[:, 1].argsort()]
                right_edge_top = rightmost_points_sorted[0]
                right_edge_bottom = rightmost_points_sorted[-1]

            dx = right_edge_bottom[0] - right_edge_top[0]
            dy = right_edge_bottom[1] - right_edge_top[1]
            edge_length = np.sqrt(dx ** 2 + dy ** 2)

            if edge_length > 0:
                unit_dx = dx / edge_length
                unit_dy = dy / edge_length

                yellow_left_edge_top = (int(right_edge_top[0] - 40 * cos_a), int(right_edge_top[1] - 40 * sin_a))
                yellow_left_edge_bottom = (
                int(right_edge_bottom[0] - 40 * cos_a), int(right_edge_bottom[1] - 40 * sin_a))

                yellow_contour = [
                    yellow_left_edge_top,
                    (int(right_edge_top[0]), int(right_edge_top[1])),
                    (int(right_edge_bottom[0]), int(right_edge_bottom[1])),
                    yellow_left_edge_bottom
                ]
            else:
                yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                                  (yellow_box_left, max_y)]
        else:
            yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                              (yellow_box_left, max_y)]

        if debug:
            cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
        print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")

        x1 = int(min(point[0] for point in yellow_contour))
        y1 = int(min(point[1] for point in yellow_contour))
        x2 = int(max(point[0] for point in yellow_contour))
        y2 = int(max(point[1] for point in yellow_contour))

        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:
            cropped_img = img[y1:y2, x1:x2]

            if debug:
                if not os.path.exists('output'):
                    os.makedirs('output')

                filename = os.path.basename(json_path).replace('_res.json', '')
                crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
                cv2.imwrite(crop_output_path, cropped_img)
                print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")

            print("对黄色轮廓区域重新进行OCR识别...")
            crop_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

            crop_result = crop_ocr.predict([crop_output_path if debug else img[y1:y2, x1:x2]])

            ocr_success = False
            print("黄色轮廓区域的OCR识别结果:")
            for res in crop_result:
                if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                    ocr_success = True
                    for line in res["rec_texts"]:
                        print(f"识别文本: {line}")

            if debug:
                for res in crop_result:
                    if hasattr(res, 'save_to_img'):
                        res.save_to_img("output")
                        res.save_to_json("output")

            if ocr_success:
                print("OCR识别成功，将使用黄色轮廓的左侧边作为文本框左侧边")
                if abs(angle) > 0.02 and len(yellow_contour) >= 4:
                    yellow_left_edge = [yellow_contour[0], yellow_contour[3]]
                else:
                    yellow_left_edge = [(yellow_box_left, min_y), (yellow_box_left, max_y)]
                use_yellow_left_edge = True
                print(f"已设置使用黄色轮廓的左侧边: {yellow_left_edge}")

    if abs(angle) > 0.02:
        poly_arr = np.array(poly)

        leftmost_points = poly_arr[poly_arr[:, 0] <= min(x_coords) + 5]
        if len(leftmost_points) >= 2:
            leftmost_points_sorted = leftmost_points[leftmost_points[:, 1].argsort()]
            left_points = [tuple(leftmost_points_sorted[0]), tuple(leftmost_points_sorted[-1])]
        else:
            leftmost_idx = poly_arr[:, 0].argmin()
            left_point = poly_arr[leftmost_idx]

            dx = np.cos(angle)
            dy = np.sin(angle)

            height_estimate = max_y - min_y
            left_points = [
                (int(left_point[0] - dy * height_estimate / 2), int(left_point[1] + dx * height_estimate / 2)),
                (int(left_point[0] + dy * height_estimate / 2), int(left_point[1] - dx * height_estimate / 2))
            ]
    else:
        poly_sorted = sorted(poly, key=lambda point: point[0])
        left_points = poly_sorted[:2]
        left_points.sort(key=lambda point: point[1])

    if use_yellow_left_edge and yellow_left_edge:
        left_points = yellow_left_edge
        print(f"已使用黄色轮廓的左侧边: {left_points}")

    left_edge = tuple(map(tuple, left_points))
    print(f"左侧边坐标: {left_edge}")

    extended_left_start = left_edge[0]
    extended_left_end = left_edge[1]

    polygon = None
    if color_line_info:
        extended_start, extended_end = color_line_info

        color_line_dx = extended_end[0] - extended_start[0]
        color_line_dy = extended_end[1] - extended_start[1]
        color_line_length = ((color_line_dx ** 2) + (color_line_dy ** 2)) ** 0.5

        left_edge_dx = left_edge[1][0] - left_edge[0][0]
        left_edge_dy = left_edge[1][1] - left_edge[0][1]
        left_edge_length = ((left_edge_dx ** 2) + (left_edge_dy ** 2)) ** 0.5

        if left_edge_length > 0 and color_line_length > 0:
            unit_dx = left_edge_dx / left_edge_length
            unit_dy = left_edge_dy / left_edge_length

            scale_factor = color_line_length / left_edge_length
            extended_left_start = (int(left_edge[0][0] - unit_dx * left_edge_length * (scale_factor - 1) / 2),
                                   int(left_edge[0][1] - unit_dy * left_edge_length * (scale_factor - 1) / 2))
            extended_left_end = (int(left_edge[1][0] + unit_dx * left_edge_length * (scale_factor - 1) / 2),
                                 int(left_edge[1][1] + unit_dy * left_edge_length * (scale_factor - 1) / 2))

    if debug:
        cv2.line(img, extended_left_start, extended_left_end, (0, 0, 255), 2)

    if color_line_info:
        extended_start, extended_end = color_line_info
        polygon = [extended_start, extended_end, extended_left_end, extended_left_start]

        if debug:
            cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=1)

    return img, polygon


def calculate_black_pixels(img, polygon, json_path, data):
    """
    统计多边形内的黑色像素数量，分为左右两部分

    参数:
        img: 图像
        polygon: 多边形顶点
        json_path: JSON文件路径
        data: JSON数据字典

    返回:
        int: 黑色像素总数量
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    polygon_arr = np.array(polygon)

    center_x = np.mean(polygon_arr[:, 0])
    center_y = np.mean(polygon_arr[:, 1])

    if len(polygon) >= 2:
        p1 = polygon_arr[0]
        p2 = polygon_arr[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            unit_dx = dx / length
            unit_dy = dy / length
        else:
            unit_dx, unit_dy = 1, 0
    else:
        unit_dx, unit_dy = 1, 0

    perp_dx = -unit_dy
    perp_dy = unit_dx

    left_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    right_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y, x] == 255:
                pixel_vec = (x - center_x, y - center_y)
                dot_product = pixel_vec[0] * perp_dx + pixel_vec[1] * perp_dy

                if dot_product > 0:
                    left_mask[y, x] = 255
                else:
                    right_mask[y, x] = 255

    left_black_pixels = cv2.countNonZero(cv2.bitwise_and(left_mask, black_mask))
    right_black_pixels = cv2.countNonZero(cv2.bitwise_and(right_mask, black_mask))

    black_pixels_in_polygon = left_black_pixels + right_black_pixels

    print(f"多边形内的黑色像素数量: {black_pixels_in_polygon}")
    print(f"左侧黑色像素数量: {left_black_pixels}")
    print(f"右侧黑色像素数量: {right_black_pixels}")

    data['black_pixels_in_polygon'] = int(black_pixels_in_polygon)
    data['left_black_pixels'] = int(left_black_pixels)
    data['right_black_pixels'] = int(right_black_pixels)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"已将黑色像素数量更新到: {json_path}")

    return black_pixels_in_polygon


def draw_target_char_right_edge(img, json_path, target_char='X', color_line_info=None, is_linear=False, debug=True,
                                textbox_angle=0.0):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break
    target_index = 0
    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None

    dt_polys = data.get('micro_poly', [])
    if len(dt_polys[0]) == 2:
        dt_polys = [dt_polys]
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None

    poly = dt_polys[target_index]
    print(f"{target_char}对应的文本框轮廓: {poly}")

    if debug:
        cv2.polylines(img, [np.array(poly)], isClosed=True, color=(147, 20, 255), thickness=2)

    angle = textbox_angle
    print(f"文本框倾斜角度: {np.degrees(angle):.2f} 度")

    use_yellow_right_edge = False
    yellow_right_edge = None

    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]

    edges = []
    for i in range(len(poly)):
        p1 = np.array(poly[i])
        p2 = np.array(poly[(i + 1) % len(poly)])
        edge_length = np.linalg.norm(p2 - p1)
        edges.append((i, edge_length, poly[i], poly[(i + 1) % len(poly)]))

    edges.sort(key=lambda x: x[1], reverse=True)
    long_edge1 = edges[0]
    long_edge2 = edges[1]

    textbox_length = max(long_edge1[1], long_edge2[1])
    print(f"文本框长度: {textbox_length} 像素")

    min_y = min(y_coords)
    max_y = max(y_coords)

    if textbox_length > 300:
        rightmost_x = max(x_coords)
        yellow_box_left = rightmost_x - 40

        if abs(angle) > 0.02:
            poly_arr = np.array(poly)

            rightmost_points = poly_arr[poly_arr[:, 0] >= rightmost_x - 5]
            if len(rightmost_points) >= 2:
                rightmost_points_sorted = rightmost_points[rightmost_points[:, 1].argsort()]
                right_edge_top = rightmost_points_sorted[0]
                right_edge_bottom = rightmost_points_sorted[-1]
            else:
                rightmost_idx = poly_arr[:, 0].argmax()
                right_point = poly_arr[rightmost_idx]
                height_estimate = max_y - min_y
                dx = np.cos(angle)
                dy = np.sin(angle)
                right_edge_top = np.array(
                    [right_point[0] - dy * height_estimate / 2, right_point[1] + dx * height_estimate / 2])
                right_edge_bottom = np.array(
                    [right_point[0] + dy * height_estimate / 2, right_point[1] - dx * height_estimate / 2])

            dx = right_edge_bottom[0] - right_edge_top[0]
            dy = right_edge_bottom[1] - right_edge_top[1]
            edge_length = np.sqrt(dx ** 2 + dy ** 2)

            if edge_length > 0:
                cos_a = dx / edge_length
                sin_a = dy / edge_length

                yellow_left_edge_top = (int(right_edge_top[0] - 40 * cos_a), int(right_edge_top[1] - 40 * sin_a))
                yellow_left_edge_bottom = (
                int(right_edge_bottom[0] - 40 * cos_a), int(right_edge_bottom[1] - 40 * sin_a))

                yellow_contour = [
                    yellow_left_edge_top,
                    (int(right_edge_top[0]), int(right_edge_top[1])),
                    (int(right_edge_bottom[0]), int(right_edge_bottom[1])),
                    yellow_left_edge_bottom
                ]
            else:
                yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                                  (yellow_box_left, max_y)]
        else:
            yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                              (yellow_box_left, max_y)]

        if debug:
            cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
        print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")

        x1 = int(min(point[0] for point in yellow_contour))
        y1 = int(min(point[1] for point in yellow_contour))
        x2 = int(max(point[0] for point in yellow_contour))
        y2 = int(max(point[1] for point in yellow_contour))

        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 > x1 and y2 > y1:
            cropped_img = img[y1:y2, x1:x2]

            if debug:
                if not os.path.exists('output'):
                    os.makedirs('output')

                filename = os.path.basename(json_path).replace('_res.json', '')
                crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
                cv2.imwrite(crop_output_path, cropped_img)
                print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")

            print("对黄色轮廓区域重新进行OCR识别...")
            crop_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

            crop_result = crop_ocr.predict([crop_output_path if debug else img[y1:y2, x1:x2]])

            ocr_success = False
            print("黄色轮廓区域的OCR识别结果:")
            for res in crop_result:
                if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                    ocr_success = True
                    for line in res["rec_texts"]:
                        print(f"识别文本: {line}")

            if debug:
                for res in crop_result:
                    if hasattr(res, 'save_to_img'):
                        res.save_to_img("output")
                        res.save_to_json("output")

            if ocr_success:
                print("OCR识别成功，将使用黄色轮廓的右侧边作为文本框右侧边")
                if abs(angle) > 0.02 and len(yellow_contour) >= 4:
                    yellow_right_edge = [yellow_contour[1], yellow_contour[2]]
                else:
                    yellow_right_edge = [(rightmost_x, min_y), (rightmost_x, max_y)]
                use_yellow_right_edge = True
                print(f"已设置使用黄色轮廓的右侧边: {yellow_right_edge}")

    # if abs(angle) > 0.02:
    #     poly_arr = np.array(poly)

    #     rightmost_points = poly_arr[poly_arr[:, 0] >= max(x_coords) - 5]
    #     if len(rightmost_points) >= 2:
    #         rightmost_points_sorted = rightmost_points[rightmost_points[:, 1].argsort()]
    #         right_points = [tuple(rightmost_points_sorted[0]), tuple(rightmost_points_sorted[-1])]
    #     else:
    #         rightmost_idx = poly_arr[:, 0].argmax()
    #         right_point = poly_arr[rightmost_idx]

    #         dx = np.cos(angle)
    #         dy = np.sin(angle)

    #         height_estimate = max_y - min_y
    #         right_points = [
    #             (int(right_point[0] - dy * height_estimate / 2), int(right_point[1] + dx * height_estimate / 2)),
    #             (int(right_point[0] + dy * height_estimate / 2), int(right_point[1] - dx * height_estimate / 2))
    #         ]
    # else:
    poly_sorted = sorted(poly, key=lambda point: point[0])
    right_points = poly_sorted[-2:]
    right_points.sort(key=lambda point: point[1])

    if use_yellow_right_edge and yellow_right_edge:
        right_points = yellow_right_edge
        print(f"已使用黄色轮廓的右侧边: {right_points}")

    right_edge = tuple(map(tuple, right_points))
    print(f"右侧边坐标: {right_edge}")

    extended_right_start = right_edge[0]
    extended_right_end = right_edge[1]

    polygon = None
    if color_line_info:
        extended_start, extended_end = color_line_info

        color_line_dx = extended_end[0] - extended_start[0]
        color_line_dy = extended_end[1] - extended_start[1]
        color_line_length = ((color_line_dx ** 2) + (color_line_dy ** 2)) ** 0.5

        right_edge_dx = right_edge[1][0] - right_edge[0][0]
        right_edge_dy = right_edge[1][1] - right_edge[0][1]
        right_edge_length = ((right_edge_dx ** 2) + (right_edge_dy ** 2)) ** 0.5

        if right_edge_length > 0 and color_line_length > 0:
            unit_dx = right_edge_dx / right_edge_length
            unit_dy = right_edge_dy / right_edge_length

            scale_factor = color_line_length / right_edge_length
            extended_right_start = (int(right_edge[0][0] - unit_dx * right_edge_length * (scale_factor - 1) / 2),
                                    int(right_edge[0][1] - unit_dy * right_edge_length * (scale_factor - 1) / 2))
            extended_right_end = (int(right_edge[1][0] + unit_dx * right_edge_length * (scale_factor - 1) / 2),
                                  int(right_edge[1][1] + unit_dy * right_edge_length * (scale_factor - 1) / 2))

    # if debug:
    #     cv2.line(img, extended_right_start, extended_right_end, (0, 0, 255), 2)

    if color_line_info:
        extended_start, extended_end = color_line_info
        polygon = [extended_start, extended_end, extended_right_end, extended_right_start]

        if debug:
            cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=1)

    return img, polygon


def detect_colors(image_path, target_char, debug=True, threshold=100):
    """
    检测图像中的颜色像素并分析位置关系

    参数:
        image_path: 图像路径
        target_char: 目标字符，用于确定使用哪种检测方式
        debug: 是否进行可视化，默认为True
        threshold: 黑色像素数量阈值，用于判断是否匹配成功，默认为100

    返回:
        tuple: (is_match: bool, match_score: float)
            - is_match: 是否匹配成功（基于黑色像素统计）
            - match_score: 匹配分数（黑色像素数量的归一化值，范围0-1）
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False, 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([70, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    total_pixels = img.shape[0] * img.shape[1]

    print(f"图像: {image_path}")
    print(f"红色像素: {red_pixels} ({red_pixels / total_pixels * 100:.2f}%)")
    print(f"黄色像素: {yellow_pixels} ({yellow_pixels / total_pixels * 100:.2f}%)")
    print(f"绿色像素: {green_pixels} ({green_pixels / total_pixels * 100:.2f}%)")

    def calculate_center(mask):
        if cv2.countNonZero(mask) == 0:
            return None
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    red_center = calculate_center(mask_red)
    yellow_center = calculate_center(mask_yellow)
    green_center = calculate_center(mask_green)
    color_centers1 = []
    if red_center:
        color_centers1.append(red_center)
        print(f"红色像素中心坐标: {red_center}")
    if yellow_center:
        color_centers1.append(yellow_center)
        print(f"黄色像素中心坐标: {yellow_center}")
    if green_center:
        color_centers1.append(green_center)
        print(f"绿色像素中心坐标: {green_center}")

    color_centers = {}
    if red_center:
        color_centers['红色'] = red_center
    if yellow_center:
        color_centers['黄色'] = yellow_center
    if green_center:
        color_centers['绿色'] = green_center

    analyze_color_relationships(color_centers)
    line_type = analyze_color_relationships1(color_centers1)
    is_linear = (line_type == 'single_line')
    print(f"线条类型: {line_type}")

    json_path = image_path.rsplit('.', 1)[0] + '.json'

    if not os.path.exists(json_path):
        print(f"JSON文件不存在: {json_path}")
        return False, 0.0

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    textbox_angle = 0.0
    dt_polys = data.get('micro_poly', [])
    if len(dt_polys) > 0:
        if len(dt_polys[0]) == 2:
            dt_polys = [dt_polys]
        poly = dt_polys[0]
        textbox_angle, _ = calculate_textbox_angle(poly)
        print(f"文本框倾斜角度: {np.degrees(textbox_angle):.2f} 度")

    if 'X' in target_char:
        vis_img, color_line_info, linear_point = process_leftmost_pixels(img, mask_red, mask_yellow, mask_green,
                                                                         red_center,
                                                                         yellow_center, green_center,
                                                                         angle=textbox_angle, debug=debug,
                                                                         is_linear=is_linear)
    else:
        vis_img, color_line_info, linear_point = process_rightmost_pixels(img, mask_red, mask_yellow, mask_green,
                                                                          red_center,
                                                                          yellow_center, green_center,
                                                                          angle=textbox_angle, debug=debug,
                                                                          is_linear=is_linear)

    # red_rightmost = calculate_rightmost(mask_red)
    # yellow_rightmost = calculate_rightmost(mask_yellow)
    # green_rightmost = calculate_rightmost(mask_green)
    #
    # rightmost_color_pixel = None
    # rightmost_pixels = []
    # if red_rightmost:
    #     rightmost_pixels.append(red_rightmost)
    # if yellow_rightmost:
    #     rightmost_pixels.append(yellow_rightmost)
    # if green_rightmost:
    #     rightmost_pixels.append(green_rightmost)
    #
    # if rightmost_pixels:
    #     rightmost_pixels.sort(key=lambda p: p[0], reverse=True)
    #     rightmost_color_pixel = rightmost_pixels[0]
    #     print(f"计算得到最右侧彩色像素: {rightmost_color_pixel}")

    polygon = None
    if 'X' in target_char:
        vis_img, polygon = draw_target_char_right_edge(vis_img, json_path, color_line_info=color_line_info,
                                                       target_char=target_char, is_linear=is_linear, debug=debug,
                                                       textbox_angle=textbox_angle)
    else:
        vis_img, polygon = draw_target_char_left_edge(vis_img, json_path, color_line_info=color_line_info,
                                                      target_char=target_char, is_linear=is_linear, debug=debug,
                                                      textbox_angle=textbox_angle)

    black_pixel_count = 0
    if is_linear:
        print("执行单线检测")
        vis_img, found_pixel, left_black, right_black = single_line_detection(vis_img, json_path, target_char,
                                                                              linear_point[0], img, debug=debug)
        black_pixel_count = left_black + right_black
    else:
        print("执行双线黑色像素统计")
        if polygon:
            black_pixel_count = calculate_black_pixels(vis_img, polygon, json_path, data)

    save_visualization(vis_img, image_path, debug=debug)

    match_score = min(black_pixel_count / 1000.0, 1.0)
    is_match = black_pixel_count >= threshold

    print(f"黑色像素总数: {black_pixel_count}, 匹配分数: {match_score:.3f}, 是否匹配: {is_match}")
    return is_match, match_score


def analyze_color_relationships(color_centers):
    """
    分析颜色之间的位置关系

    参数:
        color_centers: 颜色中心点字典
    """
    print("\n颜色位置关系分析:")

    # 分析每对颜色的位置关系
    colors = list(color_centers.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            color1 = colors[i]
            color2 = colors[j]
            center1 = color_centers[color1]
            center2 = color_centers[color2]

            # 比较x坐标（左右关系），允许5像素偏差
            x_diff = abs(center1[0] - center2[0])
            if x_diff <= 5:
                left_right = f"{color1}和{color2}在同一垂直位置（偏差{int(x_diff)}像素）"
            elif center1[0] < center2[0]:
                left_right = f"{color1}在{color2}的左侧（偏差{int(x_diff)}像素）"
            else:
                left_right = f"{color1}在{color2}的右侧（偏差{int(x_diff)}像素）"

            # 比较y坐标（上下关系），允许5像素偏差
            y_diff = abs(center1[1] - center2[1])
            if y_diff <= 5:
                up_down = f"{color1}和{color2}在同一水平位置（偏差{int(y_diff)}像素）"
            elif center1[1] < center2[1]:
                up_down = f"{color1}在{color2}的上方（偏差{int(y_diff)}像素）"
            else:
                up_down = f"{color1}在{color2}的下方（偏差{int(y_diff)}像素）"

            # 打印位置关系
            print(f"{left_right}，{up_down}")

    # 分析颜色的排列顺序
    if len(color_centers) >= 2:
        # 按x坐标排序（从左到右）
        sorted_by_x = sorted(color_centers.items(), key=lambda item: item[1][0])
        left_to_right = [color for color, _ in sorted_by_x]
        print(f"从左到右的颜色顺序: {left_to_right}")

        # 按y坐标排序（从上到下）
        sorted_by_y = sorted(color_centers.items(), key=lambda item: item[1][1])
        top_to_bottom = [color for color, _ in sorted_by_y]
        print(f"从上到下的颜色顺序: {top_to_bottom}")


def analyze_color_relationships1(color_centers):
    """
    分析颜色中心点，判断是单线还是双线
    - 如果中心点数量 <= 2，判定为单线
    - 如果中心点数量 > 2，根据x值排序，用首尾点构建矩形，外扩5像素，
      如果其余点中有任意一个在矩形外，则为双线，否则为单线

    参数:
        color_centers: 颜色中心点列表，每个元素为 (x, y) 元组

    返回:
        str: 'single_line' 或 'double_line'
    """
    if len(color_centers) <= 2:
        print(f"中心点数量为 {len(color_centers)}，判定为单线")
        return 'single_line'

    sorted_by_x = sorted(color_centers, key=lambda p: p[0])
    first_point = sorted_by_x[0]
    last_point = sorted_by_x[-1]

    x1, y1 = first_point
    x2, y2 = last_point

    half_height = 5

    vertical = (x2 - x1 == 0)
    if vertical:
        slope = 0
    else:
        slope = (y2 - y1) / (x2 - x1)

    rect_y_min = min(y1, y2) - half_height
    rect_y_max = max(y1, y2) + half_height

    for point in sorted_by_x[1:-1]:
        px, py = point
        if vertical:
            rect_x_min = min(x1, x2) - half_height
            rect_x_max = max(x1, x2) + half_height
            in_rect = (rect_x_min <= px <= rect_x_max and rect_y_min <= py <= rect_y_max)
        else:
            A = slope
            B = -1
            C = y1 - slope * x1
            dist = abs(A * px + B * py + C) / np.sqrt(A * A + B * B)
            in_rect = (dist <= half_height and rect_y_min <= py <= rect_y_max)

        if not in_rect:
            print(f"发现中心点 {point} 在矩形范围外，判定为双线")
            return 'double_line'

    print(f"所有中心点都在矩形范围内，判定为单线")
    return 'single_line'


def single_line_detection(img, json_path, target_char, linear_point, origin_img, debug=True):
    """
    单线检测函数，根据文本内容从文本框的左侧或右侧边开始找黑色像素

    参数:
        img: 原始图像
        json_path: JSON文件路径
        target_char: 目标字符
        linear_point: 颜色最左/最右点坐标
        origin_img: 原始图像
        debug: 是否进行可视化绘制，默认为True

    返回:
        tuple: (可视化图像, 找到的黑色像素位置, 左侧黑色像素数量, 右侧黑色像素数量)
    """
    left_black_pixels = 0
    right_black_pixels = 0

    if debug:
        if not os.path.exists('output'):
            os.makedirs('output')
        filename = os.path.basename(json_path).replace('_res.json', '')
        input_output_path = os.path.join('output', f'{filename}_input.png')
        cv2.imwrite(input_output_path, origin_img)
        print(f"已保存输入的原始图像到: {input_output_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_index = None
    for i, text in enumerate(data.get('matched_keys', [])):
        if text == target_char:
            target_index = i
            break
    target_index = 0
    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None, left_black_pixels, right_black_pixels

    dt_polys = data.get('micro_poly', [])
    if len(dt_polys[0]) == 2:
        dt_polys = [dt_polys]
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None, left_black_pixels, right_black_pixels

    poly = dt_polys[target_index]

    angle, mid_points = calculate_textbox_angle(poly)
    print(f"文本框倾斜角度: {np.degrees(angle):.2f} 度")

    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    gray0 = copy.deepcopy(gray)
    _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    if debug:
        if not os.path.exists('output'):
            os.makedirs('output')
        filename = os.path.basename(json_path).replace('_res.json', '')
        mask_output_path = os.path.join('output', f'{filename}_black_mask.png')
        cv2.imwrite(mask_output_path, black_mask)
        print(f"已保存黑色像素掩码到: {mask_output_path}")

    found_pixel = None
    if target_char == 'XⅠ':
        print(1)

    def scan_along_angle(start_x, end_x, min_y, max_y, angle, step=1):
        scan_result = None
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        if step > 0:
            x_range = range(start_x, end_x + 1, step)
        else:
            x_range = range(start_x, end_x - 1, step)

        for x in x_range:
            if abs(angle) > 0.02:
                center_y = (min_y + max_y) / 2
                y_offset = (x - (min_x + max_x) / 2) * sin_a / cos_a if abs(cos_a) > 0.01 else 0
                y_start = max(min_y, int(min_y + y_offset - (max_y - min_y) * 0.1))
                y_end = min(max_y, int(max_y + y_offset + (max_y - min_y) * 0.1))

                for y in range(y_start, y_end + 1):
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        if black_mask[y, x] == 255:
                            return (x, y)
            else:
                for y in range(min_y, max_y + 1):
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        if black_mask[y, x] == 255:
                            return (x, y)
        return scan_result

    if target_char.startswith('S'):
        print("从文本框右侧边开始找黑色像素（考虑倾斜角度）")
        if abs(angle) > 0.02:
            poly_arr = np.array(poly)
            rightmost_points = poly_arr[poly_arr[:, 0] >= max_x - 5]
            if len(rightmost_points) >= 2:
                rightmost_points_sorted = rightmost_points[rightmost_points[:, 1].argsort()]
                right_edge_top = rightmost_points_sorted[0]
                right_edge_bottom = rightmost_points_sorted[-1]
            else:
                rightmost_idx = poly_arr[:, 0].argmax()
                right_point = poly_arr[rightmost_idx]
                height_estimate = max_y - min_y
                dx = np.cos(angle)
                dy = np.sin(angle)
                right_edge_top = np.array(
                    [right_point[0] - dy * height_estimate / 2, right_point[1] + dx * height_estimate / 2])
                right_edge_bottom = np.array(
                    [right_point[0] + dy * height_estimate / 2, right_point[1] - dx * height_estimate / 2])

            edge_dx = right_edge_bottom[0] - right_edge_top[0]
            edge_dy = right_edge_bottom[1] - right_edge_top[1]
            edge_length = np.sqrt(edge_dx ** 2 + edge_dy ** 2)

            if edge_length > 0:
                unit_edge_dx = edge_dx / edge_length
                unit_edge_dy = edge_dy / edge_length

                perp_dx = -unit_edge_dy
                perp_dy = unit_edge_dx

                for dist in range(0, int(max_x - min_x), 1):
                    for t in np.linspace(0, 1, int(edge_length)):
                        base_x = right_edge_top[0] + t * edge_dx
                        base_y = right_edge_top[1] + t * edge_dy

                        scan_x = int(base_x - dist * perp_dx)
                        scan_y = int(base_y - dist * perp_dy)

                        if 0 <= scan_x < img.shape[1] and 0 <= scan_y < img.shape[0]:
                            if black_mask[scan_y, scan_x] == 255:
                                found_pixel = (scan_x, scan_y)
                                print(f"找到第一个黑色像素: {found_pixel}")
                                break
                    if found_pixel:
                        break
        else:
            for x in range(max_x, min_x - 1, -1):
                for y in range(min_y, max_y + 1):
                    if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                        if black_mask[y, x] == 255:
                            found_pixel = (x, y)
                            print(f"找到第一个黑色像素: {found_pixel}")
                            break
                if found_pixel:
                    break
    elif target_char.startswith('X'):
        print("从文本框左侧边开始找黑色像素（考虑倾斜角度）")
        if abs(angle) > 0.02:
            poly_arr = np.array(poly)
            leftmost_points = poly_arr[poly_arr[:, 0] <= min_x + 5]
            if len(leftmost_points) >= 2:
                leftmost_points_sorted = leftmost_points[leftmost_points[:, 1].argsort()]
                left_edge_top = leftmost_points_sorted[0]
                left_edge_bottom = leftmost_points_sorted[-1]
            else:
                leftmost_idx = poly_arr[:, 0].argmin()
                left_point = poly_arr[leftmost_idx]
                height_estimate = max_y - min_y
                dx = np.cos(angle)
                dy = np.sin(angle)
                left_edge_top = np.array(
                    [left_point[0] - dy * height_estimate / 2, left_point[1] + dx * height_estimate / 2])
                left_edge_bottom = np.array(
                    [left_point[0] + dy * height_estimate / 2, left_point[1] - dx * height_estimate / 2])

            edge_dx = left_edge_bottom[0] - left_edge_top[0]
            edge_dy = left_edge_bottom[1] - left_edge_top[1]
            edge_length = np.sqrt(edge_dx ** 2 + edge_dy ** 2)

            if edge_length > 0:
                unit_edge_dx = edge_dx / edge_length
                unit_edge_dy = edge_dy / edge_length

                perp_dx = unit_edge_dy
                perp_dy = -unit_edge_dx

                for dist in range(0, int(max_x - min_x), 1):
                    for t in np.linspace(0, 1, int(edge_length)):
                        base_x = left_edge_top[0] + t * edge_dx
                        base_y = left_edge_top[1] + t * edge_dy

                        scan_x = int(base_x + dist * perp_dx)
                        scan_y = int(base_y + dist * perp_dy)

                        if 0 <= scan_x < img.shape[1] and 0 <= scan_y < img.shape[0]:
                            if black_mask[scan_y, scan_x] == 255:
                                found_pixel = (scan_x, scan_y)
                                print(f"找到第一个黑色像素: {found_pixel}")
                                break
                    if found_pixel:
                        break
        else:
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                        if black_mask[y, x] == 255:
                            found_pixel = (x, y)
                            print(f"找到第一个黑色像素: {found_pixel}")
                            break
                if found_pixel:
                    break

    if found_pixel:
        if debug:
            cv2.circle(img, found_pixel, 3, (0, 0, 255), -1)
            print("已可视化找到的黑色像素")

        color_pixel = linear_point
        if color_pixel:
            print(f"颜色点: {color_pixel}")
            if debug:
                cv2.line(img, found_pixel, color_pixel, (255, 0, 0), 2)
                print("已绘制黑色像素与彩色像素的连线")

            dx = color_pixel[0] - found_pixel[0]
            dy = color_pixel[1] - found_pixel[1]

            length = ((dx ** 2) + (dy ** 2)) ** 0.5
            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length

                perp_dx = -unit_dy
                perp_dy = unit_dx

                p1 = (int(found_pixel[0] + perp_dx * 8), int(found_pixel[1] + perp_dy * 8))
                p2 = (int(found_pixel[0] - perp_dx * 8), int(found_pixel[1] - perp_dy * 8))
                p3 = (int(color_pixel[0] - perp_dx * 8), int(color_pixel[1] - perp_dy * 8))
                p4 = (int(color_pixel[0] + perp_dx * 8), int(color_pixel[1] + perp_dy * 8))

                rectangle = [p1, p2, p3, p4]

                def get_line_points(p1, p2):
                    x1, y1 = p1
                    x2, y2 = p2
                    points = []
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    sx = 1 if x1 < x2 else -1
                    sy = 1 if y1 < y2 else -1
                    err = dx - dy
                    x, y = x1, y1
                    while True:
                        points.append((x, y))
                        if x == x2 and y == y2:
                            break
                        e2 = 2 * err
                        if e2 > -dy:
                            err -= dy
                            x += sx
                        if e2 < dx:
                            err += dx
                            y += sy
                    return points

                line_points = get_line_points(found_pixel, color_pixel)

                first_white_pixel = None
                for point in line_points[1:]:
                    x, y = point
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        if gray[y, x] > 200:
                            first_white_pixel = point
                            print(f"找到第一个白色像素: {first_white_pixel}")
                            break

                if first_white_pixel:
                    if debug:
                        cv2.circle(img, first_white_pixel, 2, (0, 255, 0), -1)
                        print("已标记第一个白色像素")

                    dx = found_pixel[0] - color_pixel[0]
                    dy = found_pixel[1] - color_pixel[1]

                    length = ((dx ** 2) + (dy ** 2)) ** 0.5
                    if length > 0:
                        unit_dx = dx / length
                        unit_dy = dy / length

                        rect_end = found_pixel

                        perp_dx = -unit_dy
                        perp_dy = unit_dx

                        expand_px = 25
                        rect_p1 = (int(color_pixel[0] + perp_dx * expand_px), int(color_pixel[1] + perp_dy * expand_px))
                        rect_p2 = (int(color_pixel[0] - perp_dx * expand_px), int(color_pixel[1] - perp_dy * expand_px))
                        rect_p3 = (int(rect_end[0] - perp_dx * expand_px), int(rect_end[1] - perp_dy * expand_px))
                        rect_p4 = (int(rect_end[0] + perp_dx * expand_px), int(rect_end[1] + perp_dy * expand_px))

                        white_rectangle = [rect_p1, rect_p2, rect_p3, rect_p4]
                        if debug:
                            cv2.polylines(img, [np.array(white_rectangle)], isClosed=True, color=(255, 255, 0),
                                          thickness=1)
                            print("已绘制从color_pixel到黑色像素，上下外扩25个像素的矩形")

                        expand_px2 = 35
                        rect2_p1 = (
                            int(color_pixel[0] + perp_dx * expand_px2), int(color_pixel[1] + perp_dy * expand_px2))
                        rect2_p2 = (
                            int(color_pixel[0] - perp_dx * expand_px2), int(color_pixel[1] - perp_dy * expand_px2))
                        rect2_p3 = (int(rect_end[0] - perp_dx * expand_px2), int(rect_end[1] - perp_dy * expand_px2))
                        rect2_p4 = (int(rect_end[0] + perp_dx * expand_px2), int(rect_end[1] + perp_dy * expand_px2))

                        rectangle2 = [rect2_p1, rect2_p2, rect2_p3, rect2_p4]
                        if debug:
                            cv2.polylines(img, [np.array(rectangle2)], isClosed=True, color=(0, 0, 255),
                                          thickness=1)
                            print("已绘制从color_pixel到黑色像素，上下外扩35个像素的矩形2")

                        mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask1, [np.array(white_rectangle)], 255)

                        mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask2, [np.array(rectangle2)], 255)

                        disjoint_mask = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))

                        def get_line_points(p1, p2):
                            x1, y1 = p1
                            x2, y2 = p2
                            points = []
                            dx = abs(x2 - x1)
                            dy = abs(y2 - y1)
                            sx = 1 if x1 < x2 else -1
                            sy = 1 if y1 < y2 else -1
                            err = dx - dy
                            x, y = x1, y1
                            while True:
                                points.append((x, y))
                                if x == x2 and y == y2:
                                    break
                                e2 = 2 * err
                                if e2 > -dy:
                                    err -= dy
                                    x += sx
                                if e2 < dx:
                                    err += dx
                                    y += sy
                            return points

                        line_points = get_line_points(color_pixel, found_pixel)

                        line_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        for point in line_points:
                            if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
                                line_mask[point[1], point[0]] = 255

                        perp_dx = -unit_dy
                        perp_dy = unit_dx

                        left_disjoint_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        right_disjoint_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                if disjoint_mask[y, x] == 255:
                                    min_dist = float('inf')
                                    closest_point = None
                                    for point in line_points:
                                        dist = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_point = point

                                    if closest_point:
                                        pixel_vec = (x - closest_point[0], y - closest_point[1])
                                        dot_product = pixel_vec[0] * perp_dx + pixel_vec[1] * perp_dy

                                        if dot_product > 0:
                                            left_disjoint_mask[y, x] = 255
                                        else:
                                            right_disjoint_mask[y, x] = 255

                        _, black_mask = cv2.threshold(gray0, 100, 255, cv2.THRESH_BINARY_INV)
                        left_black_pixels = cv2.countNonZero(cv2.bitwise_and(left_disjoint_mask, black_mask))
                        right_black_pixels = cv2.countNonZero(cv2.bitwise_and(right_disjoint_mask, black_mask))

                        print(f"左侧不相交区域的黑色像素数量: {left_black_pixels}")
                        print(f"右侧不相交区域的黑色像素数量: {right_black_pixels}")

                        if debug:
                            black_visualization = img.copy()

                            for y in range(img.shape[0]):
                                for x in range(img.shape[1]):
                                    if disjoint_mask[y, x] == 255 and black_mask[y, x] == 255:
                                        if left_disjoint_mask[y, x] == 255:
                                            black_visualization[y, x] = (255, 0, 0)
                                        elif right_disjoint_mask[y, x] == 255:
                                            black_visualization[y, x] = (0, 255, 0)
                            if not os.path.exists('output'):
                                os.makedirs('output')
                            filename = os.path.basename(json_path).replace('_res.json', '')
                            mask_output_path = os.path.join('output', f'{filename}_black_mask1.png')
                            cv2.imwrite(mask_output_path, black_mask)
                            black_output_path = os.path.join('output', f'{filename}_black_pixels.png')
                            cv2.imwrite(black_output_path, black_visualization)
                            print(f"已保存黑色像素可视化结果到: {black_output_path}")

                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data['left_black_pixels'] = int(left_black_pixels)
                        data['right_black_pixels'] = int(right_black_pixels)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        print(f"已将黑色像素数量更新到: {json_path}")

    return img, found_pixel, left_black_pixels, right_black_pixels


def check_colors_in_line(color_centers):
    """
    检测颜色是否在一条直线上

    参数:
        color_centers: 颜色中心点字典

    返回:
        bool: 是否在一条直线上
    """
    is_linear = False
    if len(color_centers) >= 2:
        # 检查是否所有颜色都在一条直线上（允许5像素偏差）
        # 收集所有颜色中心点
        centers = list(color_centers.values())

        # 计算拟合直线
        x = [p[0] for p in centers]
        y = [p[1] for p in centers]

        # 使用线性回归计算拟合直线
        if len(centers) == 2:
            # 两个点一定在一条直线上
            is_linear = True
        else:
            # 计算回归直线
            coefficients = np.polyfit(x, y, 1)
            line_func = np.poly1d(coefficients)

            # 检查所有点是否在直线上（允许5像素偏差）
            is_linear = True
            for i, (px, py) in enumerate(centers):
                predicted_y = line_func(px)
                if abs(py - predicted_y) > 5:
                    is_linear = False
                    break

    return is_linear


def calculate_leftmost(mask):
    """
    计算颜色像素的最左端坐标

    参数:
        mask: 颜色掩码

    返回:
        tuple: 最左端像素坐标 (x, y)
    """
    if cv2.countNonZero(mask) == 0:
        return None
    # 找到所有非零像素
    coords = np.column_stack(np.where(mask > 0))
    # 按x坐标排序，取最小的x坐标
    coords_sorted = coords[coords[:, 1].argsort()]
    leftmost = coords_sorted[0]
    # 转换为 (x, y) 格式
    return (leftmost[1], leftmost[0])


def process_leftmost_pixels(img, mask_red, mask_yellow, mask_green, red_center, yellow_center, green_center, angle=0.0,
                            debug=True, is_linear=False):
    """
    处理最左端像素，根据文本框倾斜角度从颜色圆心沿角度方向找最边缘点

    参数:
        img: 原始图像
        mask_red: 红色掩码
        mask_yellow: 黄色掩码
        mask_green: 绿色掩码
        red_center: 红色中心坐标
        yellow_center: 黄色中心坐标
        green_center: 绿色中心坐标
        angle: 文本框倾斜角度（弧度）
        debug: 是否进行可视化绘制，默认为True

    返回:
        tuple: (可视化图像, 颜色线信息)
    """
    if abs(angle) > 0.02:
        print(f"使用倾斜角度 {np.degrees(angle):.2f} 度计算边缘点")
        red_leftmost = calculate_edge_point_along_angle(mask_red, red_center, angle, 'left')
        yellow_leftmost = calculate_edge_point_along_angle(mask_yellow, yellow_center, angle, 'left')
        green_leftmost = calculate_edge_point_along_angle(mask_green, green_center, angle, 'left')
    else:
        red_leftmost = calculate_leftmost(mask_red)
        yellow_leftmost = calculate_leftmost(mask_yellow)
        green_leftmost = calculate_leftmost(mask_green)

    if red_leftmost:
        print(f"红色像素最左端坐标: {red_leftmost}")
    if yellow_leftmost:
        print(f"黄色像素最左端坐标: {yellow_leftmost}")
    if green_leftmost:
        print(f"绿色像素最左端坐标: {green_leftmost}")

    vis_img = img.copy()

    if debug:
        if red_center:
            cv2.circle(vis_img, red_center, 2, (255, 0, 0), -1)
        if yellow_center:
            cv2.circle(vis_img, yellow_center, 2, (255, 0, 0), -1)
        if green_center:
            cv2.circle(vis_img, green_center, 2, (255, 0, 0), -1)

        if red_leftmost:
            cv2.circle(vis_img, red_leftmost, 2, (255, 0, 0), -1)
        if yellow_leftmost:
            cv2.circle(vis_img, yellow_leftmost, 2, (255, 0, 0), -1)

        if abs(angle) > 0.02:
            line_length = 100
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            if red_center:
                line_start_h = (int(red_center[0] - line_length), int(red_center[1]))
                line_end_h = (int(red_center[0] + line_length), int(red_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(red_center[0] - cos_a * line_length), int(red_center[1] + sin_a * line_length))
                line_end = (int(red_center[0] + cos_a * line_length), int(red_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if red_leftmost:
                    cv2.line(vis_img, red_center, red_leftmost, (255, 0, 255), 1)

            if yellow_center:
                line_start_h = (int(yellow_center[0] - line_length), int(yellow_center[1]))
                line_end_h = (int(yellow_center[0] + line_length), int(yellow_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(yellow_center[0] - cos_a * line_length), int(yellow_center[1] + sin_a * line_length))
                line_end = (int(yellow_center[0] + cos_a * line_length), int(yellow_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if yellow_leftmost:
                    cv2.line(vis_img, yellow_center, yellow_leftmost, (255, 0, 255), 1)

            if green_center:
                line_start_h = (int(green_center[0] - line_length), int(green_center[1]))
                line_end_h = (int(green_center[0] + line_length), int(green_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(green_center[0] - cos_a * line_length), int(green_center[1] + sin_a * line_length))
                line_end = (int(green_center[0] + cos_a * line_length), int(green_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if green_leftmost:
                    cv2.line(vis_img, green_center, green_leftmost, (255, 0, 255), 1)

            print(f"已绘制水平参考线（灰色）、逆时针旋转角度后的延长线（黄色）和到边缘点的连线（紫色）")

    color_line_info = None
    leftmost_points = []
    if red_leftmost:
        leftmost_points.append(red_leftmost)
    if yellow_leftmost:
        leftmost_points.append(yellow_leftmost)
    if green_leftmost:
        leftmost_points.append(green_leftmost)
    linear_point = None
    if leftmost_points:
        leftmost_sorted = sorted(leftmost_points, key=lambda p: p[0])

        if is_linear:
            far_left_points = [leftmost_sorted[0]]
            linear_point = far_left_points
        else:
            far_left_points = leftmost_sorted[:2]

        if len(far_left_points) == 1:
            center_point = far_left_points[0]
            extended_start = (center_point[0], center_point[1] - 10)
            extended_end = (center_point[0], center_point[1] + 10)
            print(f"左侧只有一种颜色，扩展像素: {center_point}")

            if debug:
                cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

        elif len(far_left_points) >= 2:
            far_left_points.sort(key=lambda p: p[1])
            point1 = far_left_points[0]
            point2 = far_left_points[-1]
            print(f"左侧有两种或以上颜色，使用原始方法连接: {point1} 和 {point2}")

            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = ((dx ** 2) + (dy ** 2)) ** 0.5

            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length
            else:
                unit_dx, unit_dy = 0, 0

            extend_length = length / 2

            extended_start = (int(point1[0] - unit_dx * extend_length),
                              int(point1[1] - unit_dy * extend_length))
            extended_end = (int(point2[0] + unit_dx * extend_length),
                            int(point2[1] + unit_dy * extend_length))

            if debug:
                cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

    color_points = []
    if red_leftmost:
        color_points.append(red_leftmost)
    if yellow_leftmost:
        color_points.append(yellow_leftmost)
    if green_leftmost:
        color_points.append(green_leftmost)

    return vis_img, color_line_info, linear_point


def calculate_rightmost(mask):
    """
    计算颜色像素的最右端坐标

    参数:
        mask: 颜色掩码

    返回:
        tuple: 最右端像素坐标 (x, y)
    """
    if cv2.countNonZero(mask) == 0:
        return None
    # 找到所有非零像素
    coords = np.column_stack(np.where(mask > 0))
    # 按x坐标排序，取最大的x坐标
    coords_sorted = coords[coords[:, 1].argsort()]
    rightmost = coords_sorted[-1]
    # 转换为 (x, y) 格式
    return (rightmost[1], rightmost[0])


def calculate_edge_point_along_angle(mask, center, angle, direction='right'):
    """
    根据倾斜角度，从颜色圆心出发沿着指定方向找最边缘点（逆时针旋转方向）

    参数:
        mask: 颜色掩码
        center: 颜色圆心坐标 (x, y)
        angle: 文本框倾斜角度（弧度）
        direction: 方向，'right' 表示往右找，'left' 表示往左找

    返回:
        tuple: 最边缘像素坐标 (x, y)
    """
    if cv2.countNonZero(mask) == 0 or center is None:
        return None

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    best_point = None
    best_projection = float('-inf') if direction == 'right' else float('inf')

    for coord in coords:
        x, y = coord[1], coord[0]

        dx = x - center[0]
        dy = y - center[1]

        projection = dx * cos_a - dy * sin_a

        if direction == 'right':
            if projection > best_projection:
                best_projection = projection
                best_point = (x, y)
        else:
            if projection < best_projection:
                best_projection = projection
                best_point = (x, y)

    return best_point


def process_rightmost_pixels(img, mask_red, mask_yellow, mask_green, red_center, yellow_center, green_center, angle=0.0,
                             debug=True, is_linear=False):
    """
    处理最右端像素，根据文本框倾斜角度从颜色圆心沿角度方向找最边缘点

    参数:
        img: 原始图像
        mask_red: 红色掩码
        mask_yellow: 黄色掩码
        mask_green: 绿色掩码
        red_center: 红色中心坐标
        yellow_center: 黄色中心坐标
        green_center: 绿色中心坐标
        angle: 文本框倾斜角度（弧度）
        debug: 是否进行可视化绘制，默认为True

    返回:
        tuple: (可视化图像, 颜色线信息)
    """
    if abs(angle) > 0.02:
        print(f"使用倾斜角度 {np.degrees(angle):.2f} 度计算边缘点")
        red_rightmost = calculate_edge_point_along_angle(mask_red, red_center, angle, 'right')
        yellow_rightmost = calculate_edge_point_along_angle(mask_yellow, yellow_center, angle, 'right')
        green_rightmost = calculate_edge_point_along_angle(mask_green, green_center, angle, 'right')
    else:
        red_rightmost = calculate_rightmost(mask_red)
        yellow_rightmost = calculate_rightmost(mask_yellow)
        green_rightmost = calculate_rightmost(mask_green)

    if red_rightmost:
        print(f"红色像素最右端坐标: {red_rightmost}")
    if yellow_rightmost:
        print(f"黄色像素最右端坐标: {yellow_rightmost}")
    if green_rightmost:
        print(f"绿色像素最右端坐标: {green_rightmost}")

    vis_img = img.copy()

    if debug:
        if red_center:
            cv2.circle(vis_img, red_center, 2, (255, 0, 0), -1)
        if yellow_center:
            cv2.circle(vis_img, yellow_center, 2, (255, 0, 0), -1)
        if green_center:
            cv2.circle(vis_img, green_center, 2, (255, 0, 0), -1)

        if red_rightmost:
            cv2.circle(vis_img, red_rightmost, 2, (255, 0, 0), -1)
        if yellow_rightmost:
            cv2.circle(vis_img, yellow_rightmost, 2, (255, 0, 0), -1)

        if abs(angle) > 0.02:
            line_length = 100
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            if red_center:
                line_start_h = (int(red_center[0] - line_length), int(red_center[1]))
                line_end_h = (int(red_center[0] + line_length), int(red_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(red_center[0] - cos_a * line_length), int(red_center[1] + sin_a * line_length))
                line_end = (int(red_center[0] + cos_a * line_length), int(red_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if red_rightmost:
                    cv2.line(vis_img, red_center, red_rightmost, (255, 0, 255), 1)

            if yellow_center:
                line_start_h = (int(yellow_center[0] - line_length), int(yellow_center[1]))
                line_end_h = (int(yellow_center[0] + line_length), int(yellow_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(yellow_center[0] - cos_a * line_length), int(yellow_center[1] + sin_a * line_length))
                line_end = (int(yellow_center[0] + cos_a * line_length), int(yellow_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if yellow_rightmost:
                    cv2.line(vis_img, yellow_center, yellow_rightmost, (255, 0, 255), 1)

            if green_center:
                line_start_h = (int(green_center[0] - line_length), int(green_center[1]))
                line_end_h = (int(green_center[0] + line_length), int(green_center[1]))
                cv2.line(vis_img, line_start_h, line_end_h, (128, 128, 128), 1)
                line_start = (int(green_center[0] - cos_a * line_length), int(green_center[1] + sin_a * line_length))
                line_end = (int(green_center[0] + cos_a * line_length), int(green_center[1] - sin_a * line_length))
                cv2.line(vis_img, line_start, line_end, (0, 255, 255), 1)
                if green_rightmost:
                    cv2.line(vis_img, green_center, green_rightmost, (255, 0, 255), 1)

            print(f"已绘制水平参考线（灰色）、逆时针旋转角度后的延长线（黄色）和到边缘点的连线（紫色）")

    color_line_info = None
    rightmost_points = []
    if red_rightmost:
        rightmost_points.append(red_rightmost)
    if yellow_rightmost:
        rightmost_points.append(yellow_rightmost)
    if green_rightmost:
        rightmost_points.append(green_rightmost)
    linear_point = None
    if rightmost_points:
        leftmost_sorted = sorted(rightmost_points, key=lambda p: p[0], reverse=True)

        if is_linear:
            far_right_points = [leftmost_sorted[0]]
            linear_point = far_right_points
        else:
            far_right_points = leftmost_sorted[:2]

        if len(far_right_points) == 1:
            center_point = far_right_points[0]
            extended_start = (center_point[0], center_point[1] - 25)
            extended_end = (center_point[0], center_point[1] + 25)
            print(f"右侧只有一种颜色，扩展像素: {center_point}")

            if debug:
                cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

        elif len(far_right_points) >= 2:
            far_right_points.sort(key=lambda p: p[1])
            point1 = far_right_points[0]
            point2 = far_right_points[-1]
            print(f"右侧有两种或以上颜色，使用原始方法连接: {point1} 和 {point2}")

            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = ((dx ** 2) + (dy ** 2)) ** 0.5

            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length
            else:
                unit_dx, unit_dy = 0, 0

            extend_length = length / 2

            extended_start = (int(point1[0] - unit_dx * extend_length),
                              int(point1[1] - unit_dy * extend_length))
            extended_end = (int(point2[0] + unit_dx * extend_length),
                            int(point2[1] + unit_dy * extend_length))

            if debug:
                cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

    color_points = []
    if red_rightmost:
        color_points.append(red_rightmost)
    if yellow_rightmost:
        color_points.append(yellow_rightmost)
    if green_rightmost:
        color_points.append(green_rightmost)

    return vis_img, color_line_info, linear_point


def save_visualization(img, image_path, debug=True):
    """
    保存可视化结果

    参数:
        img: 可视化图像
        image_path: 原始图像路径
        debug: 是否保存可视化结果，默认为True
    """
    if not debug:
        return

    if not os.path.exists('output'):
        os.makedirs('output')

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    vis_output_path = os.path.join('output', f'{name}_color_centers{ext}')

    cv2.imwrite(vis_output_path, img)
    print(f"颜色中心可视化结果已保存到: {vis_output_path}")


if __name__ == "__main__":
    # 输入图像路径
    input_image = r"D:\work\ocr+Transformer\little_light\micro_0084_X8.jpg"

    # 输入目标字符
    target_char = "X8"
    # target_char = "SVII"
    # target_char = "S"

    time1 = time.time()

    # 进行OCR识别
    result = ocr.predict([input_image])
    for i, res in enumerate(result):
        res.save_to_img("output")
        res.save_to_json("output")
        # 检测颜色像素
        detect_colors(input_image, target_char)

    time2 = time.time()
    print(f"处理时间: {(time2 - time1):.4f} 秒")
