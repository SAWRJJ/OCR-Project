import copy
import time
import cv2
import numpy as np
import json
import os

from paddleocr import PaddleOCR

# 初始化OCR
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)  # 更换 PP-OCRv5_mobile 模型


def draw_target_char_left_edge(img, json_path, target_char='S5', color_line_info=None, is_linear=False):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 找到目标字符对应的索引
    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break

    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None

    # 获取对应的dt_polys
    dt_polys = data.get('dt_polys', [])
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None

    # 获取文本框轮廓坐标
    poly = dt_polys[target_index]
    print(f"{target_char}对应的文本框轮廓: {poly}")

    # 绘制原始文本框（粉色）
    cv2.polylines(img, [np.array(poly)], isClosed=True, color=(147, 20, 255), thickness=2)  # 粉色线条

    # 初始化变量，标记是否使用黄色轮廓的左侧边
    use_yellow_left_edge = False
    yellow_left_edge = None

    # 计算文本框长度（水平方向）
    x_coords = [point[0] for point in poly]
    textbox_length = max(x_coords) - min(x_coords)
    print(f"文本框长度: {textbox_length} 像素")

    # 如果文本框长度大于300像素，在最右侧取50长度的轮廓用黄色画出来
    if textbox_length > 300:
        # 找到最右侧的x坐标
        rightmost_x = max(x_coords)
        # 计算40长度的左侧x坐标
        yellow_box_left = rightmost_x - 40

        # 获取文本框的y范围
        y_coords = [point[1] for point in poly]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 创建黄色轮廓（最右侧40长度）
        yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                          (yellow_box_left, max_y)]

        # 绘制黄色轮廓
        cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)  # 黄色线条
        print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")

        # 对黄色轮廓区域进行裁剪并重新OCR识别
        # 计算裁剪区域的边界
        x1 = int(min(point[0] for point in yellow_contour))
        y1 = int(min(point[1] for point in yellow_contour))
        x2 = int(max(point[0] for point in yellow_contour))
        y2 = int(max(point[1] for point in yellow_contour))

        # 确保裁剪区域在图像范围内
        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 裁剪图像
        if x2 > x1 and y2 > y1:
            cropped_img = img[y1:y2, x1:x2]

            # 保存裁剪后的图像
            if not os.path.exists('output'):
                os.makedirs('output')

            filename = os.path.basename(json_path).replace('_res.json', '')
            crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
            cv2.imwrite(crop_output_path, cropped_img)
            print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")

            # 对裁剪区域重新进行OCR识别
            print("对黄色轮廓区域重新进行OCR识别...")
            # 使用与主OCR相同的配置
            crop_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

            # 识别裁剪后的图像
            crop_result = crop_ocr.predict([crop_output_path])

            # 检查OCR识别是否成功
            ocr_success = False
            # 打印识别结果
            print("黄色轮廓区域的OCR识别结果:")
            for res in crop_result:
                if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                    ocr_success = True
                    for line in res["rec_texts"]:
                        print(f"识别文本: {line}")

            # 将识别结果绘制到裁剪图像上
            for res in crop_result:
                if hasattr(res, 'save_to_img'):
                    res.save_to_img("output")
                    res.save_to_json("output")

            # 如果OCR识别成功，使用黄色轮廓的左侧边作为文本框的左侧边
            if ocr_success:
                print("OCR识别成功，将使用黄色轮廓的左侧边作为文本框左侧边")
                # 计算黄色轮廓的左侧边
                yellow_left_edge = [(yellow_box_left, min_y), (yellow_box_left, max_y)]
                use_yellow_left_edge = True
                print(f"已设置使用黄色轮廓的左侧边: {yellow_left_edge}")

    # 计算左侧边
    # 按x坐标排序，找到最小的两个点（左侧边）
    poly_sorted = sorted(poly, key=lambda point: point[0])
    left_points = poly_sorted[:2]
    # 按y坐标排序，确保顺序正确
    left_points.sort(key=lambda point: point[1])

    # 如果OCR识别成功，使用黄色轮廓的左侧边
    if use_yellow_left_edge and yellow_left_edge:
        left_points = yellow_left_edge
        print(f"已使用黄色轮廓的左侧边: {left_points}")

    left_edge = tuple(map(tuple, left_points))
    print(f"左侧边坐标: {left_edge}")

    # 延长左侧边到与颜色线相同长度
    extended_left_start = left_edge[0]
    extended_left_end = left_edge[1]

    adjusted_color_line = None
    polygon = None
    if color_line_info:
        extended_start, extended_end = color_line_info

        # 检查颜色线是否水平排列
        color_line_dx = abs(extended_end[0] - extended_start[0])
        color_line_dy = abs(extended_end[1] - extended_start[1])

        # 如果是水平排列，调整为垂直线条，与文本框左侧边同高
        if color_line_dy < color_line_dx * 0.5:  # 水平排列判断条件
            # 找到最右侧的x坐标
            rightmost_x = max(extended_start[0], extended_end[0])
            # 创建垂直线条，与文本框左侧边同高
            adjusted_color_line = [(rightmost_x, left_edge[0][1]), (rightmost_x, left_edge[1][1])]
            print(f"颜色线水平排列，已调整为垂直线条: {adjusted_color_line}")
            extended_start, extended_end = adjusted_color_line

        # 计算颜色线长度
        color_line_dx = extended_end[0] - extended_start[0]
        color_line_dy = extended_end[1] - extended_start[1]
        color_line_length = ((color_line_dx ** 2) + (color_line_dy ** 2)) ** 0.5

        # 计算左侧边长度
        left_edge_dx = left_edge[1][0] - left_edge[0][0]
        left_edge_dy = left_edge[1][1] - left_edge[0][1]
        left_edge_length = ((left_edge_dx ** 2) + (left_edge_dy ** 2)) ** 0.5

        if left_edge_length > 0 and color_line_length > 0:
            # 计算单位方向向量
            unit_dx = left_edge_dx / left_edge_length
            unit_dy = left_edge_dy / left_edge_length

            # 计算延长后的起点和终点
            scale_factor = color_line_length / left_edge_length
            extended_left_start = (int(left_edge[0][0] - unit_dx * left_edge_length * (scale_factor - 1) / 2),
                                   int(left_edge[0][1] - unit_dy * left_edge_length * (scale_factor - 1) / 2))
            extended_left_end = (int(left_edge[1][0] + unit_dx * left_edge_length * (scale_factor - 1) / 2),
                                 int(left_edge[1][1] + unit_dy * left_edge_length * (scale_factor - 1) / 2))

    # 绘制延长后的左侧边
    cv2.line(img, extended_left_start, extended_left_end, (0, 0, 255), 2)  # 红色线条

    # 构成封闭多边形
    if color_line_info:
        # 使用调整后的颜色线信息（如果有）
        if adjusted_color_line:
            extended_start, extended_end = adjusted_color_line
        else:
            extended_start, extended_end = color_line_info

        # 构建多边形顶点
        # 顺序：颜色线外扩起点 -> 颜色线外扩终点 -> 左侧边延长终点 -> 左侧边延长起点
        polygon = [extended_start, extended_end, extended_left_end, extended_left_start]

        # 绘制多边形
        cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=1)  # 绿色线条

    return img, polygon


def calculate_black_pixels(img, polygon, json_path, data):
    """
    统计多边形内的黑色像素数量

    参数:
        img: 图像
        polygon: 多边形顶点
        json_path: JSON文件路径
        data: JSON数据字典

    返回:
        int: 黑色像素数量
    """
    # 创建掩码
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 255)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值处理，提取黑色像素
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 计算多边形内的黑色像素
    black_pixels_in_polygon = cv2.countNonZero(cv2.bitwise_and(mask, black_mask))

    print(f"多边形内的黑色像素数量: {black_pixels_in_polygon}")

    # 更新JSON文件
    data['black_pixels_in_polygon'] = int(black_pixels_in_polygon)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"已将黑色像素数量更新到: {json_path}")

    return black_pixels_in_polygon


def draw_target_char_right_edge(img, json_path, target_char='X', color_line_info=None, is_linear=False):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 找到目标字符对应的索引
    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break

    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None

    # 获取对应的dt_polys
    dt_polys = data.get('dt_polys', [])
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None

    # 获取文本框轮廓坐标
    poly = dt_polys[target_index]
    print(f"{target_char}对应的文本框轮廓: {poly}")

    # 绘制原始文本框（粉色）
    cv2.polylines(img, [np.array(poly)], isClosed=True, color=(147, 20, 255), thickness=2)  # 粉色线条

    # 初始化变量，标记是否使用黄色轮廓的右侧边
    use_yellow_right_edge = False
    yellow_right_edge = None

    # 计算文本框长度（水平方向）
    x_coords = [point[0] for point in poly]
    textbox_length = max(x_coords) - min(x_coords)
    print(f"文本框长度: {textbox_length} 像素")

    # 如果文本框长度大于300像素，在最右侧取50长度的轮廓用黄色画出来
    if textbox_length > 300:
        # 找到最右侧的x坐标
        rightmost_x = max(x_coords)
        # 计算40长度的左侧x坐标
        yellow_box_left = rightmost_x - 40

        # 获取文本框的y范围
        y_coords = [point[1] for point in poly]
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 创建黄色轮廓（最右侧40长度）
        yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                          (yellow_box_left, max_y)]

        # 绘制黄色轮廓
        cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)  # 黄色线条
        print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")

        # 对黄色轮廓区域进行裁剪并重新OCR识别
        # 计算裁剪区域的边界
        x1 = int(min(point[0] for point in yellow_contour))
        y1 = int(min(point[1] for point in yellow_contour))
        x2 = int(max(point[0] for point in yellow_contour))
        y2 = int(max(point[1] for point in yellow_contour))

        # 确保裁剪区域在图像范围内
        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 裁剪图像
        if x2 > x1 and y2 > y1:
            cropped_img = img[y1:y2, x1:x2]

            # 保存裁剪后的图像
            if not os.path.exists('output'):
                os.makedirs('output')

            filename = os.path.basename(json_path).replace('_res.json', '')
            crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
            cv2.imwrite(crop_output_path, cropped_img)
            print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")

            # 对裁剪区域重新进行OCR识别
            print("对黄色轮廓区域重新进行OCR识别...")
            # 使用与主OCR相同的配置
            crop_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

            # 识别裁剪后的图像
            crop_result = crop_ocr.predict([crop_output_path])

            # 检查OCR识别是否成功
            ocr_success = False
            # 打印识别结果
            print("黄色轮廓区域的OCR识别结果:")
            for res in crop_result:
                if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                    ocr_success = True
                    for line in res["rec_texts"]:
                        print(f"识别文本: {line}")

            # 将识别结果绘制到裁剪图像上
            for res in crop_result:
                if hasattr(res, 'save_to_img'):
                    res.save_to_img("output")
                    res.save_to_json("output")

            # 如果OCR识别成功，使用黄色轮廓的右侧边作为文本框的右侧边
            if ocr_success:
                print("OCR识别成功，将使用黄色轮廓的右侧边作为文本框右侧边")
                # 计算黄色轮廓的右侧边
                yellow_right_edge = [(rightmost_x, min_y), (rightmost_x, max_y)]
                use_yellow_right_edge = True
                print(f"已设置使用黄色轮廓的右侧边: {yellow_right_edge}")

    # 计算右侧边
    # 按x坐标排序，找到最大的两个点（右侧边）
    poly_sorted = sorted(poly, key=lambda point: point[0])
    right_points = poly_sorted[-2:]
    # 按y坐标排序，确保顺序正确
    right_points.sort(key=lambda point: point[1])

    # 如果OCR识别成功，使用黄色轮廓的右侧边
    if use_yellow_right_edge and yellow_right_edge:
        right_points = yellow_right_edge
        print(f"已使用黄色轮廓的右侧边: {right_points}")

    right_edge = tuple(map(tuple, right_points))
    print(f"右侧边坐标: {right_edge}")

    # 延长右侧边到与颜色线相同长度
    extended_right_start = right_edge[0]
    extended_right_end = right_edge[1]

    adjusted_color_line = None
    polygon = None
    if color_line_info:
        extended_start, extended_end = color_line_info

        # 检查颜色线是否水平排列
        color_line_dx = abs(extended_end[0] - extended_start[0])
        color_line_dy = abs(extended_end[1] - extended_start[1])

        # 如果是水平排列，调整为垂直线条，与文本框右侧边同高
        if color_line_dy < color_line_dx * 0.5:  # 水平排列判断条件
            # 找到最左侧的x坐标
            leftmost_x = min(extended_start[0], extended_end[0])
            # 创建垂直线条，与文本框右侧边同高
            adjusted_color_line = [(leftmost_x, right_edge[0][1]), (leftmost_x, right_edge[1][1])]
            print(f"颜色线水平排列，已调整为垂直线条: {adjusted_color_line}")
            extended_start, extended_end = adjusted_color_line

        # 计算颜色线长度
        color_line_dx = extended_end[0] - extended_start[0]
        color_line_dy = extended_end[1] - extended_start[1]
        color_line_length = ((color_line_dx ** 2) + (color_line_dy ** 2)) ** 0.5

        # 计算右侧边长度
        right_edge_dx = right_edge[1][0] - right_edge[0][0]
        right_edge_dy = right_edge[1][1] - right_edge[0][1]
        right_edge_length = ((right_edge_dx ** 2) + (right_edge_dy ** 2)) ** 0.5

        if right_edge_length > 0 and color_line_length > 0:
            # 计算单位方向向量
            unit_dx = right_edge_dx / right_edge_length
            unit_dy = right_edge_dy / right_edge_length

            # 计算延长后的起点和终点
            scale_factor = color_line_length / right_edge_length
            extended_right_start = (int(right_edge[0][0] - unit_dx * right_edge_length * (scale_factor - 1) / 2),
                                    int(right_edge[0][1] - unit_dy * right_edge_length * (scale_factor - 1) / 2))
            extended_right_end = (int(right_edge[1][0] + unit_dx * right_edge_length * (scale_factor - 1) / 2),
                                  int(right_edge[1][1] + unit_dy * right_edge_length * (scale_factor - 1) / 2))

    # 绘制延长后的右侧边
    cv2.line(img, extended_right_start, extended_right_end, (0, 0, 255), 2)  # 红色线条

    # 构成封闭多边形
    if color_line_info:
        # 使用调整后的颜色线信息（如果有）
        if adjusted_color_line:
            extended_start, extended_end = adjusted_color_line
        else:
            extended_start, extended_end = color_line_info

        # 构建多边形顶点
        # 顺序：颜色线外扩起点 -> 颜色线外扩终点 -> 右侧边延长终点 -> 右侧边延长起点
        polygon = [extended_start, extended_end, extended_right_end, extended_right_start]

        # 绘制多边形
        cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=1)  # 绿色线条

    return img, polygon


def detect_colors(image_path, target_char):
    """
    检测图像中的颜色像素并分析位置关系

    参数:
        image_path: 图像路径
        target_char: 目标字符，用于确定使用哪种检测方式
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色范围（HSV）
    # 红色范围（两个区间，因为红色在HSV中是环绕的）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 黄色范围
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 绿色范围
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([70, 255, 255])

    # 创建掩码
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 计算像素数量
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    # 计算总像素数
    total_pixels = img.shape[0] * img.shape[1]

    # 打印结果
    print(f"图像: {image_path}")
    print(f"红色像素: {red_pixels} ({red_pixels / total_pixels * 100:.2f}%)")
    print(f"黄色像素: {yellow_pixels} ({yellow_pixels / total_pixels * 100:.2f}%)")
    print(f"绿色像素: {green_pixels} ({green_pixels / total_pixels * 100:.2f}%)")

    # 计算颜色像素的中心坐标
    def calculate_center(mask):
        if cv2.countNonZero(mask) == 0:
            return None
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    # 计算各颜色中心坐标
    red_center = calculate_center(mask_red)
    yellow_center = calculate_center(mask_yellow)
    green_center = calculate_center(mask_green)

    # 打印中心坐标
    if red_center:
        print(f"红色像素中心坐标: {red_center}")
    if yellow_center:
        print(f"黄色像素中心坐标: {yellow_center}")
    if green_center:
        print(f"绿色像素中心坐标: {green_center}")

    # 收集有效的颜色中心点
    color_centers = {}
    if red_center:
        color_centers['红色'] = red_center
    if yellow_center:
        color_centers['黄色'] = yellow_center
    if green_center:
        color_centers['绿色'] = green_center

    # 分析颜色位置关系
    analyze_color_relationships(color_centers)

    # 检测颜色是否在一条直线上
    is_linear = check_colors_in_line(color_centers)
    print(f"颜色是否在一条直线上: {is_linear}")
    # 根据目标字符选择不同的像素检测方法
    if target_char == 'X':
        # 对于X，使用最左端像素
        vis_img, color_line_info = process_leftmost_pixels(img, mask_red, mask_yellow, mask_green, red_center,
                                                           yellow_center, green_center)
    else:
        # 对于其他字符，使用最右端像素
        vis_img, color_line_info = process_rightmost_pixels(img, mask_red, mask_yellow, mask_green, red_center,
                                                            yellow_center, green_center)

    # 绘制目标字符边
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    json_path = os.path.join("output", f"{name}_res.json")

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 计算各颜色的最右端像素
    red_rightmost = calculate_rightmost(mask_red)
    yellow_rightmost = calculate_rightmost(mask_yellow)
    green_rightmost = calculate_rightmost(mask_green)

    # 计算最右侧的颜色像素
    rightmost_color_pixel = None
    rightmost_pixels = []
    if red_rightmost:
        rightmost_pixels.append(red_rightmost)
    if yellow_rightmost:
        rightmost_pixels.append(yellow_rightmost)
    if green_rightmost:
        rightmost_pixels.append(green_rightmost)

    if rightmost_pixels:
        # 按x坐标排序，取最大的
        rightmost_pixels.sort(key=lambda p: p[0], reverse=True)
        rightmost_color_pixel = rightmost_pixels[0]
        print(f"计算得到最右侧彩色像素: {rightmost_color_pixel}")

    # 绘制目标字符边（不包含黑色像素统计）
    polygon = None
    if target_char == 'X':
        vis_img, polygon = draw_target_char_right_edge(vis_img, json_path, color_line_info=color_line_info,
                                                       target_char=target_char, is_linear=is_linear)
    else:
        vis_img, polygon = draw_target_char_left_edge(vis_img, json_path, color_line_info=color_line_info,
                                                      target_char=target_char, is_linear=is_linear)

    # 处理黑色像素统计
    if is_linear:
        # 如果颜色在一条直线上，执行单线检测
        print("执行单线检测")
        vis_img, found_pixel = single_line_detection(vis_img, json_path, target_char, rightmost_color_pixel, img)
    else:
        # 如果颜色不在一条直线上，执行双线黑色像素统计
        print("执行双线黑色像素统计")
        # 使用返回的多边形信息计算黑色像素数量
        if polygon:
            calculate_black_pixels(vis_img, polygon, json_path, data)

    # 保存可视化结果
    save_visualization(vis_img, image_path)


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


def single_line_detection(img, json_path, target_char, rightmost_color_pixel, origin_img):
    """
    单线检测函数，根据文本内容从文本框的左侧或右侧边开始找黑色像素

    参数:
        img: 原始图像
        json_path: JSON文件路径
        target_char: 目标字符
        rightmost_color_pixel: 最右侧的颜色坐标

    返回:
        tuple: (可视化图像, 找到的黑色像素位置)
    """
    # 保存输入的原始图像
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = os.path.basename(json_path).replace('_res.json', '')
    input_output_path = os.path.join('output', f'{filename}_input.png')
    cv2.imwrite(input_output_path, origin_img)
    print(f"已保存输入的原始图像到: {input_output_path}")

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 找到目标字符对应的索引
    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break

    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img, None

    # 获取对应的dt_polys
    dt_polys = data.get('dt_polys', [])
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img, None

    # 获取文本框轮廓坐标
    poly = dt_polys[target_index]

    # 计算文本框的边界
    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # 转换为灰度图像
    gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    gray0 = copy.deepcopy(gray)
    # 阈值处理，提取黑色像素
    _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # 保存black_mask
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = os.path.basename(json_path).replace('_res.json', '')
    mask_output_path = os.path.join('output', f'{filename}_black_mask.png')
    cv2.imwrite(mask_output_path, black_mask)
    print(f"已保存黑色像素掩码到: {mask_output_path}")

    # 初始化找到的黑色像素位置
    found_pixel = None

    # 根据目标字符决定从左侧还是右侧开始找
    if target_char.startswith('S'):
        # 从右侧边开始找，先左右（按列）再上下（按行）
        print("从文本框右侧边开始找黑色像素")
        # 从最右侧列开始，向左扫描每一列
        for x in range(max_x, min_x - 1, -1):
            # 遍历当前列的每一行，从上到下
            for y in range(min_y, max_y + 1):
                # 检查是否在图像范围内
                if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                    # 检查是否是黑色像素
                    if black_mask[y, x] == 255:
                        found_pixel = (x, y)
                        print(f"找到第一个黑色像素: {found_pixel}")
                        break
            if found_pixel:
                break
    elif target_char == 'X':
        # 从左侧边开始找，先左右（按列）再上下（按行）
        print("从文本框左侧边开始找黑色像素")
        # 从最左侧列开始，向右扫描每一列
        for x in range(min_x, max_x + 1):
            # 遍历当前列的每一行，从上到下
            for y in range(min_y, max_y + 1):
                # 检查是否在图像范围内
                if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                    # 检查是否是黑色像素
                    if black_mask[y, x] == 255:
                        found_pixel = (x, y)
                        print(f"找到第一个黑色像素: {found_pixel}")
                        break
            if found_pixel:
                break

    # 可视化找到的黑色像素
    if found_pixel:
        # 绘制一个红色圆圈标记找到的黑色像素
        cv2.circle(img, found_pixel, 3, (0, 0, 255), -1)
        print("已可视化找到的黑色像素")

        # 使用传入的最右侧颜色坐标
        color_pixel = rightmost_color_pixel
        if color_pixel:
            print(f"最右侧彩色像素: {color_pixel}")
            # 绘制黑色像素与彩色像素的连线
            cv2.line(img, found_pixel, color_pixel, (255, 0, 0), 2)  # 蓝色线条
            print("已绘制黑色像素与彩色像素的连线")

            # 计算连线的方向和长度
            dx = color_pixel[0] - found_pixel[0]
            dy = color_pixel[1] - found_pixel[1]

            # 计算单位方向向量
            length = ((dx ** 2) + (dy ** 2)) ** 0.5
            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length

                # 计算垂直方向的单位向量（逆时针旋转90度）
                perp_dx = -unit_dy
                perp_dy = unit_dx

                # 计算矩形的四个顶点
                # 起点向左上方和右下方外扩8个像素
                p1 = (int(found_pixel[0] + perp_dx * 8), int(found_pixel[1] + perp_dy * 8))
                p2 = (int(found_pixel[0] - perp_dx * 8), int(found_pixel[1] - perp_dy * 8))
                # 终点向左上方和右下方外扩8个像素
                p3 = (int(color_pixel[0] - perp_dx * 8), int(color_pixel[1] - perp_dy * 8))
                p4 = (int(color_pixel[0] + perp_dx * 8), int(color_pixel[1] + perp_dy * 8))

                # 绘制矩形
                rectangle = [p1, p2, p3, p4]

                # cv2.polylines(img, [np.array(rectangle)], isClosed=True, color=(0, 255, 255), thickness=2)  # 黄色线条
                # print("已绘制上下外扩8个像素的矩形")

                # 从黑色像素开始，沿连线寻找第一个白色像素
                def get_line_points(p1, p2):
                    """生成两点之间的直线上的所有点"""
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

                # 生成连线上的所有点
                line_points = get_line_points(found_pixel, color_pixel)

                # 从黑色像素开始，寻找第一个白色像素
                first_white_pixel = None
                for point in line_points[1:]:  # 跳过第一个点（黑色像素）
                    x, y = point
                    # 检查是否在图像范围内
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        # 检查是否是白色像素（灰度值大于200）
                        if gray[y, x] > 200:
                            first_white_pixel = point
                            print(f"找到第一个白色像素: {first_white_pixel}")
                            break

                # 标记第一个白色像素
                if first_white_pixel:
                    cv2.circle(img, first_white_pixel, 2, (0, 255, 0), -1)  # 绿色圆圈
                    print("已标记第一个白色像素")

                    # 计算从color_pixel到黑色像素的方向向量
                    dx = found_pixel[0] - color_pixel[0]
                    dy = found_pixel[1] - color_pixel[1]

                    # 计算单位方向向量
                    length = ((dx ** 2) + (dy ** 2)) ** 0.5
                    if length > 0:
                        unit_dx = dx / length
                        unit_dy = dy / length

                        # 矩形的结束点为黑色像素
                        rect_end = found_pixel

                        # 计算垂直方向的单位向量（逆时针旋转90度）
                        perp_dx = -unit_dy
                        perp_dy = unit_dx

                        # 创建第一个矩形（white_rectangle），expand_px=25
                        expand_px = 25
                        # 计算矩形的四个顶点
                        # color_pixel点向左上方和右下方外扩expand_px个像素
                        rect_p1 = (int(color_pixel[0] + perp_dx * expand_px), int(color_pixel[1] + perp_dy * expand_px))
                        rect_p2 = (int(color_pixel[0] - perp_dx * expand_px), int(color_pixel[1] - perp_dy * expand_px))
                        # 结束点（黑色像素）向左上方和右下方外扩expand_px个像素
                        rect_p3 = (int(rect_end[0] - perp_dx * expand_px), int(rect_end[1] - perp_dy * expand_px))
                        rect_p4 = (int(rect_end[0] + perp_dx * expand_px), int(rect_end[1] + perp_dy * expand_px))

                        # 绘制矩形
                        white_rectangle = [rect_p1, rect_p2, rect_p3, rect_p4]
                        cv2.polylines(img, [np.array(white_rectangle)], isClosed=True, color=(255, 255, 0),
                                      thickness=1)  # 青色线条
                        print("已绘制从color_pixel到黑色像素，上下外扩25个像素的矩形")

                        # 创建第二个矩形（rectangle2），expand_px=35
                        expand_px2 = 35
                        # 计算矩形2的四个顶点
                        rect2_p1 = (
                        int(color_pixel[0] + perp_dx * expand_px2), int(color_pixel[1] + perp_dy * expand_px2))
                        rect2_p2 = (
                        int(color_pixel[0] - perp_dx * expand_px2), int(color_pixel[1] - perp_dy * expand_px2))
                        rect2_p3 = (int(rect_end[0] - perp_dx * expand_px2), int(rect_end[1] - perp_dy * expand_px2))
                        rect2_p4 = (int(rect_end[0] + perp_dx * expand_px2), int(rect_end[1] + perp_dy * expand_px2))

                        # 绘制矩形2
                        rectangle2 = [rect2_p1, rect2_p2, rect2_p3, rect2_p4]
                        cv2.polylines(img, [np.array(rectangle2)], isClosed=True, color=(0, 0, 255),
                                      thickness=1)  # 红色线条
                        print("已绘制从color_pixel到黑色像素，上下外扩35个像素的矩形2")

                        # 计算矩形2和white_rectangle的不相交区域
                        # 创建两个矩形的掩码
                        mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask1, [np.array(white_rectangle)], 255)

                        mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask2, [np.array(rectangle2)], 255)

                        # 计算不相交区域的掩码（mask2 - mask1）
                        disjoint_mask = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))

                        # 计算黑色像素与彩色像素的连线，用于划分区域
                        # 生成连线上的所有点
                        def get_line_points(p1, p2):
                            """生成两点之间的直线上的所有点"""
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

                        # 生成连线上的所有点
                        line_points = get_line_points(color_pixel, found_pixel)

                        # 创建连线掩码
                        line_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        for point in line_points:
                            if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
                                line_mask[point[1], point[0]] = 255

                        # 计算垂直方向的单位向量（用于区分两侧）
                        perp_dx = -unit_dy
                        perp_dy = unit_dx

                        # 创建左侧和右侧的掩码
                        left_disjoint_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        right_disjoint_mask = np.zeros(img.shape[:2], dtype=np.uint8)

                        # 遍历不相交区域的所有像素
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                if disjoint_mask[y, x] == 255:
                                    # 计算像素到连线的垂直距离和方向
                                    # 找到连线上最近的点
                                    min_dist = float('inf')
                                    closest_point = None
                                    for point in line_points:
                                        dist = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_point = point

                                    if closest_point:
                                        # 向量从最近点到当前像素
                                        pixel_vec = (x - closest_point[0], y - closest_point[1])
                                        # 计算与垂直方向的点积
                                        dot_product = pixel_vec[0] * perp_dx + pixel_vec[1] * perp_dy

                                        # 根据点积的符号判断是左侧还是右侧
                                        if dot_product > 0:
                                            left_disjoint_mask[y, x] = 255
                                        else:
                                            right_disjoint_mask[y, x] = 255

                        # 统计左侧不相交区域的黑色像素数量
                        _, black_mask = cv2.threshold(gray0, 100, 255, cv2.THRESH_BINARY_INV)
                        left_black_pixels = cv2.countNonZero(cv2.bitwise_and(left_disjoint_mask, black_mask))
                        # 统计右侧不相交区域的黑色像素数量
                        right_black_pixels = cv2.countNonZero(cv2.bitwise_and(right_disjoint_mask, black_mask))

                        print(f"左侧不相交区域的黑色像素数量: {left_black_pixels}")
                        print(f"右侧不相交区域的黑色像素数量: {right_black_pixels}")

                        # 可视化黑色像素
                        # 创建一个彩色图像用于标记黑色像素
                        black_visualization = img.copy()

                        # 遍历不相交区域的所有像素
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                if disjoint_mask[y, x] == 255 and black_mask[y, x] == 255:
                                    # 在左侧的黑色像素用蓝色标记
                                    if left_disjoint_mask[y, x] == 255:
                                        black_visualization[y, x] = (255, 0, 0)  # 蓝色
                                    # 在右侧的黑色像素用绿色标记
                                    elif right_disjoint_mask[y, x] == 255:
                                        black_visualization[y, x] = (0, 255, 0)  # 绿色
                        # 保存black_mask
                        if not os.path.exists('output'):
                            os.makedirs('output')
                        filename = os.path.basename(json_path).replace('_res.json', '')
                        mask_output_path = os.path.join('output', f'{filename}_black_mask1.png')
                        cv2.imwrite(mask_output_path, black_mask)
                        # 保存黑色像素可视化结果
                        black_output_path = os.path.join('output', f'{filename}_black_pixels.png')
                        cv2.imwrite(black_output_path, black_visualization)
                        print(f"已保存黑色像素可视化结果到: {black_output_path}")

                        # 更新JSON文件，添加黑色像素数量
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data['left_black_pixels'] = int(left_black_pixels)
                        data['right_black_pixels'] = int(right_black_pixels)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        print(f"已将黑色像素数量更新到: {json_path}")

    return img, found_pixel


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


def process_leftmost_pixels(img, mask_red, mask_yellow, mask_green, red_center, yellow_center, green_center):
    """
    处理最左端像素

    参数:
        img: 原始图像
        mask_red: 红色掩码
        mask_yellow: 黄色掩码
        mask_green: 绿色掩码
        red_center: 红色中心坐标
        yellow_center: 黄色中心坐标
        green_center: 绿色中心坐标

    返回:
        tuple: (可视化图像, 颜色线信息)
    """
    # 计算各颜色最左端像素坐标
    red_leftmost = calculate_leftmost(mask_red)
    yellow_leftmost = calculate_leftmost(mask_yellow)
    green_leftmost = calculate_leftmost(mask_green)

    # 打印最左端像素坐标
    if red_leftmost:
        print(f"红色像素最左端坐标: {red_leftmost}")
    if yellow_leftmost:
        print(f"黄色像素最左端坐标: {yellow_leftmost}")
    if green_leftmost:
        print(f"绿色像素最左端坐标: {green_leftmost}")

    # 可视化中心坐标
    vis_img = img.copy()

    # 绘制中心坐标
    if red_center:
        cv2.circle(vis_img, red_center, 2, (255, 0, 0), -1)  # 蓝色圆圈
    if yellow_center:
        cv2.circle(vis_img, yellow_center, 2, (255, 0, 0), -1)  # 蓝色圆圈
    if green_center:
        cv2.circle(vis_img, green_center, 2, (255, 0, 0), -1)  # 蓝色圆圈

    # 绘制最左端像素并连线
    if red_leftmost:
        cv2.circle(vis_img, red_leftmost, 2, (255, 0, 0), -1)  # 红色圆圈
    if yellow_leftmost:
        cv2.circle(vis_img, yellow_leftmost, 2, (255, 0, 0), -1)  # 黄色圆圈

    # 连接最左侧的颜色像素
    color_line_info = None
    leftmost_points = []
    if red_leftmost:
        leftmost_points.append(red_leftmost)
    if yellow_leftmost:
        leftmost_points.append(yellow_leftmost)
    if green_leftmost:
        leftmost_points.append(green_leftmost)

    if leftmost_points:
        # 找到最左侧的x坐标
        min_x = min(p[0] for p in leftmost_points)

        # 获取所有在最左侧的像素点 (允许5像素误差)
        far_left_points = [p for p in leftmost_points if abs(p[0] - min_x) <= 5]

        if len(far_left_points) == 1:
            # 如果只有一个最左侧颜色，以该像素为中心创建一条短垂直线
            center_point = far_left_points[0]
            # 创建一个20像素高的垂直线
            extended_start = (center_point[0], center_point[1] - 10)
            extended_end = (center_point[0], center_point[1] + 10)
            print(f"左侧只有一种颜色，扩展像素: {center_point}")

            # 绘制线
            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

        elif len(far_left_points) >= 2:
            # 如果有两种或以上颜色，使用原始方法
            # 按y坐标排序找到最高和最低点
            far_left_points.sort(key=lambda p: p[1])
            point1 = far_left_points[0]
            point2 = far_left_points[-1]
            print(f"左侧有两种或以上颜色，使用原始方法连接: {point1} 和 {point2}")

            # 原始的线扩展逻辑
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = ((dx ** 2) + (dy ** 2)) ** 0.5

            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length
            else:
                unit_dx, unit_dy = 0, 0

            # 外扩长度为线长的一半
            extend_length = length / 2

            extended_start = (int(point1[0] - unit_dx * extend_length),
                              int(point1[1] - unit_dy * extend_length))
            extended_end = (int(point2[0] + unit_dx * extend_length),
                            int(point2[1] + unit_dy * extend_length))

            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

    return vis_img, color_line_info


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


def process_rightmost_pixels(img, mask_red, mask_yellow, mask_green, red_center, yellow_center, green_center):
    """
    处理最右端像素

    参数:
        img: 原始图像
        mask_red: 红色掩码
        mask_yellow: 黄色掩码
        mask_green: 绿色掩码
        red_center: 红色中心坐标
        yellow_center: 黄色中心坐标
        green_center: 绿色中心坐标

    返回:
        tuple: (可视化图像, 颜色线信息)
    """
    # 计算各颜色最右端像素坐标
    red_rightmost = calculate_rightmost(mask_red)
    yellow_rightmost = calculate_rightmost(mask_yellow)
    green_rightmost = calculate_rightmost(mask_green)

    # 打印最右端像素坐标
    if red_rightmost:
        print(f"红色像素最右端坐标: {red_rightmost}")
    if yellow_rightmost:
        print(f"黄色像素最右端坐标: {yellow_rightmost}")
    if green_rightmost:
        print(f"绿色像素最右端坐标: {green_rightmost}")

    # 可视化中心坐标
    vis_img = img.copy()

    # 绘制中心坐标
    if red_center:
        cv2.circle(vis_img, red_center, 2, (255, 0, 0), -1)  # 蓝色圆圈
    if yellow_center:
        cv2.circle(vis_img, yellow_center, 2, (255, 0, 0), -1)  # 蓝色圆圈
    if green_center:
        cv2.circle(vis_img, green_center, 2, (255, 0, 0), -1)  # 蓝色圆圈

    # 绘制最右端像素并连线
    if red_rightmost:
        cv2.circle(vis_img, red_rightmost, 2, (255, 0, 0), -1)  # 红色圆圈
    if yellow_rightmost:
        cv2.circle(vis_img, yellow_rightmost, 2, (255, 0, 0), -1)  # 黄色圆圈

    # 连接最右侧的颜色像素
    color_line_info = None
    rightmost_points = []
    if red_rightmost:
        rightmost_points.append(red_rightmost)
    if yellow_rightmost:
        rightmost_points.append(yellow_rightmost)
    if green_rightmost:
        rightmost_points.append(green_rightmost)

    if rightmost_points:
        # 找到最右侧的x坐标
        max_x = max(p[0] for p in rightmost_points)

        # 获取所有在最右侧的像素点 (允许5像素误差)
        far_right_points = [p for p in rightmost_points if abs(p[0] - max_x) <= 5]

        if len(far_right_points) == 1:
            # 如果只有一个最右侧颜色，以该像素为中心创建一条短垂直线
            center_point = far_right_points[0]
            # 创建一个40像素高的垂直线
            extended_start = (center_point[0], center_point[1] - 25)
            extended_end = (center_point[0], center_point[1] + 25)
            print(f"右侧只有一种颜色，扩展像素: {center_point}")

            # 绘制线
            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

        elif len(far_right_points) >= 2:
            # 如果有两种或以上颜色，使用原始方法
            # 按y坐标排序找到最高和最低点
            far_right_points.sort(key=lambda p: p[1])
            point1 = far_right_points[0]
            point2 = far_right_points[-1]
            print(f"右侧有两种或以上颜色，使用原始方法连接: {point1} 和 {point2}")

            # 原始的线扩展逻辑
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = ((dx ** 2) + (dy ** 2)) ** 0.5

            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length
            else:
                unit_dx, unit_dy = 0, 0

            # 外扩长度为线长的一半
            extend_length = length / 2

            extended_start = (int(point1[0] - unit_dx * extend_length),
                              int(point1[1] - unit_dy * extend_length))
            extended_end = (int(point2[0] + unit_dx * extend_length),
                            int(point2[1] + unit_dy * extend_length))

            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)

    return vis_img, color_line_info


def save_visualization(img, image_path):
    """
    保存可视化结果

    参数:
        img: 可视化图像
        image_path: 原始图像路径
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    # 提取文件名
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    vis_output_path = os.path.join('output', f'{name}_color_centers{ext}')

    cv2.imwrite(vis_output_path, img)
    print(f"颜色中心可视化结果已保存到: {vis_output_path}")


if __name__ == "__main__":
    # 输入图像路径
    input_image = r"D:\jjr\ocr_work\xbd\data\image.png"

    # 输入目标字符
    target_char = "SII"
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
