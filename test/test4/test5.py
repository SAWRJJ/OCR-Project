import json
import cv2
import os
import math
import numpy as np
import sys
sys.path.insert(0, r'/')
from ocr.ocr_engine import OCREngine

def find_first_non_white_column_along_tilt(poly, gray_img, angle):
    """
    利用文本框的横向倾斜角度，从图片最左侧开始沿着倾斜方向扫描
    找到第一列（有非白色像素的列），然后向右外扩像素
    """
    poly = np.array(poly, dtype=np.float64)

    top_left = poly[0]
    top_right = poly[1]

    tilt_angle = angle

    print(f"文本框倾斜角度: {math.degrees(tilt_angle):.2f} 度")

    poly_y_center = (np.min(poly[:, 1]) + np.max(poly[:, 1])) / 2
    poly_x_min = np.min(poly[:, 0])

    img_height = gray_img.shape[0]
    y_min = 15
    y_max = img_height - 15

    first_non_white_col = None
    non_white_pixels = []

    start_x = 0
    start_y = poly_y_center

    step_size = 1
    num_steps = max(gray_img.shape[1] * 2, 3000)

    for step in range(num_steps):
        x = start_x + step * math.cos(tilt_angle) * step_size
        y = start_y + step * math.sin(tilt_angle) * step_size

        check_x = int(x)

        if check_x < 0 or check_x >= gray_img.shape[1]:
            continue

        col_has_non_white = False

        for check_y in range(int(y_min), int(y_max) + 1):
            if 0 <= check_y < gray_img.shape[1]:
                pixel_value = gray_img[check_y, check_x]
                if pixel_value < 128:
                    col_has_non_white = True

        if first_non_white_col is None:
            if col_has_non_white:
                first_non_white_col = check_x
                for check_y in range(int(y_min), int(y_max) + 1):
                    if 0 <= check_y < gray_img.shape[1]:
                        pixel_value = gray_img[check_y, check_x]
                        if pixel_value < 128:
                            non_white_pixels.append((check_x, check_y))
                print(f"找到第一列有非白色像素: x={check_x}, 共{len(non_white_pixels)}个非白像素")
                break

        if x > poly_x_min + 50:
            break

    expand_col = None
    if first_non_white_col is not None:
        expand_col = int(first_non_white_col + 70 * math.cos(tilt_angle))
        print(f"向右外扩像素: x={expand_col}")

    return first_non_white_col, expand_col, non_white_pixels


def calculate_tilt_angle(poly):
    """
    计算文本框水平方向的倾斜角度（文本框上下两边的倾斜程度）

    Args:
        poly: 文本框多边形坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        angle: 倾斜角度（度），正值表示右下倾斜，负值表示左下倾斜
    """
    poly = np.array(poly, dtype=np.float64)

    top_left = poly[0]
    top_right = poly[1]
    bottom_right = poly[2]
    bottom_left = poly[3]

    top_y = (top_left[1] + top_right[1]) / 2
    bottom_y = (bottom_left[1] + bottom_right[1]) / 2

    left_x = (top_left[0] + bottom_left[0]) / 2
    right_x = (top_right[0] + bottom_right[0]) / 2

    dy = top_right[1] - top_left[1]
    dx = top_right[0] - top_left[0]

    angle = math.degrees(math.atan2(dy, dx))

    return angle, top_left, top_right


def perform_ocr(image_path):
    """
    对图片进行OCR识别
    """
    try:
        ocr_engine = OCREngine()
        results = ocr_engine.predict(image_path, output_dir=r'/test/test4/output/ocr', adjust_type=True)
        return results
    except Exception as e:
        print(f"OCR识别失败: {e}")
        return []


def visualize_result(image_path, first_black_col, expand_col, non_white_pixels, output_path):
    """
    可视化结果：画出非白色像素位置和外扩100像素的列
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    if first_black_col is not None:
        cv2.line(img, (first_black_col, 15), (first_black_col, img.shape[0] - 15), (0, 255, 255), 2)

    if expand_col is not None:
        cv2.line(img, (expand_col, 15), (expand_col, img.shape[0] - 15), (255, 0, 0), 2)

    for px, py in non_white_pixels:
        cv2.circle(img, (px, py), 1, (0, 0, 255), -1)

    if first_black_col is not None and expand_col is not None:
        left = min(first_black_col, expand_col)
        right = max(first_black_col, expand_col)
        cropped = img[:, left:right+1]
        cropped_path = output_path.replace('.jpg', '_cropped.jpg')
        cv2.imwrite(cropped_path, cropped)
        print(f"裁剪图片已保存到: {cropped_path}")

        print("\n=== OCR识别结果 ===")
        ocr_results = perform_ocr(cropped_path)
        for result in ocr_results:
            if len(result) >= 2:

                text = result['rec_texts'][0]
                conf = result['rec_scores'][0]

                print(f"文本: {text}, 置信度: {conf:.2f}")

    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")


def main():
    json_path = r'/test/test4/micro_0018_0s3.json'
    image_path = r'/test/test4/micro_0018_0s3.jpg'
    output_path = r'/test/test4/output/black_column_visualization.jpg'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    micro_poly = data['micro_poly']

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    gray_output_path = output_path.replace('.jpg', '_gray.jpg')
    cv2.imwrite(gray_output_path, binary)
    print(f"二值图已保存到: {gray_output_path}")

    angle, top_left, top_right = calculate_tilt_angle(micro_poly)

    first_non_white_col, expand_col, non_white_pixels = find_first_non_white_column_along_tilt(micro_poly, binary, angle)

    print(f"\n=== 结果 ===")
    if first_non_white_col is not None:
        print(f"第一列有非白色像素: x={first_non_white_col}")
    if expand_col is not None:
        print(f"向右外扩像素: x={expand_col}")
    else:
        print("未找到黑色像素")

    print(f"文本框倾斜角度: {angle:.2f} 度")
    print(f"文本框左上: ({top_left[0]:.2f}, {top_left[1]:.2f})")
    print(f"文本框右上: ({top_right[0]:.2f}, {top_right[1]:.2f})")

    first_black_col = first_non_white_col
    first_black_row = 15

    visualize_result(image_path, first_black_col, expand_col, non_white_pixels, output_path)


if __name__ == '__main__':
    main()
