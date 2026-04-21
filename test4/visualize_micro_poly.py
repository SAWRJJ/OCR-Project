import cv2
import json
import numpy as np
import math

try:
    from ocr.ocr_engine import OCREngine
    OCREngine
except ImportError:
    OCREngine = None
    print("警告: 无法导入 OCREngine，跳过 OCR 识别")
def calculate_horizontal_tilt_angle(poly):
    """
    计算文本框的水平倾斜角度
    poly: 四边形四个点的坐标列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
          顺序为：左上、右上、右下、左下
    返回：倾斜角度（度），正值表示右倾斜，负值表示左倾斜
    """
    poly = np.array(poly)
    top_left = poly[0]
    top_right = poly[1]
    bottom_right = poly[2]
    bottom_left = poly[3]

    top_angle = math.degrees(math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
    bottom_angle = math.degrees(math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0]))

    avg_angle = (top_angle + bottom_angle) / 2
    return round(avg_angle, 2)


def expand_poly(poly, expand_x=10, expand_y=5, angle=0):
    """
    扩展多边形
    poly: 四边形四个点的坐标列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
          顺序为：左上、右上、右下、左下
    expand_x: 水平方向右侧扩展像素数（左侧不变）
    expand_y: 垂直方向上下扩展像素数
    angle: 倾斜角度（度），正值右倾斜，负值左倾斜
    返回：扩展后的多边形
    """
    poly = np.array(poly, dtype=np.float32)
    angle_rad = math.radians(angle)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rotate_point(point, center, cos_a, sin_a):
        x, y = point[0] - center[0], point[1] - center[1]
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        return [new_x + center[0], new_y + center[1]]

    center_x = (poly[0][0] + poly[2][0]) / 2
    center_y = (poly[0][1] + poly[2][1]) / 2
    center = [center_x, center_y]

    rotated_poly = []
    for p in poly:
        rotated_poly.append(rotate_point(p, center, cos_a, -sin_a))

    rotated_poly = np.array(rotated_poly, dtype=np.float32)

    expanded = rotated_poly.copy()
    expanded[0][0] = rotated_poly[0][0]
    expanded[0][1] -= expand_y
    expanded[1][0] = rotated_poly[1][0] + expand_x
    expanded[1][1] -= expand_y
    expanded[2][0] = rotated_poly[2][0] + expand_x
    expanded[2][1] = rotated_poly[2][1] + expand_y
    expanded[3][0] = rotated_poly[3][0]
    expanded[3][1] = rotated_poly[3][1] + expand_y

    final_poly = []
    for p in expanded:
        final_poly.append(rotate_point(p, center, cos_a, sin_a))

    return [[int(p[0]), int(p[1])] for p in final_poly]


img_path = r"d:\work\ocr+Transformer\test4\micro_0004_S.jpg"
json_path = r"d:\work\ocr+Transformer\test4\micro_0004_S.json"
output_path = r"d:\work\ocr+Transformer\test4\micro_0004_S_visualized.jpg"

img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图像: {img_path}")
    exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

micro_poly = data['micro_poly']
print(f"原始poly: {micro_poly}")

tilt_angle = calculate_horizontal_tilt_angle(micro_poly)
print(f"水平倾斜角度: {tilt_angle}°")

expanded_poly = expand_poly(micro_poly, expand_x=30, expand_y=6, angle=tilt_angle)
print(f"扩展后poly: {expanded_poly}")

poly_array = np.array(micro_poly, dtype=np.int32)
# cv2.polylines(img, [poly_array], True, (0, 0, 255), 2)

expanded_array = np.array(expanded_poly, dtype=np.int32)


x_min = max(0, int(min(p[0] for p in expanded_poly)))
x_max = min(img.shape[1], int(max(p[0] for p in expanded_poly)))
y_min = max(0, int(min(p[1] for p in expanded_poly)))
y_max = min(img.shape[0], int(max(p[1] for p in expanded_poly)))
cropped = img[y_min:y_max, x_min:x_max]
print(f"裁切区域: x={x_min}:{x_max}, y={y_min}:{y_max}, 裁切后尺寸: {cropped.shape}")

cropped_path = r"d:\work\ocr+Transformer\test4\micro_0004_S_cropped.jpg"
cv2.imwrite(cropped_path, cropped)
print(f"裁切图像已保存: {cropped_path}")

ocr_engine = OCREngine()
results = ocr_engine.ocr.predict(cropped)
cv2.polylines(img, [expanded_array], True, (255, 0, 0), 2)
rec_text = ''.join(results[0]['rec_texts']).replace(' ', '')
print(f"OCR结果: {rec_text}")
for i, point in enumerate(micro_poly):
    cv2.circle(img, tuple(point), 1, (0, 255, 0), -1)

cv2.putText(img, f"Angle: {tilt_angle}°", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(img, f"Angle: {tilt_angle}°", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imwrite(output_path, img)
print(f"可视化结果已保存: {output_path}")
