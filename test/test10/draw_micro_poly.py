import cv2
import numpy as np
import json
from ocr.LW_detect import detect_colors, calculate_textbox_angle
from ocr.utils import calculate_angle_to_horizontal
img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test10/micro_0296_SL4082600.jpg"
json_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test10/micro_0014_FXII_K.json"

img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

micro_poly = np.array(data['micro_poly'], dtype=np.int32)
text = data['text']
print(text)
print(len(text))
# [[[17  5], [79 39], [61 60], [ 0 26]], [[ 9 40], [58 66], [41 87], [ 0 61]]]
# ploy = [[17,5], [79,39], [61,60], [ 0,26]]
# poly1 = [[ 9,40], [58,66], [41,87], [ 0,61]]
#
# ploy_np = np.array(ploy)
# poly1_np = np.array(poly1)
# center_ploy = (ploy_np[:, 0].mean(), ploy_np[:, 1].mean())
# center_poly1 = (poly1_np[:, 0].mean(), poly1_np[:, 1].mean())
# print(f"ploy 中心点: {center_ploy}")
# print(f"poly1 中心点: {center_poly1}")
# angle = calculate_angle_to_horizontal(center_ploy,center_poly1)
# textbox_angle, _ = calculate_textbox_angle(ploy)
# textbox_angle1, _ = calculate_textbox_angle(poly1)
# print("===============")
# print(np.degrees(textbox_angle))
# print(np.degrees(textbox_angle1))
# print(np.degrees(angle))
# print("===============")
micro_poly = np.array([[284, 129], [317, 162], [341, 186], [308, 153]], dtype=np.int32)
# micro_poly1 = np.array([[300, 40], [300, 67], [419, 70], [419, 43]], dtype=np.int32)
# cv2.polylines(img, [micro_poly1], isClosed=True, color=(0, 255, 255), thickness=2)
h, w = img.shape[:2]
print(f"图片尺寸: {w}x{h}")

x_coords = [p[0] for p in micro_poly]
y_coords = [p[1] for p in micro_poly]
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

rect_width = max_x - min_x
rect_height = max_y - min_y

print(f"micro_poly 顶点: {micro_poly.tolist()}")
print(f"X坐标范围: [{min_x}, {max_x}], 宽度: {rect_width}")
print(f"Y坐标范围: [{min_y}, {max_y}], 高度: {rect_height}")

output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test10/micro_0110_2300_1X5_with_poly.jpg"

cv2.polylines(img, [micro_poly], isClosed=True, color=(0, 255, 0), thickness=2)


# cv2.circle(img, (int(center_ploy[0]), int(center_ploy[1])), 5, (0, 0, 255), -1)
# cv2.putText(img, f"C1({int(center_ploy[0])},{int(center_ploy[1])})", (int(center_ploy[0]) + 10, int(center_ploy[1]) - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
# cv2.circle(img, (int(center_poly1[0]), int(center_poly1[1])), 5, (255, 0, 0), -1)
# cv2.putText(img, f"C2({int(center_poly1[0])},{int(center_poly1[1])})", (int(center_poly1[0]) + 10, int(center_poly1[1]) - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# ((332, 151), (323, 194))
extra_point = (332, 151)
cv2.circle(img, extra_point, 5, (255, 0, 0), -1)
cv2.putText(img, f"P({extra_point[0]},{extra_point[1]})", (extra_point[0] + 10, extra_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

extra_point1 = (323, 194)
dist = np.sqrt((extra_point[0] - extra_point1[0])**2 + (extra_point[1] - extra_point1[1])**2)
print(f"两个extra_point之间的距离: {dist:.2f}")
cv2.circle(img, extra_point1, 5, (255, 0, 255), -1)
cv2.putText(img, f"P({extra_point1[0]},{extra_point1[1]})", (extra_point1[0] + 10, extra_point1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

center_x = (min_x + max_x) // 2
center_y = (min_y + max_y) // 2
cv2.putText(img, f"W:{rect_width}", (center_x - 30, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(img, f"H:{rect_height}", (max_x + 5, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite(output_path, img)
print(f"已保存带标注的图片: {output_path}")