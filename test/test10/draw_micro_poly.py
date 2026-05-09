import cv2
import numpy as np
import json

img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test10/micro_0018_D.jpg"
json_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test10/micro_0018_D.json"

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
micro_poly = np.array([[221, 164], [334, 62], [394, 124], [280, 226]], dtype=np.int32)
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

extra_point = (100, 72)
cv2.circle(img, extra_point, 5, (255, 0, 0), -1)
cv2.putText(img, f"P({extra_point[0]},{extra_point[1]})", (extra_point[0] + 10, extra_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

center_x = (min_x + max_x) // 2
center_y = (min_y + max_y) // 2
cv2.putText(img, f"W:{rect_width}", (center_x - 30, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(img, f"H:{rect_height}", (max_x + 5, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite(output_path, img)
print(f"已保存带标注的图片: {output_path}")