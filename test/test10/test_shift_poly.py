import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/Users/saw/WorkSpace/work/OCR-Project')
from ocr.X_detect import shift_poly_along_angle

filename = "micro_0049_2300_1XVII"
image_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.jpg'
json_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.json'

def calculate_textbox_angle(poly):
    poly = np.array(poly, dtype=np.float32)
    if poly.shape[0] != 4:
        return 0.0, None
    p0, p1, p2, p3 = poly
    top_mid = (p0 + p1) / 2
    bottom_mid = (p2 + p3) / 2
    dx = bottom_mid[0] - top_mid[0]
    dy = bottom_mid[1] - top_mid[1]
    angle = np.arctan2(dx, dy)
    return angle, (tuple(top_mid.astype(int)), tuple(bottom_mid.astype(int)))

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

poly = np.array(data['micro_poly'], dtype=np.int32)
poly = np.array([[0, 61], [270, 40], [276, 114], [6, 135]], dtype=np.int32)
text = data["text"]
text_list = list(text)
index0 = text_list.index("X")
print(f"index0:{index0}")
img = cv2.imread(image_path)

textbox_angle, _ = calculate_textbox_angle(poly)
print(f"文本框倾斜角度: {np.degrees(textbox_angle):.2f} 度")

left_edge_start, left_edge_end = poly[0], poly[3]
right_edge_start, right_edge_end = poly[1], poly[2]

left_x_avg = (left_edge_start[0] + left_edge_end[0]) / 2
left_y_avg = (left_edge_start[1] + left_edge_end[1]) / 2
right_x_avg = (right_edge_start[0] + right_edge_end[0]) / 2
right_y_avg = (right_edge_start[1] + right_edge_end[1]) / 2

dx = right_x_avg - left_x_avg
dy = right_y_avg - left_y_avg
euclidean_distance = np.sqrt(dx**2 + dy**2)

print(f"左侧边中心点: ({left_x_avg:.2f}, {left_y_avg:.2f})")
print(f"右侧边中心点: ({right_x_avg:.2f}, {right_y_avg:.2f})")
print(f"水平距离(dx): {dx:.2f}")
print(f"垂直距离(dy): {dy:.2f}")
print(f"勾股定理距离: {euclidean_distance:.2f} 像素")

output_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}_poly_output.png'

new_poly, shift_line_start, shift_line_end = shift_poly_along_angle(
    poly=poly,
    angle=textbox_angle,
    shift_distance=32*6,
    debug_img=img,
    output_path=output_path
)

print(f"\n原始 poly: {poly.tolist()}")
print(f"新 poly: {new_poly.tolist()}")
print(f"平移线起点: {shift_line_start}")
print(f"平移线终点: {shift_line_end}")