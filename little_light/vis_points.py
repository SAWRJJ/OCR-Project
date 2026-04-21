import cv2
import numpy as np

img_path = r"d:\work\ocr+Transformer\little_light\micro_0079_S8.jpg"
img = cv2.imread(img_path)

point1 = (303, 165)
point2 = (296, 78)

cv2.circle(img, point1, 5, (0, 0, 255), -1)
cv2.circle(img, point2, 5, (0, 255, 0), -1)

cv2.putText(img, f"P1({point1[0]}, {point1[1]})", (point1[0] + 10, point1[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(img, f"P2({point2[0]}, {point2[1]})", (point2[0] + 10, point2[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.line(img, point1, point2, (255, 0, 255), 2)

output_path = r"d:\work\ocr+Transformer\little_light\micro_0079_S8_vis.jpg"
cv2.imwrite(output_path, img)
print(f"已保存可视化图像到: {output_path}")
