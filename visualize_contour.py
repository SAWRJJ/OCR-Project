import cv2
import numpy as np

image_path = r"d:\work\ocr+Transformer\output\t3\micro_img\micro_0002_SI.jpg"
output_path = r"d:\work\ocr+Transformer\output\t3\micro_img\micro_0002_SI_vis.jpg"

image = cv2.imread(image_path)

points = np.array([[300, 40], [378, 40], [378, 116], [300, 116]], dtype=np.int32)

cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

x, y, w, h = cv2.boundingRect(points)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imwrite(output_path, image)
print(f"已保存可视化图像到: {output_path}")

cv2.imshow("Contour Visualization", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
