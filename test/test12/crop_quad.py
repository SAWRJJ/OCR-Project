from PIL import Image, ImageDraw
import numpy as np

image_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test12/micro_0090_X.jpg"
output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test12/micro_0090_X_cropped.jpg"

img = Image.open(image_path)
img_array = np.array(img)

points = np.array([[79, 40], [128, 53], [117, 99], [68, 87]], dtype=np.int32)

mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
cv2_points = points.reshape((-1, 1, 2))
import cv2
cv2.fillPoly(mask, [cv2_points], 255)

masked_image = cv2.bitwise_and(img_array, img_array, mask=mask)

x, y, w, h = cv2.boundingRect(cv2_points)
cropped = img_array[y:y+h, x:x+w]
mask_cropped = mask[y:y+h, x:x+w]

result = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)

result_pil = Image.fromarray(result)
result_pil.save(output_path)
print(f"Cropped image saved to {output_path}")