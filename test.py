import time

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import os
import json


def resize_image_to_width(image_path, target_width=4000):
    img = Image.open(image_path)
    original_width, original_height = img.size
    if original_width < target_width:
        return np.array(img)
    ratio = target_width / original_width
    new_height = int(original_height * ratio)
    resized_img = img.resize((target_width, new_height), Image.LANCZOS)
    return np.array(resized_img)


# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu")

image_path = r"D:\work\ocr+Transformer\test1.jpg"
original_img = Image.open(image_path)
original_width, original_height = original_img.size

t1 = time.time()
result = ocr.predict(image_path)
t2 = time.time()
print(t2 - t1)

for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

json_path = os.path.join(os.path.dirname(image_path), "output",
                         os.path.splitext(os.path.basename(image_path))[0] + "_res.json")
output_dir = f"output/micro_img/{os.path.splitext(os.path.basename(image_path))[0]}"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    ocr_data = json.load(f)

dt_polys = ocr_data.get('dt_polys', [])
rec_texts = ocr_data.get('rec_texts', [])

for idx, (points, text) in enumerate(zip(dt_polys, rec_texts)):
    text_upper = text.upper()
    if any(char in text_upper for char in
           ['S', 'D', 'X']) and 'Y' not in text_upper and 'G' not in text_upper and text_upper != "D":
        box = np.array(points)
        x_coords = box[:, 0]
        y_coords = box[:, 1]

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        x_expand = 70 if 'D' in text_upper else 220
        x_min = max(0, x_min - x_expand)
        y_min = max(0, y_min - 25)
        x_max = min(original_width, x_max + x_expand)
        y_max = min(original_height, y_max + 25)

        cropped_img = original_img.crop((x_min, y_min, x_max, y_max))
        safe_text = "".join(c if c.isalnum() else "_" for c in text)
        save_path = os.path.join(output_dir, f"micro_{idx}_{safe_text}.jpg")
        cropped_img.save(save_path)
        print(f"已保存: {save_path}")
