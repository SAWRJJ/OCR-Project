
from paddleocr import PaddleOCR

# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False) # 文本检测+文本识别
# ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # 文本图像预处理+文本检测+方向分类+文本识别
# ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # 文本检测+文本行方向分类+文本识别
import os
import time

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False) # 更换 PP-OCRv5_mobile 模型

image_dir = r"test/test2/micro_img"
output_dir = r"test/test2/outpit"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_start_time = time.time()
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

print(f"Found {len(image_files)} images to process.")

for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    print(f"Processing {filename}...")
    
    start_time = time.time()
    result = ocr.predict(image_path)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Time taken for {filename}: {elapsed_time:.4f} seconds")
    
    for res in result:
        res.print()
        res.save_to_img(output_dir)
        res.save_to_json(output_dir)

total_end_time = time.time()
print(f"Total processing time: {total_end_time - total_start_time:.4f} seconds")