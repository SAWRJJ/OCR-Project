import cv2
import numpy as np
import json
import sys
sys.path.insert(0, '/Users/saw/WorkSpace/work/OCR-Project')
from ocr.X_detect import shift_poly_along_angle, shift_poly_along_angle_step

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


# poly = np.array([[0, 61], [270, 40], [276, 114], [6, 135]], dtype=np.int32)

def shift_step(
    img,
    data,
    textbox_angle,
    output_path,
    target_char = "X"
):
    poly = np.array(data['micro_poly'], dtype=np.int32)
    text = data["text"]
    text_list = list(text)
    index0 = text_list.index(target_char)
    print(f"index0:{index0}")
    all_polys, final_poly, shift_info, black_pixel_positions, first_black_lines, target_poly, target_crop_img = shift_poly_along_angle_step(
        poly=poly,
        angle=textbox_angle,
        step_size=2,
        debug_img=img,
        output_path=output_path,
        target_first_black_index=index0,
        shift_after_black=6
    )

    print(f"检测到黑色像素数量: {len(black_pixel_positions)}")
    print(f"首次出现黑色的检测线个数: {len(first_black_lines)}")
    print(f"首次出现黑色的检测线索引: {first_black_lines}")

    if target_poly is not None:
        print(f"\n目标新poly (第1个找黑检测线): {target_poly.tolist()}")
        if target_crop_img is not None and target_crop_img.size > 0:
            crop_output_path = output_path.replace('_poly_output', '_crop_target.png')
            cv2.imwrite(crop_output_path, target_crop_img)
            print(f"目标截图已保存到: {crop_output_path}")
            print(f"目标截图尺寸: {target_crop_img.shape}")
    return target_poly, target_crop_img
if __name__ == '__main__':
    filename = "micro_0049_2300_1XVII"  # micro_0049_2300_1XVII micro_0110_2300_1X5
    image_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.jpg'
    json_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img = cv2.imread(image_path)
    output_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}_poly_output.png'
    textbox_angle, _ = calculate_textbox_angle(np.array(data['micro_poly'], dtype=np.int32))
    a,b = shift_step(img,data,textbox_angle=textbox_angle,output_path=output_path)