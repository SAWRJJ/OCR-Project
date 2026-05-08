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

    remaining_poly = None
    remaining_crop_img = None

    if target_poly is not None:
        print(f"\n目标新poly (第1个找黑检测线): {target_poly.tolist()}")
        if target_crop_img is not None and target_crop_img.size > 0:
            crop_output_path = output_path.replace('_poly_output', '_crop_target.png')
            cv2.imwrite(crop_output_path, target_crop_img)
            print(f"目标截图已保存到: {crop_output_path}")
            print(f"目标截图尺寸: {target_crop_img.shape}")

        p0 = poly[0]
        p1 = poly[1]
        p2 = poly[2]
        p3 = poly[3]
        # target poly 左边界
        tx_min = np.min(target_poly[:, 0])

        # 原poly四个点
        p0, p1, p2, p3 = poly

        # 计算 remaining poly
        remaining_poly = np.array([
            p0,
            [tx_min, p0[1]],
            [tx_min, p3[1]],
            p3
        ], dtype=np.int32)
        print(f"\n剩余poly (原始poly的左侧边): {remaining_poly.tolist()}")

        rx_min = max(0, int(np.min(remaining_poly[:, 0])) - 5)
        rx_max = min(img.shape[1], int(np.max(remaining_poly[:, 0])) + 5)
        ry_min = max(0, int(np.min(remaining_poly[:, 1])) - 5)
        ry_max = min(img.shape[0], int(np.max(remaining_poly[:, 1])) + 5)
        remaining_crop_img = img[ry_min:ry_max, rx_min:rx_max]

        remaining_output_path = output_path.replace('_poly_output', '_crop_remaining.png')
        cv2.imwrite(remaining_output_path, remaining_crop_img)
        print(f"剩余截图已保存到: {remaining_output_path}")
        print(f"剩余截图尺寸: {remaining_crop_img.shape}")

    return target_poly, target_crop_img, remaining_poly, remaining_crop_img
if __name__ == '__main__':
    filename = "micro_0049_2300_1XVII"  # micro_0049_2300_1XVII micro_0110_2300_1X5
    image_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.jpg'
    json_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img = cv2.imread(image_path)
    output_path = f'/Users/saw/WorkSpace/work/OCR-Project/test/test10/{filename}_poly_output.png'
    textbox_angle, _ = calculate_textbox_angle(np.array(data['micro_poly'], dtype=np.int32))
    a,b,c,d = shift_step(img,data,textbox_angle=textbox_angle,output_path=output_path)