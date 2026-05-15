import copy
import os
import json
import logging
import math
import traceback
from functools import reduce

import numpy as np

from sqlalchemy import true

from ocr.ocr_engine import OCREngine
from ocr.LW_detect import detect_colors, calculate_textbox_angle, find_cluster_centers
from ocr.config import Config
from ocr.X_detect import expand_poly_vertical, count_dark_pixels_in_expanded_region, \
    find_first_non_white_column_along_tilt, calculate_horizontal_tilt_angle, expand_poly, shift_poly_along_angle, \
    count_vertical_strokes
from ocr.find_boundary_dark import find_drak_remove
from ocr.utils import calculate_shift_params, fullwidth_to_halfwidth
from ocr.scan_dark_pixels import process_image_high_circularity_to_white
from ocr.shift_VII import shift_step
from ocr.find_nearest_point import find_nearest_point_to_poly, calculate_distances_to_all,filter_color_points_by_distance
import cv2

logger = logging.getLogger("ocr_system")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_PATH = os.path.join(PROJECT_ROOT, "resource", "target.json")
THRESHOLD = 100


def make_json_serializable(obj):
    """Convert numpy types and ndarray to JSON serializable Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    return obj


# 模板图片路径，假设 ll1.png 在项目根目录的 test1 文件夹下，或者 resource 下
# 根据用户描述 "ll1.png的匹配方法"，假设模板图片名为 ll1.png
# 如果不确定位置，可以尝试在 resource 或项目根目录查找
def load_template_paths(resource_dir):
    """Scans a directory for template images and returns a name:path map."""
    templates = {}
    if not os.path.isdir(resource_dir):
        logger.warning(f"Template directory not found: {resource_dir}")
        return templates
    for fname in os.listdir(resource_dir):
        # We only want image files
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            template_name = os.path.splitext(fname)[0]
            templates[template_name] = os.path.join(resource_dir, fname)
    if templates:
        logger.info(f"Loaded {len(templates)} templates from {resource_dir}")
    else:
        logger.warning(f"No template images found in {resource_dir}")
    return templates


def load_target_definitions(target_path=TARGET_PATH):
    if os.path.exists(target_path):
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 将字典逆序，以便优先匹配列表后部的项（假设列表后部优先级更高或用户有此需求）
                if isinstance(data, dict):
                    return dict(reversed(list(data.items())))
                return data
        except Exception as e:
            logger.error(f"加载 target.json 失败: {e}")
            return {}
    else:
        logger.warning(f"target.json 不存在: {target_path}")
        return {}


def check_text(potential_text, target_defs0=None):
    if target_defs0:
        target_defs = target_defs0
    else:
        target_defs = load_target_definitions()
    filename_matched_key = None
    found_match_in_filename = False
    for key, variants in target_defs.items():
        for variant in variants:
            if variant in potential_text:
                filename_matched_key = key
                found_match_in_filename = True
                break
        if found_match_in_filename:
            break
    return filename_matched_key, found_match_in_filename


def check_all_text(potential_text, target_defs0=None):
    if target_defs0:
        target_defs = target_defs0
    else:
        target_defs = load_target_definitions()
    filename_matched_key = None
    found_match_in_filename = False
    results = []
    remaining_text = potential_text

    first_round = True
    while True:
        found_this_round = False
        if first_round:
            target_defs1 = target_defs
        else:
            target_defs1 = load_target_definitions(os.path.join(PROJECT_ROOT, "resource", "target1.json"))

        if not target_defs1:
            break
        if len(potential_text) == 0:
            break
        for key, variants in target_defs1.items():
            for variant in variants:
                while variant in remaining_text:
                    variant_pos = remaining_text.find(variant)
                    if variant_pos != -1:
                        last_char_pos = variant_pos + len(variant) - 1
                        remaining_text = remaining_text[last_char_pos + 1:].strip()
                    else:
                        break
                    if not found_this_round:
                        results.append(key)
                        found_this_round = True
                        found_match_in_filename = True
                if found_this_round:
                    break
            if found_this_round:
                break

        if first_round and not found_this_round:
            return None, False

        first_round = False

        if not found_this_round:
            break

    if results:
        filename_matched_key = ''.join(results)

    return filename_matched_key, found_match_in_filename


def rotate_image_and_poly(img, poly, angle, center_point):
    angle_deg = np.degrees(angle)
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center_point, -angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))

    poly = np.array(poly, dtype=np.float32)
    ones = np.ones((poly.shape[0], 1))
    poly_hom = np.hstack([poly, ones])
    rotated_poly = rotation_matrix @ poly_hom.T
    rotated_poly = rotated_poly.T.astype(np.int32)

    return rotated_img, rotated_poly


def rotate_polys_back(poly, angle, center_point, original_shape):
    angle_deg = np.degrees(angle)
    h, w = original_shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle_deg, 1.0)
    rotated_polys = []
    # for poly in polys:
    poly = np.array(poly, dtype=np.float32)
    ones = np.ones((poly.shape[0], 1))
    poly_hom = np.hstack([poly, ones])
    rotated = rotation_matrix @ poly_hom.T
    rotated_polys.append(rotated.T.astype(np.int32))
    return rotated_polys[0]


def process_micro_images(micro_img_dir):
    """
    对指定目录下的所有小窗口图片进行二次OCR识别，并将结果更新到对应的JSON文件中。
    同时根据 target.json 进行匹配，返回所有匹配到的 key 列表。
    """
    logger.info(f"====== 开始对小窗口图片进行二次OCR识别: {micro_img_dir} ======")

    if not os.path.exists(micro_img_dir):
        logger.warning(f"目录不存在: {micro_img_dir}")
        return []

    target_defs = load_target_definitions()
    template_paths_map = load_template_paths(os.path.join(PROJECT_ROOT, "resource"))
    all_matched_keys = []

    # 获取 OCR 引擎实例
    ocr_engine = OCREngine()

    # 用于去重的字典： key -> {confidence, item_data}
    # key 的格式可以是 "filename_matchedkey" 或者全局唯一的 "matchedkey" (取决于您的去重范围)
    # 如果是“整批图片中同一个 Matched Key 只保留置信度最高的一个”，则用 key = matched_key
    # 如果是“同一张图片内”，则逻辑会有所不同。
    # 根据上下文“如果某个matchedkey已经在最后的结果中 则比较 conference 取更高的”，通常指全局去重。
    # 假设需求是：最终输出的 Excel 表中，每个 Matched Key 只出现一次（置信度最高的那次）
    best_results_map = {}

    # 遍历目录中的文件
    count = 0
    for filename in os.listdir(micro_img_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        # micro_0005_S
        if filename == "micro_0089_D3116.jpg" or "YD409_D403" in filename:  # micro_0110_2300_1X5 # micro_0085__5c0f_D # micro_0064_DOQOOSN micro_0048_XI micro_0093_XL_I_HO_00.json_input.png
            print(-1)
        img_path = os.path.join(micro_img_dir, filename)
        json_path = os.path.join(micro_img_dir, os.path.splitext(filename)[0] + ".json")

        try:
            # 初始化变量
            all_detected_texts = []
            matched_keys = []
            first_confidence = 0.0
            detailed_results = []

            # 优先检查文件名中是否包含目标 key
            # 文件名格式假设: micro_0025_S3.jpg -> S3
            # 新逻辑：micro_0044_XL_II -> XL II (去掉前两个部分，剩余部分用空格连接)
            filename_base = os.path.splitext(filename)[0]
            parts = filename_base.split('_')
            potential_text = ""
            if len(parts) > 2:
                # 舍弃前两个部分 (micro, 0044)，保留后面的部分 (XL, II)，并用空格连接
                potential_text = " ".join(parts[2:])
            elif len(parts) > 0:
                # 如果不足3个部分，取最后一个或者保持原样，视具体情况而定
                # 这里保留原有的逻辑作为 fallback，取最后一个
                potential_text = parts[-1]

            import re
            potential_text = re.sub(r'[a-z]', lambda m: m.group(0).upper(), potential_text)

            if potential_text == "XI":
                print(potential_text)
            filename_matched_key, found_match_in_filename = check_text(potential_text)
            # 新增判断：如果匹配到 S 或 X，但文件名中的文本不是精确的 S 或 X，则强制二次 OCR
            # if filename_matched_key in ('S', 'X') and potential_text != filename_matched_key:
            #     logger.info(
            #         f"文件名中匹配到 '{filename_matched_key}' 但非完全匹配 ('{potential_text}'), 将执行二次OCR: {filename}")
            #     print("不完全为S 或者X 重新判断 识别名称")
            #     filename_matched_key = None  # 作废文件名匹配，强制OCR

            if filename_matched_key:
                initial_polys = None
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        initial_json_data = json.load(f)
                    initial_polys = initial_json_data.get('micro_poly', [])
                first_poly = None
                if len(initial_polys[0]) != 2:
                    first_poly = initial_polys[0]
                else:
                    first_poly = initial_polys
                img = cv2.imread(img_path)
                filename_matched_key1 = None
                found_match_in_filename1 = False
                textbox_angle, _ = calculate_textbox_angle(first_poly)
                final_poly = copy.copy(first_poly)
                if "X" in filename_matched_key:
                    dark_count, total_count, dark_ratio = count_dark_pixels_in_expanded_region(
                        cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR),
                        first_poly,
                        expand_poly_vertical(first_poly, 5),
                        dark_threshold=118)
                    if dark_ratio > 0.2 and potential_text == "X":
                        continue
                        # print("execute test5 detect")
                        #
                        # img0 = find_drak_remove(img, dark_threshold=230)
                        # gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                        # debug_vis_path = os.path.join(micro_img_dir, 'debug',
                        #                               filename.replace('.jpg', '_scan_debug.jpg'))
                        # ex_px = 70
                        # s = filename_matched_key[1:]
                        # if s.isdigit():
                        #     s = int(s)
                        # if "L" in filename_matched_key or "Z" in filename_matched_key or len(filename_matched_key) > 2:
                        #     ex_px = 95
                        # first_non_white_col, found_scan_line_start, found_scan_line_end, non_white_pixels, expand_x, expand_y, final_scan_line_start, final_scan_line_end = find_first_non_white_column_along_tilt(
                        #     first_poly, binary, textbox_angle, debug_img=img0, output_path=debug_vis_path, ex_p=ex_px)
                        # if first_non_white_col is not None and expand_x is not None and expand_y is not None and final_scan_line_start is not None and final_scan_line_end is not None:
                        #     left_line = sorted([found_scan_line_start, found_scan_line_end], key=lambda p: p[0])[0]
                        #     right_line = sorted([final_scan_line_start, final_scan_line_end], key=lambda p: p[0])[1]
                        #     left = left_line[0] - 3
                        #     right = right_line[0]
                        #     left = max(0, left)
                        #     right = min(img.shape[1], right)
                        #     if left >= right or right <= left or left < 0 or right > img.shape[1]:
                        #         print(f"警告: 裁剪区域无效 left={left}, right={right}, img_width={img.shape[1]}")
                        #     else:
                        #         cropped = img[:, left:right + 1]
                        #     debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                        #     os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                        #     cropped_path = os.path.join(micro_img_dir, 'debug',
                        #                                 filename.replace('.jpg', '_cropped.jpg'))
                        #     cropped_path1 = os.path.join(micro_img_dir, 'debug',
                        #                                  filename.replace('.jpg', '_cropped1.jpg'))
                        #
                        #     cropped0 = find_drak_remove(cropped)
                        #     if "L" in filename_matched_key or "Z" in filename_matched_key:
                        #         _, binary_img, _ = process_image_high_circularity_to_white(
                        #             cropped0,
                        #             dark_threshold=200,
                        #             min_circularity=0.75,
                        #             binary_output_path=cropped_path1
                        #         )
                        #         cropped0 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
                        #     cv2.imwrite(cropped_path, cropped0)
                        #     results = ocr_engine.ocr.predict(cropped0)
                        #     for result in results:
                        #         if len(result) >= 2 and len(result['rec_texts']) > 0:
                        #             for i in range(len(result['rec_texts'])):
                        #                 text = result['rec_texts'][i]
                        #                 conf = result['rec_scores'][i]
                        #                 rec_poly = result['rec_polys'][i]
                        #                 restored_poly = [[int(point[0] + left), int(point[1])] for point in rec_poly]
                        #                 filename_matched_key1, found_match_in_filename1 = check_all_text(text)
                        #                 if found_match_in_filename1 and filename_matched_key1 not in matched_keys:
                        #                     matched_keys.append(filename_matched_key1)
                        #                 if found_match_in_filename1:
                        #                     break
                        #             if found_match_in_filename1:
                        #                 if os.path.exists(json_path):
                        #                     with open(json_path, 'r', encoding='utf-8') as f:
                        #                         json_data = json.load(f)
                        #                     json_data['micro_poly'] = restored_poly
                        #                     with open(json_path, 'w', encoding='utf-8') as f:
                        #                         json.dump(make_json_serializable(json_data), f, ensure_ascii=False,
                        #                                   indent=2)
                        #                 detail_item = {
                        #                     "text": text,
                        #                     "confidence": conf,
                        #                     "color_info": None,
                        #                     "matched_key": filename_matched_key1
                        #                 }
                        #                 if filename_matched_key1 and ((
                        #                         "S" in filename_matched_key1 or "X" in filename_matched_key1)):
                        #                     is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                        #                         img_path,
                        #                         filename_matched_key1,
                        #                         debug=True,
                        #                         threshold=THRESHOLD
                        #                     )
                        #                     print(template_match_res)
                        #                     detail_item["template_match_res"] = template_match_res
                        #                     detail_item["template_match_score"] = is_match
                        #                     detail_item["black_pixel_count"] = black_pixel_count
                        #                     detail_item["black_radio"] = black_radio
                        #                     detail_item["template_name"] = "color_detection"
                        #                     detail_item["color_centers_separate"] = color_centers_separate
                        #                 detailed_results.append(detail_item)
                        #             print(f"文本: {text}, 置信度: {conf:.2f}")
                        # else:
                        #     print("X识别错误")
                    else:
                        shifted_poly = first_poly
                        x_coords = [p[0] for p in first_poly]
                        y_coords = [p[1] for p in first_poly]
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        cropped_1 = img[min_y:max_y, min_x:max_x]
                        if "X" in potential_text and potential_text[0].isdigit():
                            results = ocr_engine.ocr.predict(cropped_1)
                            found_match_in_filename1 = None
                            for result in results:
                                for i in range(len(result['rec_texts'])):
                                    text = result['rec_texts'][i]
                                    if len(text) > 0 and text != potential_text and text[0] == "X":
                                        potential_text = text
                                        first_confidence = result['rec_scores'][i]
                                        rec_poly = result['rec_polys'][i]
                                        restored_poly = [[int(point[0] + min_x), int(point[1] + min_y)] for point in
                                                         rec_poly]
                                        x_coords = [p[0] for p in restored_poly]
                                        y_coords = [p[1] for p in restored_poly]
                                        poly_min_x, poly_max_x = min(x_coords), max(x_coords)
                                        poly_min_y, poly_max_y = min(y_coords), max(y_coords)
                                        expand_pixels = 2
                                        restored_poly = [
                                            [poly_min_x - expand_pixels, poly_min_y - expand_pixels],
                                            [poly_max_x + expand_pixels, poly_min_y - expand_pixels],
                                            [poly_max_x + expand_pixels, poly_max_y + expand_pixels],
                                            [poly_min_x - expand_pixels, poly_max_y + expand_pixels]
                                        ]
                                        if os.path.exists(json_path):
                                            with open(json_path, 'r', encoding='utf-8') as f:
                                                json_data = json.load(f)
                                            json_data['text'] = potential_text
                                            json_data['micro_poly'] = restored_poly
                                            first_poly = restored_poly
                                            final_poly = restored_poly
                                            with open(json_path, 'w', encoding='utf-8') as f:
                                                json.dump(make_json_serializable(json_data), f, ensure_ascii=False,
                                                          indent=2)
                                        break
                        rect_width = max_x - min_x
                        if rect_width > 130:
                            ex_px = 75
                            # 'XⅣ'
                            if "F" in filename_matched_key or "L" in filename_matched_key or "Z" in filename_matched_key or "Ⅳ" in filename_matched_key or "V" in filename_matched_key:
                                ex_px = 120
                            elif "YXD" in potential_text:
                                ex_px = 155
                            elif "XD" in potential_text:
                                ex_px = 130

                            debug_vis_path = os.path.join(micro_img_dir, 'debug',
                                                          filename.replace('.jpg', '_shift_poly.jpg'))
                            shifted_poly, shift_line_start, shift_line_end, shifted_left_poly = shift_poly_along_angle(
                                first_poly,
                                textbox_angle,
                                shift_distance=ex_px,
                                debug_img=img,
                                output_path=debug_vis_path
                            )
                        x_min = max(0, int(min(p[0] for p in shifted_poly)))
                        x_max = min(img.shape[1], int(max(p[0] for p in shifted_poly)))
                        y_min = max(0, int(min(p[1] for p in shifted_poly)))
                        y_max = min(img.shape[0], int(max(p[1] for p in shifted_poly)))
                        cropped = img[y_min:y_max, x_min:x_max]

                        if "X" in potential_text and potential_text[0].isdigit():
                            debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                            output_path = os.path.join(micro_img_dir, 'debug',
                                                       filename.replace('.jpg', '_poly_output.jpg'))
                            if os.path.exists(json_path):
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                target_poly, cropped = shift_step(img, data, textbox_angle=textbox_angle,
                                                                  output_path=output_path)
                                shifted_poly = target_poly
                                x_min = max(0, int(min(p[0] for p in shifted_poly)))
                                x_max = min(img.shape[1], int(max(p[0] for p in shifted_poly)))
                                y_min = max(0, int(min(p[1] for p in shifted_poly)))
                                y_max = min(img.shape[0], int(max(p[1] for p in shifted_poly)))

                        if os.path.exists(json_path):
                            with open(json_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            shifted_poly_list = [[int(point[0]), int(point[1])] for point in shifted_poly]
                        debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                        cropped_path = os.path.join(micro_img_dir, 'debug',
                                                    filename.replace('.jpg', '_cropped_expect.jpg'))
                        # vis_img = img.copy()
                        # cv2.polylines(vis_img, [expanded_array], isClosed=True, color=(0, 255, 0), thickness=2)
                        # cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cropped0 = find_drak_remove(cropped, dark_threshold=190, find_adjacent_color_regions=True,
                                                    save_circle=False, remove_light_white=True)
                        if "FX" in filename_matched_key:
                            cropped_path1 = os.path.join(micro_img_dir, 'debug',
                                                         filename.replace('.jpg', '_cropped1.jpg'))
                            _, binary_img, _ = process_image_high_circularity_to_white(
                                cropped0,
                                dark_threshold=200,
                                min_circularity=0.75,
                                binary_output_path=cropped_path1,
                                remove_circle=False
                            )
                            cropped0 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(cropped_path, cropped0)
                        results = ocr_engine.ocr.predict(cropped0)
                        first_confidence = 1.0
                        if len(results) == 0:
                            continue
                        elif len(results[0]["rec_texts"]) == 0:
                            continue
                        elif len(results[0]["rec_texts"]) > 0:
                            for result in results:
                                for i in range(len(result['rec_texts'])):
                                    potential_text = result['rec_texts'][i]
                                    if "X" in potential_text[-1]:
                                        continue
                                    if "T" in potential_text[-1]:
                                        potential_text = potential_text.replace("T", "J")
                                    if "VII" in potential_text or "VI" in potential_text or "VⅢ" in potential_text:
                                        VII_count = count_vertical_strokes(cropped0)
                                        if VII_count <= 2:
                                            potential_text = "XVII"
                                        elif VII_count == 3:
                                            potential_text = "XVIII"
                                    first_confidence = result['rec_scores'][i]
                                    rec_poly = result['rec_polys'][i]
                                    restored_poly = [[int(point[0] + x_min), int(point[1] + y_min)] for point in
                                                     rec_poly]
                                    filename_matched_key, found_match_in_filename1 = check_all_text(potential_text)
                                    if found_match_in_filename1 and filename_matched_key not in matched_keys:
                                        matched_keys.append(filename_matched_key)
                                    if found_match_in_filename1:
                                        if os.path.exists(json_path):
                                            with open(json_path, 'r', encoding='utf-8') as f:
                                                json_data = json.load(f)
                                            json_data['text'] = potential_text
                                            json_data['micro_poly'] = restored_poly
                                            final_poly = restored_poly
                                            with open(json_path, 'w', encoding='utf-8') as f:
                                                json.dump(make_json_serializable(json_data), f, ensure_ascii=False,
                                                          indent=2)
                                        break
                        # 如果文件名已经包含了 key，则跳过 OCR，直接构造结果
                        logger.info(f"文件名匹配成功，跳过 OCR: {filename} -> {filename_matched_key}")
                        matched_keys.append(filename_matched_key)
                        all_detected_texts.append(potential_text)

                        detail_item = {
                            "text": potential_text,
                            "confidence": first_confidence,
                            "color_info": None,
                            "matched_key": filename_matched_key,
                            "micro_poly": final_poly
                        }

                        if filename_matched_key and ((
                                "S" in filename_matched_key or "X" in filename_matched_key)):
                            is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                                img_path,
                                filename_matched_key,
                                debug=True,
                                threshold=THRESHOLD
                            )
                            print(template_match_res)
                            detail_item["template_match_res"] = template_match_res
                            detail_item["template_match_score"] = is_match
                            detail_item["black_pixel_count"] = black_pixel_count
                            detail_item["black_radio"] = black_radio
                            detail_item["template_name"] = "color_detection"
                            detail_item["color_centers_separate"] = color_centers_separate

                        detailed_results.append(detail_item)
                elif "S" in filename_matched_key:
                    tilt_angle = calculate_horizontal_tilt_angle(first_poly)
                    x_coords = [point[0] for point in first_poly]
                    textbox_length = max(x_coords) - min(x_coords)
                    first_poly_array = np.array(first_poly)
                    poly_center_x = int(np.mean(first_poly_array[:, 0]))
                    poly_center_y = int(np.mean(first_poly_array[:, 1]))
                    center_point = (poly_center_x, poly_center_y)
                    angle_threshold = 40
                    if abs(np.degrees(textbox_angle)) > angle_threshold:
                        print(f"角度大于40度，进行旋转校正...")
                        img, first_poly = rotate_image_and_poly(img, first_poly, textbox_angle, center_point)
                    print(textbox_length)
                    if textbox_length > 120 and potential_text[0] == "S" and len(potential_text) > 4:
                        expand_length = 55
                        if "F" in filename_matched_key or len(filename_matched_key) > 2:
                            expand_length = 120
                        second_poly, expanded_poly = calculate_shift_params(first_poly, extend_length=expand_length)
                    else:
                        expanded_poly = expand_poly(first_poly, expand_x=50, expand_y=6, angle=tilt_angle)
                    if textbox_length > 350:
                        debug_vis_path = os.path.join(micro_img_dir, 'debug',
                                                      filename.replace('.jpg', '_shift_poly.jpg'))
                        ex_px = textbox_length // 2
                        shifted_poly, shift_line_start, shift_line_end, shifted_left_poly = shift_poly_along_angle(
                            first_poly,
                            textbox_angle,
                            shift_distance=ex_px,
                            debug_img=img,
                            output_path=debug_vis_path
                        )
                        expanded_poly = shifted_left_poly
                    # x_min = max(0, int(min(p[0] for p in expanded_poly)) - 2)
                    # x_max = min(img.shape[1], int(max(p[0] for p in expanded_poly)) + 2)
                    # y_min = max(0, int(min(p[1] for p in expanded_poly)) - 2)
                    # y_max = min(img.shape[0], int(max(p[1] for p in expanded_poly)) + 2)
                    x_min = max(0, int(min(p[0] for p in expanded_poly)))
                    x_max = min(img.shape[1], int(max(p[0] for p in expanded_poly)))
                    y_min = max(0, int(min(p[1] for p in expanded_poly)))
                    y_max = min(img.shape[0], int(max(p[1] for p in expanded_poly)))
                    cropped = img[y_min:y_max, x_min:x_max]
                    debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    cropped_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_cropped.jpg'))
                    cropped_path0 = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_cropped0.jpg'))
                    # vis_img = img.copy()
                    # cv2.polylines(vis_img, [expanded_array], isClosed=True, color=(0, 255, 0), thickness=2)
                    # cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.imwrite(cropped_path0, cropped)
                    cropped0 = find_drak_remove(cropped, dark_threshold=190, find_adjacent_color_regions=True,
                                                save_circle=False, remove_light_white=True)
                    if "SL" in filename_matched_key:
                        cropped_path1 = os.path.join(micro_img_dir, 'debug',
                                                     filename.replace('.jpg', '_cropped1.jpg'))
                        gray = cv2.cvtColor(cropped0, cv2.COLOR_BGR2GRAY)
                        _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                        cv2.imwrite(cropped_path1, binary_img)
                        cropped0 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(cropped_path, cropped0)
                    expanded_textbox_angle, _ = calculate_textbox_angle(expanded_poly)
                    # if abs(np.degrees(expanded_textbox_angle)) > 40:
                    #     print(f"角度大于40度，进行旋转校正...")
                    #     cropped0,expanded_poly = rotate_image_and_poly(cropped0, expanded_poly, expanded_textbox_angle, center_point)
                    #     cv2.imwrite(cropped_path, cropped0)
                    results = ocr_engine.ocr.predict(cropped0)
                    for result in results:
                        if len(result) >= 2 and len(result['rec_texts']) > 0:
                            main_text = ""
                            main_poly = None
                            main_angle_deg = None
                            for i in range(len(result['rec_texts'])):
                                text = result['rec_texts'][i]
                                if "S" in text:
                                    main_text = text
                                    main_poly = result['rec_polys'][i]
                                    # 计算主框弧度并直接转换为角度
                                    rad, _ = calculate_textbox_angle(main_poly)
                                    main_angle_deg = np.degrees(rad)
                                    break
                            filtered_texts = []
                            for i in range(len(result['rec_texts'])):
                                text = result['rec_texts'][i]
                                conf = result['rec_scores'][i]
                                rec_poly = result['rec_polys'][i]
                                # 如果找到了主文本，则进行角度比对
                                if main_angle_deg is not None:
                                    current_rad, _ = calculate_textbox_angle(rec_poly)
                                    current_angle_deg = np.degrees(current_rad)
                                    # 计算角度差的绝对值
                                    angle_diff = abs(current_angle_deg - main_angle_deg)
                                    # 处理角度周期性（防止 179° 和 -179° 被判定为差 358°）
                                    if angle_diff > 180:
                                        angle_diff = 360 - angle_diff
                                    # 如果与主框角度差距小于 20°，则跳过该文本
                                    if 0 < angle_diff < 20:
                                        continue
                                # 清洗并添加符合条件的文本
                                clean_text = text.replace(' ', '')
                                filtered_texts.append(clean_text)

                            rec_poly = main_poly
                            text = fullwidth_to_halfwidth(''.join(filtered_texts))
                            if "VII" in text or "VI" in text:
                                VII_count = count_vertical_strokes(cropped0)
                                if VII_count <= 2:
                                    text = "SVII"
                                elif VII_count == 3:
                                    text = "SVIII"
                            if abs(np.degrees(textbox_angle)) > angle_threshold:
                                rec_poly = rotate_polys_back(rec_poly, textbox_angle,
                                                             center_point,
                                                             img.shape)
                            restored_poly = [[int(point[i] + x_min), int(point[1] + y_min)] for point in rec_poly]
                            filename_matched_key1, found_match_in_filename1 = check_all_text(text)
                            if found_match_in_filename1 and filename_matched_key1 not in matched_keys:
                                matched_keys.append(filename_matched_key1)
                            # if found_match_in_filename1:
                            #     break
                        if found_match_in_filename1:
                            if os.path.exists(json_path):
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                                json_data['micro_poly'] = restored_poly
                                json_data["text"] = text
                                final_poly = restored_poly
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(make_json_serializable(json_data), f, ensure_ascii=False, indent=2)
                            detail_item = {
                                "text": text,
                                "confidence": conf,
                                "color_info": None,
                                "matched_key": filename_matched_key1,
                                "micro_poly": final_poly
                            }
                            if filename_matched_key1 and ((
                                    "S" in filename_matched_key1 or "X" in filename_matched_key1)):
                                is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                                    img_path,
                                    filename_matched_key1,
                                    debug=True,
                                    threshold=THRESHOLD
                                )
                                print(template_match_res)
                                detail_item["template_match_res"] = template_match_res
                                detail_item["template_match_score"] = is_match
                                detail_item["black_pixel_count"] = black_pixel_count
                                detail_item["black_radio"] = black_radio
                                detail_item["template_name"] = "color_detection"
                                detail_item["color_centers_separate"] = color_centers_separate
                            detailed_results.append(detail_item)
                        print(f"文本: {text}, 置信度: {conf:.2f}")
                else:
                    # 如果文件名已经包含了 key，则跳过 OCR，直接构造结果
                    logger.info(f"文件名匹配成功，跳过 OCR: {filename} -> {filename_matched_key}")
                    filename_matched_key1, found_match_in_filename1 = check_all_text(potential_text)
                    # if found_match_in_filename1 and filename_matched_key1 not in matched_keys:
                    #     matched_keys.append(filename_matched_key1)
                    # potential_text = filename_matched_key
                    expanded_poly = first_poly
                    first_confidence = 1.0
                    x_min = max(0, int(min(p[0] for p in expanded_poly)))
                    x_max = min(img.shape[1], int(max(p[0] for p in expanded_poly)))
                    y_min = max(0, int(min(p[1] for p in expanded_poly)))
                    y_max = min(img.shape[0], int(max(p[1] for p in expanded_poly)))
                    cropped = img[y_min:y_max, x_min:x_max]
                    debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    cropped_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_cropped.jpg'))
                    cropped_path0 = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_cropped0.jpg'))
                    # vis_img = img.copy()
                    # cv2.polylines(vis_img, [expanded_array], isClosed=True, color=(0, 255, 0), thickness=2)
                    # cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.imwrite(cropped_path0, cropped)
                    cropped0 = find_drak_remove(cropped, dark_threshold=190, find_adjacent_color_regions=True,
                                                save_circle=False, remove_light_white=True, not_save_boundary=True,
                                                min_circularity=0.81)
                    cv2.imwrite(cropped_path, cropped0)
                    results = ocr_engine.ocr.predict(cropped0)
                    for result in results:
                        rec_texts = result.get('rec_texts', [])
                        rec_scores = result.get('rec_scores', [])
                        rec_polys = result.get('rec_polys', [])
                        for i in range(len(rec_texts)):
                            if i >= len(rec_scores) or i >= len(rec_polys):
                                continue
                            potential_text = rec_texts[i]
                            potential_text = potential_text.replace("YD"," ")
                            first_confidence = rec_scores[i]
                            rec_poly = rec_polys[i]
                            restored_poly = [[int(point[0] + x_min), int(point[1] + y_min)] for point in rec_poly]
                            filename_matched_key, found_match_in_filename1 = check_all_text(potential_text)
                            if found_match_in_filename1 and filename_matched_key not in matched_keys:
                                matched_keys.append(filename_matched_key)
                            if found_match_in_filename1:
                                if os.path.exists(json_path):
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    json_data['text'] = potential_text
                                    json_data['micro_poly'] = restored_poly
                                    final_poly = restored_poly
                                    with open(json_path, 'w', encoding='utf-8') as f:
                                        json.dump(make_json_serializable(json_data), f, ensure_ascii=False, indent=2)
                                break
                    # filename_matched_key =potential_text
                    # matched_keys.append(filename_matched_key)
                    # all_detected_texts.append(potential_text)

                    detail_item = {
                        "text": potential_text,
                        "confidence": first_confidence,
                        "color_info": None,
                        "matched_key": filename_matched_key,
                        "micro_poly": final_poly
                    }

                    if filename_matched_key:
                        is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                            img_path,
                            filename_matched_key,
                            debug=True,
                            threshold=THRESHOLD
                        )
                        print(template_match_res)
                        detail_item["template_match_res"] = template_match_res
                        detail_item["template_match_score"] = is_match
                        detail_item["black_pixel_count"] = black_pixel_count
                        detail_item["black_radio"] = black_radio
                        detail_item["template_name"] = "color_detection"
                        detail_item["color_centers_separate"] = color_centers_separate

                    detailed_results.append(detail_item)

            existing_color_presencse = None
            existing_color_stats = None
            json_data = None

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # 保留已有的颜色信息（如果存在）
                existing_color_presence = json_data.get("color_presence")
                existing_color_stats = json_data.get("color_stats")

            # 将整图颜色信息赋给每个检测到的字符结果作为参考，并更新 best_results_map
            # 即使没有 json 文件，也要处理并返回结果
            for res in detailed_results:
                res["color_presence"] = existing_color_presence
                res["color_stats"] = existing_color_stats
                color_centers_separate = res.get("color_centers_separate")
                micro_poly = res.get("micro_poly")
                if filename == "micro_0110_2300_1X5.jpg" or "XII0" in filename:  # micro_0110_2300_1X5 # micro_0085__5c0f_D # micro_0064_DOQOOSN micro_0048_XI micro_0093_XL_I_HO_00.json_input.png
                    print(-1)
                color_centers_separate["yellow"] = find_cluster_centers(color_centers_separate.get("yellow", []), distance_threshold=25)
                nearest_point, min_dist, poly_center, nearest_white_points = find_nearest_point_to_poly(micro_poly, color_centers_separate)
                color_centers_separate = filter_color_points_by_distance(color_centers_separate, threshold=255, reference_point=nearest_point)
                res["color_centers_separate"] = color_centers_separate
                m_key = res.get("matched_key")
                if "F" in m_key and color_centers_separate and all(len(v) == 0 for v in color_centers_separate.values()):
                    continue
                if "F" not in m_key and "Y" not in m_key and color_centers_separate and len(nearest_white_points)==0:
                    continue
                # 检查并更新全局最佳结果

                if m_key == "X":
                    print(0)
                if m_key:
                    if "D" in m_key and "D" not in filename:
                        continue
                    # 构造最终结果项
                    final_item = {
                        "filename": filename,
                        "matched_keys": [m_key],  # 为了兼容之前的结构，虽然这里只有一个
                        "details": [res]  # 只包含这一个 detail
                    }

                    # 根据是否有模板匹配结果，决定使用哪个分数进行比较
                    compare_score = res["confidence"]

                    # 如果 m_key 已存在，进行比较
                    if m_key in best_results_map:
                        existing_item = best_results_map[m_key]
                        # 如果之前的item有模板匹配分数，优先用它比较
                        existing_score = existing_item.get("item", {}).get("details", [{}])[0].get(
                            "confidence", existing_item["confidence"])

                        if compare_score - existing_score > 0.12:
                            best_results_map[m_key] = {
                                "confidence": res["confidence"],  # 仍然记录原始置信度
                                "item": final_item
                            }
                    else:  # 如果 m_key 不存在，直接添加
                        best_results_map[m_key] = {
                            "confidence": res["confidence"],
                            "item": final_item
                        }

            # 如果对应 JSON 文件存在，则更新
            if json_data is not None:
                data = json_data
                # 添加二次识别结果字段
                # 将所有识别到的文本用空格拼接，保留所有信息
                data['re_ocr_text'] = " ".join(all_detected_texts)
                # confidence 取第一个（如果有）或者 0.0，仅作参考
                data['re_ocr_confidence'] = first_confidence

                # 新增：详细的识别结果列表
                data['re_ocr_details'] = detailed_results

                # 将匹配到的 key 放入结果列表（写入 JSON）
                data['matched_keys'] = matched_keys

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(make_json_serializable(data), f, ensure_ascii=False, indent=2)

                count += 1
                # logger.debug(f"已更新: {filename} -> {all_detected_texts}, matched: {matched_keys}")
            else:
                # 如果没有 JSON，也可以选择创建一个，但按现有逻辑主要是补充信息
                # 既然已经识别了，可以考虑创建一个新的 json？
                # 暂时保持 pass，不强制创建文件，但上面的 best_results_map 已经更新了
                pass

        except Exception as e:
            logger.error(f"处理小窗口图片失败 {filename}: {e}\n{traceback.format_exc()}")

    # 将最佳结果转换为列表返回
    all_matched_keys = [val["item"] for val in best_results_map.values()]
    print(all_matched_keys)
    logger.info(f"小窗口二次OCR识别完成，共处理 {count} 张图片")
    return all_matched_keys


def save_results_to_excel(all_results, output_file="ocr_results.xlsx", micro_img_dir=None):
    """
    将 OCR 结果保存为 Excel 文件
    包含字段: matched_keys, confidence, image_path, color_presence
    """
    import pandas as pd
    import shutil

    results_dir = None
    if micro_img_dir:
        results_dir = os.path.join(os.path.dirname(micro_img_dir), "results")
        os.makedirs(results_dir, exist_ok=True)

    data_list = []
    copied_files = set()
    print(all_results)
    for item in all_results:
        filename = item.get("filename", "")
        details = item.get("details", [])

        # 获取完整图片路径（假设 micro_img_dir 是已知的或者可以通过 filename 推断）
        # 这里只保存文件名作为相对路径
        image_path = filename

        # 复制图片和JSON到results目录
        if results_dir and filename:
            src_img_path = os.path.join(micro_img_dir, filename)
            src_json_path = os.path.join(micro_img_dir, os.path.splitext(filename)[0] + ".json")
            if os.path.exists(src_img_path) and filename not in copied_files:
                shutil.copy2(src_img_path, os.path.join(results_dir, filename))
                copied_files.add(filename)
            if os.path.exists(src_json_path):
                shutil.copy2(src_json_path, os.path.join(results_dir, os.path.basename(src_json_path)))

        # 如果没有 details，也要记录一条
        if not details:
            data_list.append({
                "filename": filename,
                "matched_key": "",
                "confidence": 0.0,
                "template_name": "",
                "template_match_res": "",
                "template_match_score": "",
                "black_pixel_count": "",
                "black_radio": "",
                "color_blue": "No",
                "color_green": "No",
                "color_yellow": "No",
                "color_red": "No",
                "color_white": "No",
                "red_centers": "",
                "yellow_centers": "",
                "green_centers": ""
            })
            continue

        for detail in details:
            text = detail.get("text", "")
            confidence = detail.get("confidence", 0.0)
            # 颜色信息
            # color_presence 结构: {'blue': False, 'green': True, ...}
            color_presence = detail.get("color_presence", {}) or {}
            clean_text = text.strip()
            current_matched_key = ""
            current_matched_key, find_match = check_all_text(clean_text)
            # 如果只想输出匹配到的结果：
            if "XJ" in text:
                print(-1)
            if current_matched_key:
                color_centers = detail.get("color_centers_separate", {})

                def format_centers(centers):
                    if not centers:
                        return ""
                    return ";".join([f"({c[0]},{c[1]})" for c in centers])

                yellow_length = len(color_centers.get("yellow", []))
                yr = find_cluster_centers(color_centers.get("yellow", []), distance_threshold=25)
                yl = len(yr)
                gl = 0
                rl = 0
                if len(color_centers.get("green", [])) > 0:
                    gl = len(color_centers.get("green", []))
                if len(color_centers.get("red", [])) > 0:
                    rl = 1
                row = {
                    "filename": filename,
                    "matched_key": current_matched_key,
                    "confidence": confidence,
                    "template_name": detail.get("template_name", ""),
                    "template_match_res": detail.get("template_match_res", ""),
                    "template_match_score": detail.get("template_match_score", ""),
                    "black_pixel_count": detail.get("black_pixel_count", ""),
                    "black_radio": detail.get("black_radio", ""),
                    # "color_blue": "Yes" if color_presence.get("blue") else "No",
                    # "color_green": "Yes" if color_presence.get("green") else "No",
                    # "color_yellow": "Yes" if color_presence.get("yellow") else "No",
                    # "color_red": "Yes" if color_presence.get("red") else "No",
                    "color_white": "Yes" if color_presence.get("white") else "No",
                    "red_centers": rl,
                    "yellow_centers": yl,
                    "green_centers": gl,
                    "blue_centers": len(color_centers.get("blue", [])),
                    "white_centers": len(color_centers.get("white", [])),
                }
                data_list.append(row)

    if not data_list:
        logger.warning("没有数据需要保存到 Excel")
        return

    df = pd.DataFrame(data_list)
    df = df.sort_values(by="matched_key", key=lambda x: x.str.lower() if x.dtype == object else x, ascending=False)
    columns = ["filename", "matched_key", "confidence", "template_name", "template_match_res", "template_match_score",
               "black_pixel_count", "black_radio",
               "color_white",
               "red_centers", "yellow_centers", "green_centers", "blue_centers",
               "white_centers"]  # "color_blue", "color_green", "color_yellow", "color_red",
    final_columns = [c for c in columns if c in df.columns]
    df = df[final_columns]

    try:
        df.to_excel(output_file, index=False)
        logger.info(f"结果已保存到 Excel: {output_file}")
    except Exception as e:
        logger.error(f"保存 Excel 失败: {e}")


def adjust_textbox_edge(img, poly, direction, json_path, crop_size=40):
    import cv2
    import numpy as np
    import os
    from paddleocr import PaddleOCR

    use_adjusted_edge = False
    adjusted_edge = None
    crop_ocr_result = None
    crop_ocr = OCREngine()
    crop_result = crop_ocr.predict(img, adjust_type=True)

    for res in crop_result:
        if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
            for line in res["rec_texts"]:
                check_res0, check_res1 = check_text(line)
                if check_res1:
                    crop_ocr_result = res
                    return True, True, crop_ocr_result
                print(f"识别文本: {line}")
    x_coords = [point[0] for point in poly]
    textbox_length = max(x_coords) - min(x_coords)

    y_coords = [point[1] for point in poly]
    min_y = min(y_coords)
    max_y = max(y_coords)

    if direction == 'right':
        rightmost_x = max(x_coords)
        yellow_box_left = rightmost_x - crop_size
        yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y),
                          (yellow_box_left, max_y)]
        cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
        x1 = int(yellow_box_left)
        y1 = int(min_y)
        x2 = int(rightmost_x)
        y2 = int(max_y)
    elif direction == 'left':
        leftmost_x = min(x_coords)
        yellow_box_right = leftmost_x + crop_size
        yellow_contour = [(leftmost_x, min_y), (yellow_box_right, min_y), (yellow_box_right, max_y),
                          (leftmost_x, max_y)]
        cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
        print(f"已绘制最左侧40长度的黄色轮廓: {yellow_contour}")

        x1 = int(leftmost_x)
        y1 = int(min_y)
        x2 = int(yellow_box_right)
        y2 = int(max_y)
    else:
        return use_adjusted_edge, adjusted_edge, crop_ocr_result
    height, width = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 > x1 and y2 > y1:
        cropped_img = img[y1:y2, x1:x2]

        if not os.path.exists('output'):
            os.makedirs('output')

        filename = os.path.basename(json_path).replace('_res.json', '')
        crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
        cv2.imwrite(crop_output_path, cropped_img)
        print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")

        crop_result = crop_ocr.predict([crop_output_path], adjust_type=True)

        ocr_success = False
        print("黄色轮廓区域的OCR识别结果:")
        for res in crop_result:
            if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                ocr_success = True
                crop_ocr_result = res
                for line in res["rec_texts"]:
                    print(f"识别文本: {line}")

        if ocr_success:
            print("OCR识别成功，将调整边缘")
            if direction == 'right':
                adjusted_edge = [(yellow_box_left, min_y), (yellow_box_left, max_y)]
            elif direction == 'left':
                adjusted_edge = [(yellow_box_right, min_y), (yellow_box_right, max_y)]
            use_adjusted_edge = True
            print(f"已设置调整后的边缘: {adjusted_edge}")

    return use_adjusted_edge, adjusted_edge, crop_ocr_result


if __name__ == '__main__':
    micro_dir = "test/test16/micro"
    res = process_micro_images(micro_dir)
    print(len(res))
    for r in res:
        print(r)
