import os
import json
import logging
import math
from .ocr_engine import OCREngine
from .LW_detect import detect_colors, calculate_textbox_angle
from .config import Config
from .X_detect import expand_poly_vertical, count_dark_pixels_in_expanded_region, \
    find_first_non_white_column_along_tilt, calculate_horizontal_tilt_angle, expand_poly
from .find_boundary_dark import find_drak_remove
from .utils import calculate_shift_params
import cv2
import numpy as np

logger = logging.getLogger("ocr_system")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_PATH = os.path.join(PROJECT_ROOT, "resource", "target.json")
THRESHOLD = 100


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


def load_target_definitions():
    if os.path.exists(TARGET_PATH):
        try:
            with open(TARGET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 将字典逆序，以便优先匹配列表后部的项（假设列表后部优先级更高或用户有此需求）
                if isinstance(data, dict):
                    return dict(reversed(list(data.items())))
                return data
        except Exception as e:
            logger.error(f"加载 target.json 失败: {e}")
            return {}
    else:
        logger.warning(f"target.json 不存在: {TARGET_PATH}")
        return {}


def check_text(potential_text):
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
        if filename == "micro_0017_S.jpg":
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

            # 新增判断：如果匹配到 S 或 X，但文件名中的文本不是精确的 S 或 X，则强制二次 OCR
            if filename_matched_key in ('S', 'X') and potential_text != filename_matched_key:
                logger.info(
                    f"文件名中匹配到 '{filename_matched_key}' 但非完全匹配 ('{potential_text}'), 将执行二次OCR: {filename}")
                print("不完全为S 或者X 重新判断 识别名称")
                filename_matched_key = None  # 作废文件名匹配，强制OCR

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
                if "X" in filename_matched_key:
                    dark_count, total_count, dark_ratio = count_dark_pixels_in_expanded_region(
                        cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR),
                        first_poly,
                        expand_poly_vertical(first_poly, 5),
                        dark_threshold=128)
                    if dark_ratio:
                        print("execute test5 detect")
                        textbox_angle, _ = calculate_textbox_angle(first_poly)
                        img0 = find_drak_remove(img, dark_threshold=230)
                        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                        debug_vis_path = os.path.join(micro_img_dir, 'debug',
                                                      filename.replace('.jpg', '_scan_debug.jpg'))
                        first_non_white_col, found_scan_line_start, found_scan_line_end, non_white_pixels, expand_x, expand_y, final_scan_line_start, final_scan_line_end = find_first_non_white_column_along_tilt(
                            first_poly, binary, textbox_angle, debug_img=img0, output_path=debug_vis_path)
                        if first_non_white_col is not None and expand_x is not None and expand_y is not None and final_scan_line_start is not None and final_scan_line_end is not None:
                            left_line = sorted([found_scan_line_start, found_scan_line_end], key=lambda p: p[0])[0]
                            right_line = sorted([final_scan_line_start, final_scan_line_end], key=lambda p: p[0])[1]
                            left = left_line[0] - 3
                            right = right_line[0]
                            left = max(0, left)
                            right = min(img.shape[1], right)
                            if left >= right or right <= left or left < 0 or right > img.shape[1]:
                                print(f"警告: 裁剪区域无效 left={left}, right={right}, img_width={img.shape[1]}")
                            else:
                                cropped = img[:, left:right + 1]
                            debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                            cropped_path = os.path.join(micro_img_dir, 'debug',
                                                        filename.replace('.jpg', '_cropped.jpg'))

                            cropped0 = find_drak_remove(cropped)
                            cv2.imwrite(cropped_path, cropped0)
                            results = ocr_engine.ocr.predict(cropped0)
                            for result in results:
                                if len(result) >= 2 and len(result['rec_texts']) > 0:
                                    for i in range(len(result['rec_texts'])):
                                        text = result['rec_texts'][i]
                                        conf = result['rec_scores'][i]
                                        rec_poly = result['rec_polys'][i]
                                        restored_poly = [[int(point[0] + left), int(point[1])] for point in rec_poly]
                                        for key, variants in target_defs.items():
                                            for variant in variants:
                                                if variant in text:
                                                    filename_matched_key1 = key
                                                    found_match_in_filename1 = True
                                                    break
                                            if found_match_in_filename1:
                                                break
                                        if found_match_in_filename1:
                                            break
                                    if found_match_in_filename1:
                                        if os.path.exists(json_path):
                                            with open(json_path, 'r', encoding='utf-8') as f:
                                                json_data = json.load(f)
                                            json_data['micro_poly'] = restored_poly
                                            with open(json_path, 'w', encoding='utf-8') as f:
                                                json.dump(json_data, f, ensure_ascii=False, indent=2)
                                        detail_item = {
                                            "text": text,
                                            "confidence": conf,
                                            "color_info": None,
                                            "matched_key": filename_matched_key1
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
                            print("X识别错误")
                    else:
                        # 如果文件名已经包含了 key，则跳过 OCR，直接构造结果
                        logger.info(f"文件名匹配成功，跳过 OCR: {filename} -> {filename_matched_key}")
                        matched_keys.append(filename_matched_key)
                        all_detected_texts.append(potential_text)
                        first_confidence = 1.0

                        detail_item = {
                            "text": potential_text,
                            "confidence": 1.0,
                            "color_info": None,
                            "matched_key": filename_matched_key
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
                    print(textbox_length)
                    if textbox_length > 120 and potential_text[0] == "S":
                        expand_length = 55
                        if "F" in filename_matched_key:
                            expand_length = 85
                        second_poly, expanded_poly = calculate_shift_params(first_poly, extend_length=expand_length)
                    else:
                        expanded_poly = expand_poly(first_poly, expand_x=40, expand_y=6, angle=tilt_angle)
                    expanded_array = np.array(expanded_poly, dtype=np.int32)
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
                    cropped0 = find_drak_remove(cropped,dark_threshold=190)
                    cv2.imwrite(cropped_path, cropped0)
                    results = ocr_engine.ocr.predict(cropped0)
                    for result in results:
                        if len(result) >= 2 and len(result['rec_texts']) > 0:
                            for i in range(len(result['rec_texts'])):
                                text = ''.join(results[i]['rec_texts']).replace(' ', '')
                                conf = result['rec_scores'][i]
                                rec_poly = result['rec_polys'][i]
                                restored_poly = [[int(point[i] + x_min), int(point[1] + y_min)] for point in rec_poly]
                                for key, variants in target_defs.items():
                                    for variant in variants:
                                        if variant in text:
                                            filename_matched_key1 = key
                                            found_match_in_filename1 = True
                                            break
                                    if found_match_in_filename1:
                                        break
                                if found_match_in_filename1:
                                    break
                            if found_match_in_filename1:
                                if os.path.exists(json_path):
                                    with open(json_path, 'r', encoding='utf-8') as f:
                                        json_data = json.load(f)
                                    json_data['micro_poly'] = restored_poly
                                    json_data["text"] = text
                                    with open(json_path, 'w', encoding='utf-8') as f:
                                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                                detail_item = {
                                    "text": text,
                                    "confidence": conf,
                                    "color_info": None,
                                    "matched_key": filename_matched_key1
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
                # if "S" in filename_matched_key and filename_matched_key1 != filename_matched_key:
                #     continue
                else:
                    # 如果文件名已经包含了 key，则跳过 OCR，直接构造结果
                    logger.info(f"文件名匹配成功，跳过 OCR: {filename} -> {filename_matched_key}")
                    matched_keys.append(filename_matched_key)
                    all_detected_texts.append(potential_text)
                    first_confidence = 1.0

                    detail_item = {
                        "text": potential_text,
                        "confidence": 1.0,
                        "color_info": None,
                        "matched_key": filename_matched_key
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

            else:

                img_data = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_data is None:
                    logger.error(f"无法读取图片: {img_path}")
                    continue

                initial_polys = None
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        initial_json_data = json.load(f)
                    initial_polys = initial_json_data.get('micro_poly', [])

                direction = None
                if 'S' in potential_text or 's' in potential_text:
                    direction = 'right'
                elif 'X' in potential_text:
                    direction = 'left'

                use_crop_ocr = False
                first_poly = None
                if direction and len(initial_polys[0]) != 2:
                    first_poly = initial_polys[0]
                else:
                    first_poly = initial_polys
                if first_poly is not None:
                    x_coords = [point[0] for point in first_poly]
                    textbox_length = max(x_coords) - min(x_coords)

                    if textbox_length > 80:
                        logger.info(f"文本框长度 {textbox_length:.2f} > 300，进行裁切检测: {filename}")
                        img_data = find_drak_remove(img_data)
                        debug_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_debug.jpg'))
                        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                        cropped_path = os.path.join(micro_img_dir, 'debug', filename.replace('.jpg', '_cropped.jpg'))
                        cv2.imwrite(cropped_path, img_data)
                        # vis_img = img.copy()
                        # cv2.polylines(vis_img, [expanded_array], isClosed=True, color=(0, 255, 0), thickness=2)
                        # cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        # cv2.imwrite(debug_path, vis_img)

                        use_adjusted, adjusted_edge, crop_ocr_result = adjust_textbox_edge(
                            img_data, first_poly, direction, json_path, crop_size=40
                        )
                        if use_adjusted and adjusted_edge and crop_ocr_result:
                            use_crop_ocr = True
                            logger.info(f"裁切检测成功，将使用裁切OCR结果")
                            crop_texts = crop_ocr_result.get('rec_texts', [])
                            crop_scores = crop_ocr_result.get('rec_scores', [])
                            if crop_texts:
                                if crop_scores and len(crop_scores) > 0:
                                    first_confidence = float(crop_scores[0])

                                for i, text in enumerate(crop_texts):
                                    score = float(crop_scores[i]) if i < len(crop_scores) else 0.0
                                    all_detected_texts.append(text)

                                    clean_text = text.strip()
                                    current_key = None
                                    found_match = False
                                    for key, variants in target_defs.items():
                                        for variant in variants:
                                            if variant in clean_text:
                                                current_key = key
                                                if key not in matched_keys:
                                                    matched_keys.append(key)
                                                found_match = True
                                                break
                                        if found_match:
                                            break

                                    detail_item = {
                                        "text": text,
                                        "confidence": score,
                                        "color_info": None,
                                        "matched_key": current_key
                                    }
                                    if current_key and (
                                            ("S" in current_key or "X" in current_key)):
                                        is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                                            img_path,
                                            current_key,
                                            debug=False,
                                            threshold=THRESHOLD
                                        )
                                        detail_item["template_match_res"] = template_match_res
                                        detail_item["template_match_score"] = is_match
                                        detail_item["black_pixel_count"] = black_pixel_count
                                        detail_item["black_radio"] = black_radio
                                        detail_item["template_name"] = "color_detection"
                                        detail_item["color_centers_separate"] = color_centers_separate
                                    detailed_results.append(detail_item)
                if not use_crop_ocr:
                    try:
                        result = ocr_engine.ocr.ocr(img_data, cls=True)
                    except TypeError:
                        result = ocr_engine.ocr.ocr(img_data)
                    except AttributeError:
                        try:
                            result = ocr_engine.ocr.predict(img_data)
                        except Exception as e:
                            logger.error(f"PaddleOCR 调用失败 {filename}: {e}")
                            continue
                    except Exception as e:
                        logger.error(f"PaddleOCR 调用失败 {filename}: {e}")
                        continue

                    # 解析结果
                    # result[0] 是第一张图片的结果（我们每次只传一张）

                    # PaddleX 或新版 PaddleOCR 返回的是一个字典对象（或包含字典的列表）
                    # 根据用户提供的结构，result[0] 是一个字典，包含 'rec_texts', 'rec_scores' 等字段

                    res_data = None
                    if isinstance(result, list) and len(result) > 0:
                        res_data = result[0]
                    elif isinstance(result, dict):  # 可能是单个字典
                        res_data = result

                    if res_data and isinstance(res_data, dict):
                        # 获取文本和置信度列表
                        rec_texts = res_data.get('rec_texts', [])
                        rec_scores = res_data.get('rec_scores', [])
                        # 假设 rec_polys 存在且与 rec_texts 对应
                        rec_polys = res_data.get('rec_polys', [])

                        if rec_texts:
                            # 记录第一个置信度作为参考
                            if rec_scores and len(rec_scores) > 0:
                                first_confidence = float(rec_scores[0])

                            for i, text in enumerate(rec_texts):
                                score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                                if score < 0.7:
                                    continue
                                all_detected_texts.append(text)

                                clean_text = text.strip()
                                # 遍历 target_defs 查找匹配
                                current_key = None
                                found_match = False
                                for key, variants in target_defs.items():
                                    for variant in variants:
                                        if variant in clean_text:
                                            current_key = key
                                            if key not in matched_keys:
                                                matched_keys.append(key)
                                            found_match = True
                                            break
                                    if found_match:
                                        break

                                # 记录详细结果
                                detail_item = {
                                    "text": text,
                                    "confidence": score,
                                    "color_info": None,
                                    "matched_key": current_key
                                }

                                if current_key and (
                                        ("S" in current_key or "X" in current_key)):
                                    is_match, black_pixel_count, template_match_res, color_centers_separate, black_radio = detect_colors(
                                        img_path,
                                        current_key,
                                        debug=False,
                                        threshold=THRESHOLD
                                    )
                                    detail_item["template_match_res"] = template_match_res
                                    detail_item["template_match_score"] = is_match
                                    detail_item["black_pixel_count"] = black_pixel_count
                                    detail_item["black_radio"] = black_radio
                                    detail_item["template_name"] = "color_detection"
                                    detail_item["color_centers_separate"] = color_centers_separate

                                detailed_results.append(detail_item)

            # 尝试加载 JSON 数据以获取颜色信息
            existing_color_presence = None
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
                if res.get("text") == "X":
                    print(1)
                # 检查并更新全局最佳结果
                m_key = res.get("matched_key")
                if m_key:
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

                        if compare_score - existing_score > 0.2:
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
                    json.dump(data, f, ensure_ascii=False, indent=2)

                count += 1
                # logger.debug(f"已更新: {filename} -> {all_detected_texts}, matched: {matched_keys}")
            else:
                # 如果没有 JSON，也可以选择创建一个，但按现有逻辑主要是补充信息
                # 既然已经识别了，可以考虑创建一个新的 json？
                # 暂时保持 pass，不强制创建文件，但上面的 best_results_map 已经更新了
                pass

        except Exception as e:
            logger.error(f"处理小窗口图片失败 {filename}: {e}")

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

            # 查找该文本对应的 matched_key (可能没有)
            # 这里的 text 是原始 OCR 文本，我们需要再次匹配或者从 item["matched_keys"] 推断
            # 但 item["matched_keys"] 是整张图的。为了准确对应，我们这里只输出整图匹配到的 keys
            # 或者，我们可以简单地将该行记录为一个检测结果

            # 根据需求：输出 matched_keys 及其对应的 confidence
            # 如果 details 里的 text 匹配到了 key，我们才输出？
            # 或者输出所有 details？
            # 用户需求："包含matched_keys及其对应的conference"

            # 我们重新进行一次简单的匹配检查，确定当前 text 对应哪个 key
            clean_text = text.strip()
            current_matched_key = ""

            # 加载 target definitions (为了避免重复加载，这里假设 process_micro_images 已经做过匹配)
            # 我们可以直接检查 item["matched_keys"]，但这无法将 key 和 confidence 一一对应
            # 所以我们需要再次匹配
            target_defs = load_target_definitions()
            for key, variants in target_defs.items():
                found_match = False
                for variant in variants:
                    if variant in clean_text:
                        current_matched_key = key
                        found_match = True
                        break
                if found_match:
                    break

            # 如果只想输出匹配到的结果：
            if current_matched_key:
                color_centers = detail.get("color_centers_separate", {})

                def format_centers(centers):
                    if not centers:
                        return ""
                    return ";".join([f"({c[0]},{c[1]})" for c in centers])

                yellow_length = len(color_centers.get("yellow", []))
                yl = 0
                gl = 0
                rl = 0
                if yellow_length == 0:
                    yl = 0
                elif yellow_length <= 4:
                    yl = 1
                elif yellow_length <= 7:
                    yl = 2
                if len(color_centers.get("green", [])) > 0:
                    gl = 1
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
                    "red_centers":rl,
                    "yellow_centers": yl,
                    "green_centers": gl,
                    "blue_centers": len(color_centers.get("blue", [])),
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
               "red_centers", "yellow_centers", "green_centers","blue_centers"] # "color_blue", "color_green", "color_yellow", "color_red",
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
        print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")

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
