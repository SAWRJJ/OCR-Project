import math

import cv2
import os
import json
import logging
import numpy as np
from ocr.utils import should_keep_text, safe_filename_component, has_text_or_number
from ocr.image_processor import ImageProcessor
from ocr.LW_detect import calculate_textbox_angle
import re

logger = logging.getLogger("ocr_system")
import copy
from ocr.ocr_engine import OCREngine

ocr_engine = OCREngine()

def has_other_colors1(
        img,
        white_thresh=200,
        black_thresh=50,
        min_ratio=0.001,
):
    """
    判断图片中是否存在非黑白颜色

    参数:
        img: BGR图像(np.ndarray)
        white_thresh: 白色阈值
        black_thresh: 黑色阈值
        min_ratio: 其他颜色最小占比

    返回:
        bool
    """

    if img is None or img.size == 0:
        return False

    # =========================
    # 1. 快速黑白过滤
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bw_mask = (
            (gray < black_thresh) |
            (gray > white_thresh)
    )

    # 全黑白直接返回
    if np.all(bw_mask):
        return False

    # =========================
    # 2. 提取非黑白区域
    # =========================
    other_mask = ~bw_mask

    # =========================
    # 3. 使用通道差判断是否有颜色
    # 灰色: RGB接近
    # 彩色: RGB差异明显
    # =========================
    pixels = img[other_mask]

    # 通道最大最小差
    color_diff = (
            np.max(pixels, axis=1).astype(np.int16) -
            np.min(pixels, axis=1).astype(np.int16)
    )

    # 差值越大越可能是彩色
    color_pixels = np.sum(color_diff > 20)

    # 占比判断
    ratio = color_pixels / (img.shape[0] * img.shape[1])

    return ratio > min_ratio
def detect_color_presence_bgr(
        img_bgr,
        min_ratio: float = 0.01,
        min_pixels: int = 50,
        s_thresh: int = 50,
        v_thresh: int = 50,
        boundary_width: int = 3,
        text: str = None,
):
    if img_bgr is None or img_bgr.size == 0:
        return {
            "presence": {"blue": False, "green": False, "yellow": False, "red": False},
            "stats": {},
            "valid_pixels": 0,
        }

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    img_h, img_w = img_bgr.shape[:2]

    boundary_mask = np.zeros((img_h, img_w), dtype=bool)
    boundary_mask[:boundary_width, :] = True
    boundary_mask[-boundary_width:, :] = True
    boundary_mask[:, :boundary_width] = True
    boundary_mask[:, -boundary_width:] = True

    # Valid Color: High Saturation, High Value
    valid_color = (s >= int(s_thresh)) & (v >= int(v_thresh))

    # Valid White: Low Saturation, High Value
    # Adjust thresholds as needed. S<30 and V>200 is a good starting point for white lights.
    valid_white = (s < 30) & (v > 200)

    # Total valid pixels (either color or white)
    valid = valid_color | valid_white
    valid_count = int(np.sum(valid))
    denom = float(valid_count) if valid_count > 0 else float(img_bgr.shape[0] * img_bgr.shape[1])

    def _mask_in_range(h_min, h_max):
        # Check hue and ensure it is considered a valid COLOR (high saturation)
        return (h >= int(h_min)) & (h <= int(h_max)) & valid_color

    masks = {
        "blue": _mask_in_range(90, 130),
        "green": _mask_in_range(35, 85),
        "yellow": _mask_in_range(20, 35),
        "red": ((_mask_in_range(0, 10) | _mask_in_range(170, 179)) & valid_color),
        "white": valid_white,
    }
    # if text == "SF":
    #     yellow_mask_vis = np.zeros_like(img_bgr)
    #     yellow_mask_vis[masks["yellow"]] = [0, 255, 255]
    #     save_path = f"/Users/saw/WorkSpace/work/OCR-Project/test/test6/debug_SF_{img_bgr.shape[0]}x{img_bgr.shape[1]}.jpg"
    #     cv2.imwrite(save_path, yellow_mask_vis)
    #     logger.info(f"已保存 SF debug 黄色掩码: {save_path}")

    presence = {}
    stats = {}
    positions = {}
    for name, m in masks.items():
        cnt = int(np.sum(m))
        ratio = float(cnt) / denom if denom > 0 else 0.0
        in_boundary = bool(np.any(m & boundary_mask)) if name != "white" else False
        presence[name] = ((cnt >= int(min_pixels)) or (ratio >= float(min_ratio))) and not in_boundary
        stats[name] = {"pixels": cnt, "ratio": ratio, "in_boundary": in_boundary}
        if cnt > 0:
            y_indices, x_indices = np.where(m)
            positions[name] = {
                "x_min": int(x_indices.min()),
                "x_max": int(x_indices.max()),
                "y_min": int(y_indices.min()),
                "y_max": int(y_indices.max()),
            }
        else:
            positions[name] = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}

    if text == "SF":
        logger.info(f"=== SF debug color positions ===")
        for name, pos in positions.items():
            logger.info(f"  {name}: x=[{pos['x_min']}, {pos['x_max']}], y=[{pos['y_min']}, {pos['y_max']}]")

    return {"presence": presence, "stats": stats, "valid_pixels": valid_count, "positions": positions}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Visualizer:

    @staticmethod
    def draw_red_lines(img_path, rec_polys_with_text, out_path="ocr_result_with_red_lines.jpg",
                       target_chars=None, exclude_substrings=None, save_boxes_dir=None):
        def calculate_centers_separate(mask, min_area=10, min_circularity=0.6, min_radius=7):
            """
            计算mask中每个独立连通域的中心坐标，并返回非圆形区域的mask
            """
            if cv2.countNonZero(mask) == 0:
                return [], np.zeros_like(mask)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            centers = []
            non_circular_mask = np.zeros_like(mask)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue

                component_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter == 0: continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                radius = np.sqrt(area / np.pi)

                if circularity <= min_circularity or radius <= min_radius:
                    cx = int(centroids[i][0])
                    cy = int(centroids[i][1])
                    centers.append((cx, cy))
                    non_circular_mask = cv2.bitwise_or(non_circular_mask, component_mask)

            return centers, non_circular_mask
        def calc_patch_bounds(vertical_expand_value):
            """
            计算patch区域
            """

            xs = []
            ys = []

            if top_ext:
                xs.extend([top_ext[0][0], top_ext[1][0]])
                ys.extend([top_ext[0][1], top_ext[1][1]])

            if bottom_ext:
                xs.extend([bottom_ext[0][0], bottom_ext[1][0]])
                ys.extend([bottom_ext[0][1], bottom_ext[1][1]])

            if angle_deg < 25:

                poly_np = np.array(poly)

                min_x = min(xs)
                max_x = max(xs)

                min_y = np.min(poly_np[:, 1])
                max_y = np.max(poly_np[:, 1])

                x0 = max(int(min_x), 0)
                x1 = min(int(max_x), raw_img.shape[1])

                y0 = max(int(min_y - vertical_expand_value), 0)
                y1 = min(int(max_y + vertical_expand_value), raw_img.shape[0])

            else:

                poly_np = np.array(poly)

                xs.extend(poly_np[:, 0].tolist())

                top_dx = top_ext[1][0] - top_ext[0][0]
                top_dy = top_ext[1][1] - top_ext[0][1]

                top_length = np.hypot(top_dx, top_dy)

                top_perp_x = -top_dy / top_length
                top_perp_y = top_dx / top_length

                bottom_dx = bottom_ext[1][0] - bottom_ext[0][0]
                bottom_dy = bottom_ext[1][1] - bottom_ext[0][1]

                bottom_length = np.hypot(bottom_dx, bottom_dy)

                bottom_perp_x = -bottom_dy / bottom_length
                bottom_perp_y = bottom_dx / bottom_length

                top_ext_np = np.array(top_ext)
                bottom_ext_np = np.array(bottom_ext)

                top_shift = np.array(
                    [top_perp_x, top_perp_y]
                ) * vertical_expand_value

                bottom_shift = np.array(
                    [bottom_perp_x, bottom_perp_y]
                ) * vertical_expand_value

                top_expand = top_ext_np + top_shift
                bottom_expand = bottom_ext_np + bottom_shift

                all_points = np.vstack([
                    poly_np,
                    top_expand,
                    bottom_expand
                ])

                x0 = max(int(np.min(all_points[:, 0])), 0)
                x1 = min(int(np.max(all_points[:, 0])), raw_img.shape[1])

                y0 = max(int(np.min(all_points[:, 1])), 0)
                y1 = min(int(np.max(all_points[:, 1])), raw_img.shape[0])

            return x0, y0, x1, y1
        if cv2 is None:
            raise ModuleNotFoundError("未安装 cv2（opencv-python）")

        img = cv2.imread(img_path)
        if img is None:
            logger.error("无法读取图像")
            return

        raw_img = copy.deepcopy(img)

        if target_chars is None:
            target_chars = ["S", "X", "D"]
        if exclude_substrings is None:
            exclude_substrings = ["/", "DG", "G", "SK"]

        if save_boxes_dir:
            os.makedirs(save_boxes_dir, exist_ok=True)

        saved_count = 0
        text_list = []
        split_info_path = os.path.join(os.path.dirname(img_path), "split_info.json")
        split_infos = []
        tt = ""
        if os.path.exists(split_info_path):
            import json
            with open(split_info_path, 'r') as f:
                split_infos = json.load(f)
            for info in split_infos:
                offset_x, offset_y = info["offset"]
                window = info.get("window", 1000)
                overlap = info.get("overlap", 300)
                x1 = offset_x
                y1 = offset_y
                x2 = offset_x + window
                y2 = offset_y + window
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, os.path.basename(info["path"]), (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for item in rec_polys_with_text:
            if len(item) == 5:
                poly, text, source_split, split_poly, score = item
                # if text == 'X_3a0_0':
                #     print(1)
            else:
                poly, text = item
                source_split = "unknown"
                split_poly = []

            if len(poly) != 4:
                continue
            textbox_angle, _ = calculate_textbox_angle(poly)
            textbox_angle = abs(np.degrees(textbox_angle))
            # if "S" in text:
            #     print(-1)
            if textbox_angle > 34 and len(text.replace(" ", "")) > 5 and has_text_or_number(text) and "DG" not in text:
                tt = text
                # print(text)
                text_list.append(source_split)
            text = text.replace("π", "II")
            text = text.replace("X", "X")
            # micro_0102_1700XL1_80_00.jpg
            # if "2300" in text and "404" in text: #'1700xL1-80-00O0'
            #     print(-1)
            if "." in text or (len(text) >= 4 and text[-1].isdigit()):

                poly_np = np.array(poly, dtype=np.int32)

                x, y, w, h = cv2.boundingRect(poly_np)

                roi = raw_img[y:y + h, x:x + w]

                # ROI mask
                roi_poly = poly_np - [x, y]

                roi_mask = np.zeros((h, w), dtype=np.uint8)

                cv2.fillPoly(roi_mask, [roi_poly], 255)

                # HSV only on ROI
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([20, 255, 255])

                lower_red2 = np.array([150, 50, 50])
                upper_red2 = np.array([180, 255, 255])

                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

                mask_red_hsv = cv2.bitwise_or(mask_red1, mask_red2)

                # BGR
                b = roi[:, :, 0].astype(np.int16)
                g = roi[:, :, 1].astype(np.int16)
                r = roi[:, :, 2].astype(np.int16)

                # HSV中的V通道（亮度）
                v = hsv[:, :, 2]

                # 放宽后的红色优势
                red_dom_mask = (
                        (r > g + 15) &
                        (r > b + 15) &
                        (r > 60) &
                        (v > 70)
                )

                red_dom_mask = red_dom_mask.astype(np.uint8) * 255

                # 最终mask
                mask_red = cv2.bitwise_and(
                    mask_red_hsv,
                    red_dom_mask
                )

                # 只保留poly内部
                mask_red = cv2.bitwise_and(mask_red, roi_mask)

                if cv2.countNonZero(mask_red) > 0:

                    red_centers, non_circular_mask = calculate_centers_separate(mask_red)

                    if cv2.countNonZero(non_circular_mask) > 0:
                        # 外扩1像素
                        kernel = np.ones((3, 3), np.uint8)
                        expanded_mask = cv2.dilate(
                            non_circular_mask,
                            kernel,
                            iterations=2
                        )
                        # 关键修改：
                        # 即使红色与黑色连通
                        # 也只处理真正红色区域
                        final_mask = cv2.bitwise_and(
                            expanded_mask,
                            mask_red
                        )
                        roi[final_mask > 0] = 255
        base_name = os.path.basename(out_path)
        name_without_ext = os.path.splitext(base_name)[0].split("_")[0]
        ext = os.path.splitext(base_name)[1]
        out_dir = os.path.dirname(out_path)
        no_red_out_path = os.path.join(out_dir, f"{name_without_ext}{ext}")
        cv2.imencode('.jpg', raw_img)[1].tofile(no_red_out_path)
        logger.info(f"去除红色像素后的原图已保存: {no_red_out_path}")
        restored_results = []
        if text_list:
            rotate_angle = 30  # 顺时针30度
            for split_img_path in set(text_list):
                split_img = cv2.imread(split_img_path)
                if split_img is None:
                    continue
                basename = os.path.basename(split_img_path)
                parts = basename.split("_")
                ox = int(parts[-2])
                oy = int(parts[-1].split(".")[0])
                h, w = split_img.shape[:2]
                # =========================
                # 1. 构建旋转矩阵（顺时针）
                # =========================
                center = (w / 2, h / 2)
                # OpenCV中正数是逆时针，所以这里用负数
                M = cv2.getRotationMatrix2D(center, -rotate_angle, 1.0)
                cos = abs(M[0, 0])
                sin = abs(M[0, 1])
                # 计算旋转后完整包围尺寸
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                # 平移补偿，保证图像完整
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                # =========================
                # 2. 旋转图片
                # =========================
                rotated_img = cv2.warpAffine(
                    split_img,
                    M,
                    (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderValue=(255, 255, 255)
                )
                # =========================
                # 3. OCR识别
                # =========================
                # 根据你当前工程替换成你的OCR接口
                ocr_results = ocr_engine.ocr.predict(rotated_img)
                # =========================
                # 4. 逆变换矩阵
                # =========================
                M_inv = cv2.invertAffineTransform(M)

                for result in ocr_results:
                    # 根据你的OCR返回结构调整
                    # 假设:
                    # result = (poly, text, score)
                    texts, polys, scores = result['rec_texts'], result['rec_polys'], result['rec_scores']
                    for i in range(len(texts)):
                        text = texts[i]
                        if len(text)>1 and has_text_or_number(text):
                            score = scores[i]
                            poly = polys[i]  # 这是一个 (4, 2) 的数组
                            # 将 poly 转换为齐次坐标形式 (N, 3)，即 [x, y, 1]
                            # poly.shape 是 (4, 2)
                            ones = np.ones((poly.shape[0], 1))
                            homogeneous_coords = np.hstack([poly, ones])  # 变成 (4, 3)
                            # 使用 M_inv 进行矩阵变换: (4, 3) dot (3, 3).T -> (4, 2)
                            # 计算公式: transformed_coords = homogeneous_coords @ M_inv.T
                            # 只取前两列 (x, y)
                            restored_poly_pts = (homogeneous_coords @ M_inv.T)[:, :2]
                            # 转换为列表格式并保存
                            offset_poly = [
                                [int(p[0] + ox), int(p[1] + oy)]
                                for p in restored_poly_pts
                            ]
                            restored_poly_pts = [
                                [int(p[0]), int(p[1])]
                                for p in restored_poly_pts
                            ]
                            restored_results.append((
                                offset_poly,
                                text,
                                split_img_path,
                                restored_poly_pts,
                                score
                            ))
                            poly = offset_poly
                            if "." in text or (len(text) >= 4 and text[-1].isdigit()):

                                poly_np = np.array(poly, dtype=np.int32)

                                x, y, w, h = cv2.boundingRect(poly_np)

                                roi = raw_img[y:y + h, x:x + w]

                                # ROI mask
                                roi_poly = poly_np - [x, y]

                                roi_mask = np.zeros((h, w), dtype=np.uint8)

                                cv2.fillPoly(roi_mask, [roi_poly], 255)

                                # HSV only on ROI
                                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                                lower_red1 = np.array([0, 50, 50])
                                upper_red1 = np.array([20, 255, 255])

                                lower_red2 = np.array([150, 50, 50])
                                upper_red2 = np.array([180, 255, 255])

                                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

                                mask_red_hsv = cv2.bitwise_or(mask_red1, mask_red2)

                                # BGR
                                b = roi[:, :, 0].astype(np.int16)
                                g = roi[:, :, 1].astype(np.int16)
                                r = roi[:, :, 2].astype(np.int16)

                                # HSV中的V通道（亮度）
                                v = hsv[:, :, 2]

                                # 放宽后的红色优势
                                red_dom_mask = (
                                        (r > g + 15) &
                                        (r > b + 15) &
                                        (r > 60) &
                                        (v > 70)
                                )

                                red_dom_mask = red_dom_mask.astype(np.uint8) * 255

                                # 最终mask
                                mask_red = cv2.bitwise_and(
                                    mask_red_hsv,
                                    red_dom_mask
                                )

                                # 只保留poly内部
                                mask_red = cv2.bitwise_and(mask_red, roi_mask)

                                if cv2.countNonZero(mask_red) > 0:

                                    red_centers, non_circular_mask = calculate_centers_separate(mask_red)

                                    if cv2.countNonZero(non_circular_mask) > 0:
                                        # 外扩1像素
                                        kernel = np.ones((3, 3), np.uint8)
                                        expanded_mask = cv2.dilate(
                                            non_circular_mask,
                                            kernel,
                                            iterations=2
                                        )
                                        # 关键修改：
                                        # 即使红色与黑色连通
                                        # 也只处理真正红色区域
                                        final_mask = cv2.bitwise_and(
                                            expanded_mask,
                                            mask_red
                                        )
                                        roi[final_mask > 0] = 255
                # restored_results
                # 即已经映射回原图坐标系的OCR结果
                logger.info(
                    f"旋转OCR完成: {split_img_path}, "
                    f"识别数量: {len(restored_results)}"
                )
        # print(len(text_list))
        rec_polys_with_text = rec_polys_with_text+restored_results
        for item in rec_polys_with_text:
            # Handle both 2-element (legacy) and 4-element (new) tuples
            if len(item) == 5:
                poly, text, source_split, split_poly, score = item
                # if text == 'X_3a0_0':
                #     print(1)
            else:
                poly, text = item
                source_split = "unknown"
                split_poly = []

            if len(poly) != 4:
                continue
            text = text.replace("π", "II")
            text = text.replace("X", "X")
            text = text.replace("×", "X")
            # if "S" in text:
            #     print(-1)
            # if "K" in text or "." in text:
            #     mask = np.zeros(img.shape[:2], dtype=np.uint8)
            #     cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
            #     is_white = np.all(img > 200, axis=2)
            #     for c in range(3):
            #         raw_img[:, :, c] = np.where(mask > 0, np.where(is_white, img[:, :, c], 0), img[:, :, c])
            #
            if not should_keep_text(text, target_chars=target_chars, exclude_substrings=exclude_substrings):
                continue
            text = text.upper()
            # 假设四边形的四个点顺序是：左上、右上、右下、左下
            top_line = [poly[0], poly[1]]
            bottom_line = [poly[3], poly[2]]

            box_width = max(p[0] for p in poly) - min(p[0] for p in poly)
            top_ext = ImageProcessor.choose_one_sided_extension(raw_img, top_line[0], top_line[1], extend_length=300,
                                                                min_non_bw_pixels=150, text=text)
            bottom_ext = ImageProcessor.choose_one_sided_extension(raw_img, bottom_line[0], bottom_line[1],
                                                                   extend_length=300, min_non_bw_pixels=150, text=text)

            top_ext = ImageProcessor.extend_opposite_side_for_small_box(top_line[0], top_line[1], top_ext, box_width)
            bottom_ext = ImageProcessor.extend_opposite_side_for_small_box(bottom_line[0], bottom_line[1], bottom_ext,
                                                                           box_width)

            if not top_ext and not bottom_ext:
                continue
            dx = poly[1][0] - poly[0][0]  # 10310 - 10227 = 83
            dy = poly[1][1] - poly[0][1]  # 2159 - 2202 = -43

            angle_rad = math.atan2(dy, dx)
            angle_deg = abs(math.degrees(angle_rad))
            if save_boxes_dir:


                # =========================
                # 第一次计算
                # =========================
                vertical_expand = 40

                x0, y0, x1, y1 = calc_patch_bounds(
                    vertical_expand
                )

                if x1 > x0 and y1 > y0:

                    patch = raw_img[y0:y1, x0:x1]

                    # =========================
                    # 如果没有颜色
                    # 重新扩大一次
                    # =========================
                    if not has_other_colors1(patch):

                        vertical_expand = 50

                        x0, y0, x1, y1 = calc_patch_bounds(
                            vertical_expand
                        )

                        if x1 > x0 and y1 > y0:
                            patch = raw_img[y0:y1, x0:x1]
                    if bool(re.search(r'[\u4e00-\u9fff]', str(text))):
                        continue

                    # 确保文件名安全，不包含中文或非法字符
                    name = safe_filename_component(str(text))
                    # 如果 name 中包含非 ASCII 字符，替换为 hex 或其他
                    # if not all(ord(c) < 128 for c in name):
                    #     name = "".join(c if ord(c) < 128 else f"_{ord(c):x}_" for c in name)

                    filename = f"micro_{saved_count:04d}_{name}.jpg" if name else f"micro_{saved_count:04d}.jpg"
                    # 使用 imencode/imdecode 处理中文路径，或者确保路径不含中文
                    # 这里文件名已经被我们处理过了，但是 save_boxes_dir 可能包含中文
                    # cv2.imwrite 在 Windows 下不支持中文路径
                    # 使用 cv2.imencode + file write 解决
                    output_path = os.path.join(save_boxes_dir, filename)
                    try:
                        cv2.imencode('.jpg', patch)[1].tofile(output_path)
                    except Exception as e:
                        logger.error(f"保存小图片失败: {output_path}, {e}")
                        continue
                    # if "X8" in text or "SF" in text:
                    #     print(top_ext)
                    #     print(bottom_ext)
                    color_info = detect_color_presence_bgr(patch, text=name)

                    # Calculate relative coordinates in the micro image
                    # The poly coordinates are global. We subtract x0, y0.
                    relative_poly = [[p[0] - x0, p[1] - y0] for p in poly]

                    patch_poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

                    info_data = {
                        "micro_image_name": filename,
                        "text": text,
                        "global_poly": poly,  # Global coordinates
                        "micro_poly": relative_poly,  # Coordinates in the micro image
                        "patch_poly": patch_poly,  # Patch bounding box coordinates
                        "source_split_image": os.path.basename(source_split),  # Which split image
                        "split_poly": split_poly,  # Coordinates in the split image
                        "color_presence": color_info["presence"],
                        "color_stats": color_info["stats"],
                        "color_valid_pixels": color_info["valid_pixels"],
                    }

                    # Save JSON for this specific image
                    json_filename = os.path.splitext(filename)[0] + ".json"
                    json_path = os.path.join(save_boxes_dir, json_filename)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(info_data, f, ensure_ascii=False, indent=2, cls=NpEncoder)

                    saved_count += 1

            # 绘制红线
            if top_ext:
                cv2.line(img, tuple(map(int, top_ext[0])), tuple(map(int, top_ext[1])), (0, 0, 255), 2)
            if bottom_ext:
                cv2.line(img, tuple(map(int, bottom_ext[0])), tuple(map(int, bottom_ext[1])), (0, 0, 255), 2)

        cv2.imwrite(out_path, img)
        logger.info(f"绘制红线后的结果已保存: {out_path}")
        if save_boxes_dir:
            logger.info(f"红线小方格及其对应的JSON文件已保存: {save_boxes_dir}，数量: {saved_count}")
        return no_red_out_path

    @staticmethod
    def visualize(img_path, results, out_path="ocr_result.jpg"):
        if cv2 is None:
            raise ModuleNotFoundError("未安装 cv2（opencv-python）")
        img = cv2.imread(img_path)
        if img is None:
            return

        for box, (text, _) in results:
            box = np.array(box, dtype=np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)
            cv2.putText(
                img,
                text[:10],
                tuple(box[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        cv2.imwrite(out_path, img)
        logger.info(f"可视化结果已保存: {out_path}")
