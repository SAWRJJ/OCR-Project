import math

import cv2
import os
import json
import logging
import numpy as np
from .utils import should_keep_text, safe_filename_component
from .image_processor import ImageProcessor

logger = logging.getLogger("ocr_system")


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

    kernel = np.ones((3, 3), np.uint8)
    img_bgr = cv2.erode(img_bgr, kernel, iterations=1)
    img_bgr = cv2.dilate(img_bgr, kernel, iterations=1)

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
    if text == "D9" or text == "D15":
        yellow_mask_vis = np.zeros_like(img_bgr)
        yellow_mask_vis[masks["blue"]] = [0, 0, 255]
        save_path = f"/Users/saw/WorkSpace/work/OCR-Project/debug_output/debug_{text}.jpg"
        cv2.imwrite(save_path, yellow_mask_vis)
        logger.info(f"已保存 SF debug 黄色掩码: {save_path}")

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

    return {"presence": presence, "stats": stats, "valid_pixels": valid_count, "positions": positions}


class Visualizer:
    @staticmethod
    def draw_red_lines(img_path, rec_polys_with_text, out_path="ocr_result_with_red_lines.jpg",
                       target_chars=None, exclude_substrings=None, save_boxes_dir=None):
        if cv2 is None:
            raise ModuleNotFoundError("未安装 cv2（opencv-python）")

        img = cv2.imread(img_path)
        if img is None:
            logger.error("无法读取图像")
            return

        raw_img = img.copy()

        if target_chars is None:
            target_chars = ["S", "X", "D"]
        if exclude_substrings is None:
            exclude_substrings = ["/", "DG", "YD", "G", "Y"]

        if save_boxes_dir:
            os.makedirs(save_boxes_dir, exist_ok=True)

        saved_count = 0

        for item in rec_polys_with_text:
            # Handle both 2-element (legacy) and 4-element (new) tuples
            if len(item) == 5:
                poly, text, source_split, split_poly, score = item
                if text == 'X_3a0_0':
                    print(1)
            else:
                poly, text = item
                source_split = "unknown"
                split_poly = []

            if len(poly) != 4:
                continue

            if "X8" in text or 'SF' in text:
                print(0)
            text = text.replace("π","II")
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

            if "X8" in text:
                print(top_ext)
                print(bottom_ext)
            if not top_ext and not bottom_ext:
                continue
            dx = poly[1][0] - poly[0][0]  # 10310 - 10227 = 83
            dy = poly[1][1] - poly[0][1]  # 2159 - 2202 = -43

            angle_rad = math.atan2(dy, dx)
            angle_deg = abs(math.degrees(angle_rad))
            if save_boxes_dir:
                vertical_expand = 40
                xs = []
                ys = []
                if top_ext:
                    xs.extend([top_ext[0][0], top_ext[1][0]])
                    ys.extend([top_ext[0][1], top_ext[1][1]])
                if bottom_ext:
                    xs.extend([bottom_ext[0][0], bottom_ext[1][0]])
                    ys.extend([bottom_ext[0][1], bottom_ext[1][1]])
                for p in poly:
                    ys.append(p[1])
                if xs and angle_deg<25:
                    min_x = min(xs)
                    max_x = max(xs)

                    min_y = min(p[1] for p in poly)
                    max_y = max(p[1] for p in poly)
                    # min_y = min(ys)
                    # max_y = max(ys)

                    x0 = max(int(min_x), 0)
                    x1 = min(int(max_x), int(raw_img.shape[1]))
                    y0 = max(int(min_y) - vertical_expand, 0)
                    y1 = min(int(max_y) + vertical_expand, int(raw_img.shape[0]))
                else:
                    for p in poly:
                        xs.append(p[0])

                    min_x = min(xs)
                    max_x = max(xs)

                    top_dx = top_ext[1][0] - top_ext[0][0]
                    top_dy = top_ext[1][1] - top_ext[0][1]
                    top_length = np.sqrt(top_dx ** 2 + top_dy ** 2)
                    top_perp_x = -top_dy / top_length
                    top_perp_y = top_dx / top_length

                    bottom_dx = bottom_ext[1][0] - bottom_ext[0][0]
                    bottom_dy = bottom_ext[1][1] - bottom_ext[0][1]
                    bottom_length = np.sqrt(bottom_dx ** 2 + bottom_dy ** 2)
                    bottom_perp_x = -bottom_dy / bottom_length
                    bottom_perp_y = bottom_dx / bottom_length

                    print(f"top_ext 方向: ({top_dx:.2f}, {top_dy:.2f})")
                    print(f"top_ext 垂直方向: ({top_perp_x:.2f}, {top_perp_y:.2f})")
                    print(f"bottom_ext 方向: ({bottom_dx:.2f}, {bottom_dy:.2f})")
                    print(f"bottom_ext 垂直方向: ({bottom_perp_x:.2f}, {bottom_perp_y:.2f})")

                    poly_np = np.array(poly)
                    poly_min_y = np.min(poly_np[:, 1])
                    poly_max_y = np.max(poly_np[:, 1])
                    poly_min_x = np.min(poly_np[:, 0])
                    poly_max_x = np.max(poly_np[:, 0])

                    top_ext_start = np.array(top_ext[0])
                    top_ext_end = np.array(top_ext[1])
                    bottom_ext_start = np.array(bottom_ext[0])
                    bottom_ext_end = np.array(bottom_ext[1])

                    top_ext_start_perp = top_ext_start + np.array([top_perp_x, top_perp_y]) * vertical_expand
                    top_ext_end_perp = top_ext_end + np.array([top_perp_x, top_perp_y]) * vertical_expand
                    bottom_ext_start_perp = bottom_ext_start + np.array(
                        [bottom_perp_x, bottom_perp_y]) * vertical_expand
                    bottom_ext_end_perp = bottom_ext_end + np.array([bottom_perp_x, bottom_perp_y]) * vertical_expand

                    all_x = [poly_min_x, poly_max_x,
                             top_ext_start_perp[0], top_ext_end_perp[0],
                             bottom_ext_start_perp[0], bottom_ext_end_perp[0]]
                    all_y = [poly_min_y, poly_max_y,
                             top_ext_start_perp[1], top_ext_end_perp[1],
                             bottom_ext_start_perp[1], bottom_ext_end_perp[1]]

                    exp_min_x = min(all_x)
                    exp_max_x = max(all_x)
                    exp_min_y = min(all_y)
                    exp_max_y = max(all_y)

                    x0 = max(int(exp_min_x), 0)
                    x1 = min(int(exp_max_x), img.shape[1])
                    y0 = max(int(exp_min_y), 0)
                    y1 = min(int(exp_max_y), img.shape[0])
                if x1 > x0 and y1 > y0:
                        patch = raw_img[y0:y1, x0:x1]

                        # 确保文件名安全，不包含中文或非法字符
                        name = safe_filename_component(str(text))
                        # 如果 name 中包含非 ASCII 字符，替换为 hex 或其他
                        if not all(ord(c) < 128 for c in name):
                            name = "".join(c if ord(c) < 128 else f"_{ord(c):x}_" for c in name)

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
                        if "X8" in text or "SF" in text:
                            print(top_ext)
                            print(bottom_ext)
                        color_info = detect_color_presence_bgr(patch, text=name)

                        # Calculate relative coordinates in the micro image
                        # The poly coordinates are global. We subtract x0, y0.
                        relative_poly = [[p[0] - x0, p[1] - y0] for p in poly]

                        info_data = {
                            "micro_image_name": filename,
                            "text": text,
                            "global_poly": poly,  # Global coordinates
                            "micro_poly": relative_poly,  # Coordinates in the micro image
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
                            json.dump(info_data, f, ensure_ascii=False, indent=2)

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
