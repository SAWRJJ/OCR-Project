import json
import math
import os
import re

import cv2
import numpy as np
import logging
from ocr.find_boundary_dark import find_drak_remove
logger = logging.getLogger("ocr_system")
import os
import cv2
import numpy as np
# from ocr.utils import is_polygon_center_close
from ocr.ocr_engine import OCREngine

def detect_and_whiten_color_with_connected_dark(
    img, dark_threshold=165, debug_name=None, debug_base_path=None
):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    valid_color = (s >= 50) & (v >= 50)

    color_ranges = {
        'blue': (90, 130),
        'green': (35, 85),
        'yellow': (20, 35),
        'red': [(0, 10), (170, 179)],
    }

    combined_color_mask = np.zeros_like(valid_color, dtype=bool)
    color_masks = {}

    for color_name, h_range in color_ranges.items():
        if isinstance(h_range, list):
            mask = np.zeros_like(valid_color, dtype=bool)
            for h_min, h_max in h_range:
                mask |= (h >= h_min) & (h <= h_max) & valid_color
        else:
            h_min, h_max = h_range
            mask = (h >= h_min) & (h <= h_max) & valid_color

        color_masks[color_name] = mask
        combined_color_mask |= mask

    # 1. 第一步：优先提取并平滑蓝色 mask
    blue_mask = color_masks.get('blue', np.zeros_like(valid_color))
    if np.any(blue_mask):
        kernel = np.ones((3, 3), np.uint8)
        blue_mask_uint8 = cv2.erode(
            blue_mask.astype(np.uint8), kernel, iterations=1
        )
        blue_mask_uint8 = cv2.dilate(blue_mask_uint8, kernel, iterations=1)
        blue_mask = blue_mask_uint8.astype(bool)
        color_masks['blue'] = blue_mask

        # 重新更新总颜色面具
        combined_color_mask = np.zeros_like(valid_color, dtype=bool)
        for m in color_masks.values():
            combined_color_mask |= m.astype(bool)

    # 2. 第二步：提取基础的暗色/黑色区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < dark_threshold).astype(np.uint8)

    # ==================== 核心修改：记录所有线段用于 Debug 绘图 ====================
    bridge_lines = []  # 新增：用来存储所有画好的线段坐标

    if np.any(blue_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            blue_mask.astype(np.uint8), connectivity=8
        )

        h_img, w_img = img.shape[:2]
        all_lines_mask = np.zeros_like(black_mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 4:
                continue

            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])

            pt1 = (center_x, np.clip(center_y - 18, 0, h_img - 1))
            pt2 = (center_x, np.clip(center_y + 18, 0, h_img - 1))

            # 存储线段供后续 debug 使用
            bridge_lines.append((pt1, pt2))

            cv2.line(black_mask, pt1, pt2, 255, thickness=3)
            cv2.line(all_lines_mask, pt1, pt2, 255, thickness=3)

        combined_color_mask |= all_lines_mask > 0
    # =====================================================================

    # 3. 第三步：让颜色与黑色面具发生碰撞
    kernel = np.ones((3, 3), np.uint8)
    dilated_color_mask = cv2.dilate(
        combined_color_mask.astype(np.uint8), kernel, iterations=2
    )

    touching_mask = (dilated_color_mask > 0) & (black_mask > 0)

    # 4. 第四步：连通域追踪
    num_labels_black, labels_black, stats_cc, centroids_black = (
        cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    )
    connected_black_mask = np.zeros_like(black_mask, dtype=bool)

    for i in range(1, num_labels_black):
        component_mask = labels_black == i
        if np.any(component_mask & touching_mask):
            connected_black_mask |= component_mask

    # ==================== 新增 Debug 导出：保存画了黑线的原图 ====================
    if debug_name is not None and debug_base_path is not None:
        debug_path = debug_base_path
        os.makedirs(debug_path, exist_ok=True)

        # 1. 保存原有的蓝色 Mask
        if np.any(blue_mask):
            blue_mask_vis = (blue_mask * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(debug_path, f"{debug_name}_blue_mask.jpg"),
                blue_mask_vis,
            )

        # 2. 核心要求：在原图上画出所有搭桥线并保存
        if len(bridge_lines) > 0:
            line_debug_img = img.copy()
            for pt1, pt2 in bridge_lines:
                # 用纯黑色 (0, 0, 0) 粗度为 3 画线。
                # 提示：如果你觉得黑色在暗色背景里看不清，可以换成鲜艳的红色 (0, 0, 255)
                cv2.line(line_debug_img, pt1, pt2, (0, 0, 0), thickness=3)

            cv2.imwrite(
                os.path.join(debug_path, f"{debug_name}_with_lines.jpg"),
                line_debug_img,
            )
    # =====================================================================

    # 5. 第五步：最终图像涂白渲染
    result_img = img.copy()

    # 先把所有独立颜色区域涂白
    for color_name, mask in color_masks.items():
        if np.sum(mask) > 0:
            result_img[mask.astype(bool)] = [255, 255, 255]

    # 再把总颜色区以及顺藤摸瓜找到的连通黑色区全部涂白
    combined_mask = combined_color_mask | connected_black_mask
    result_img[combined_mask] = [255, 255, 255]

    output_stats = {
        'blue': int(np.sum(color_masks.get("blue", 0))),
        'green': int(np.sum(color_masks['green'])),
        'yellow': int(np.sum(color_masks['yellow'])),
        'red': int(np.sum(color_masks['red'])),
        'dark': int(np.sum(black_mask)),
        'connected_dark': int(np.sum(connected_black_mask)),
    }

    return result_img, output_stats

# 1. 递归转换函数：防止 Numpy 数据导致 JSON 序列化失败
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int16, np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    return obj


def is_polygon_center_close(poly1, poly2, distance_threshold=4.0):
    p1 = np.array(poly1, dtype=np.float32)
    p2 = np.array(poly2, dtype=np.float32)
    center1 = np.mean(p1, axis=0)
    center2 = np.mean(p2, axis=0)
    distance = math.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )
    return distance < distance_threshold

class ImageProcessor:
    @staticmethod
    def filter_ocr_results(first_ocr_res):
        """筛选 OCR 结果，去除空的和全是中文的结果
        Args:
            first_ocr_res: OCR 引擎返回的原始结果列表
        Returns:
            list: 筛选后的 OCR 结果列表
        """
        if not first_ocr_res:
            return []
        filtered_results = []
        for res in first_ocr_res:
            # 如果结果为空，跳过
            if not res:
                continue
            # 检查是否有有效的文本内容
            has_valid_text = False
            # PaddleOCR 结果可能包含 'rec_texts' 字段或其他结构
            if isinstance(res, dict):
                # 检查 rec_texts 字段
                if 'rec_texts' in res and res['rec_texts']:
                    for text_item in res['rec_texts']:
                        text = str(text_item) if text_item else ""
                        # 检查是否包含非中文字符（字母、数字、符号等）
                        if text and not ImageProcessor._is_all_chinese(text):
                            has_valid_text = True
                            break
            elif isinstance(res, list):
                # 如果结果是列表格式 [[box, (text, conf)], ...]
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        box, info = item[0], item[1]
                        if isinstance(info, (list, tuple)) and len(info) >= 1:
                            text = str(info[0]) if info[0] else ""
                            if text and not ImageProcessor._is_all_chinese(text):
                                has_valid_text = True
                                break
            # 只有包含有效文本的结果才保留
            if has_valid_text:
                filtered_results.append(res)
        logger.info(f"OCR 结果筛选: 原始 {len(first_ocr_res)} 个，筛选后 {len(filtered_results)} 个")
        return filtered_results
    
    @staticmethod
    def _is_all_chinese(text):
        """判断文本是否全是中文字符
        
        Args:
            text: 待判断的文本字符串
            
        Returns:
            bool: 如果文本全是中文则返回 True，否则返回 False
        """
        if not text or not text.strip():
            return True
        
        # 匹配中文字符的正则表达式
        chinese_pattern = re.compile(r'^[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+$')
        
        # 如果完全匹配中文正则，说明全是中文
        return bool(chinese_pattern.match(text.strip()))

    @staticmethod
    def split_image(img_path, window=1000, overlap=300,remove_color=False,clean_img = False):
        img = cv2.imread(img_path)
        if img is None:
            logger.error("无法读取图像")
            return [], []

        h, w = img.shape[:2]
        bordered_img = cv2.copyMakeBorder(
            img, overlap, overlap, overlap, overlap,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        bordered_h, bordered_w = bordered_img.shape[:2]
        img_base_name = os.path.splitext(os.path.basename(img_path))[0]
        splits_img_dir = os.path.join("splits", img_base_name)
        os.makedirs(splits_img_dir, exist_ok=True)
        debug_path = os.path.join(splits_img_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)
        clean_path = os.path.join(splits_img_dir, "clean")
        os.makedirs(clean_path, exist_ok=True)

        infos = []
        valid_paths = []
        step = window - overlap
        idx = 0

        img_size = os.path.getsize(img_path)
        for y in range(overlap, bordered_h, step):
            is_clean=False
            for x in range(overlap, bordered_w, step):
                x2 = min(x + window, bordered_w)
                y2 = min(y + window, bordered_h)
                orig_x, orig_y = x - overlap, y - overlap

                patch = bordered_img[y:y2, x:x2]
                debug_name = f"split_{idx}_{orig_x}_{orig_y}"
                if img_size < 450 * 1024 or remove_color:
                    is_clean = True
                    clean_img = False
                    patch, _ = detect_and_whiten_color_with_connected_dark(patch, debug_name=debug_name, debug_base_path=debug_path)
                    # patch = find_drak_remove(patch, not_save_boundary=True, save_circle=False,
                    #                           remove_light_white=True)
                h_patch, w_patch = patch.shape[:2]

                # --- 正常图保存 ---
                path = os.path.join(splits_img_dir, f"split_{idx}_{orig_x}_{orig_y}.jpg")
                cv2.imwrite(path, patch)
                if clean_img and not is_clean:
                    patch0, _ = detect_and_whiten_color_with_connected_dark(patch, debug_name=debug_name,
                                                                           debug_base_path=debug_path)
                    # patch0 = find_drak_remove(patch0,dark_threshold=190,not_save_boundary=True,save_circle=False,remove_light_white=True)
                    clean_path0 = os.path.join(clean_path, f"split_{idx}_{orig_x}_{orig_y}.jpg")
                    # print(clean_path0)
                    cv2.imwrite(clean_path0, patch0)


                # # --- 旋转图保存 ---
                # center_patch = (w_patch // 2, h_patch // 2)
                # angle_patch = 30
                # M = cv2.getRotationMatrix2D(center_patch, angle_patch, 1)
                # cos = np.abs(M[0, 0])
                # sin = np.abs(M[0, 1])
                # new_w = int(h_patch * sin + w_patch * cos)
                # new_h = int(h_patch * cos + w_patch * sin)
                # M[0, 2] += (new_w - w_patch) / 2
                # M[1, 2] += (new_h - h_patch) / 2
                #
                # rotated_patch = cv2.warpAffine(patch, M, (new_w, new_h))
                # rotated_path = f"splits/split_{idx}_{orig_x}_{orig_y}_rotated.jpg"
                # cv2.imwrite(rotated_path, rotated_patch)

                is_white = np.all(patch >= 250)
                if not is_white:
                    # 将图片转为 HSV
                    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv_patch)

                    # --- 判断 1: 是否无彩色 ---
                    # S (饱和度) 低于一定阈值（比如 30）通常认为是灰度色
                    # np.max(s) < 30 表示整张图没有任何一个像素有明显的颜色
                    is_grayscale = np.max(s) < 30

                    # --- 判断 2: 黑色像素占比 ---
                    # V (亮度) 低于一定阈值（比如 50）认为是黑色
                    black_mask = v < 50
                    black_pixel_count = np.sum(black_mask)
                    total_pixels = patch.shape[0] * patch.shape[1]
                    black_ratio = black_pixel_count / total_pixels

                    # --- 综合判断 ---
                    # 如果无彩色 且 黑色占比超过一半 (0.5)
                    if is_grayscale and black_ratio > 0.5:
                        is_white = True  # 这里借用变量名，意为“跳过该图”或“标记为无效图”

                infos.append({
                    "path": path,
                    "offset": (orig_x, orig_y)
                })

                if not is_white:
                    valid_paths.append(path)
                idx += 1

        split_info_path = os.path.join(os.path.dirname(img_path), "split_info.json")
        import json
        with open(split_info_path, 'w') as f:
            json.dump(infos, f)

        logger.info(f"图像切分完成，共 {len(infos)*2} 张，其中非白图 {len(valid_paths)} 张")
        return infos, valid_paths,clean_img

    @staticmethod
    def ocr_again(ocr_list,image_base_name):
        ocr_engine = OCREngine()

        for ocr_res in ocr_list:
            input_path = ocr_res["input_path"]
            if not input_path:
                continue

            # 解析路径：splits/t15/split_0_0_0.jpg
            dirname, filename = os.path.split(input_path)
            base_name, _ = os.path.splitext(filename)  # 得到 'split_0_0_0'

            # 1. 读取用于二次识别的 clean 图像路径
            input_path0 = os.path.join(dirname, "clean", filename)

            if not os.path.exists(input_path0):
                print(f"警告: 找不到清洗后的图片 {input_path0}，跳过。")
                continue

            # 2. 调用 PaddleX 执行二次预测
            img = cv2.imread(input_path0)
            results = ocr_engine.ocr.predict(img)

            # 3. 解析并合并新结果
            for result in results:
                res_dict = (
                    result if isinstance(result, dict) else result.get_to_dict()
                )

                if "rec_texts" not in res_dict or not res_dict["rec_texts"]:
                    continue

                new_texts = res_dict["rec_texts"]
                new_scores = res_dict["rec_scores"]

                new_polys = [
                    poly.tolist() if isinstance(poly, np.ndarray) else poly
                    for poly in res_dict["rec_polys"]
                ]
                new_dt_polys = [
                    poly.tolist() if isinstance(poly, np.ndarray) else poly
                    for poly in res_dict["dt_polys"]
                ]

                # 比对中心点距离，追加缺失框
                for idx, new_poly in enumerate(new_polys):
                    is_already_exist = False
                    if new_texts[idx] == "D5":
                        print(-1)
                    for old_poly in ocr_res["rec_polys"]:
                        if is_polygon_center_close(
                                new_poly, old_poly, distance_threshold=10.0
                        ):
                            is_already_exist = True
                            break

                    if not is_already_exist:
                        # print(
                        #     f"[{filename}] 补回漏检文本: {new_texts[idx]}"
                        # )
                        if new_texts[idx] == "0" or new_texts[idx] == "":
                            continue
                        ocr_res["rec_polys"].append(new_poly)
                        ocr_res["dt_polys"].append(new_dt_polys[idx])
                        ocr_res["rec_texts"].append(new_texts[idx])
                        ocr_res["rec_scores"].append(new_scores[idx])

                        # 动态补齐外接矩形框 [xmin, ymin, xmax, ymax]
                        poly_np = np.array(new_poly)
                        xmin = int(np.min(poly_np[:, 0]))
                        ymin = int(np.min(poly_np[:, 1]))
                        xmax = int(np.max(poly_np[:, 0]))
                        ymax = int(np.max(poly_np[:, 1]))
                        ocr_res["rec_boxes"].append([xmin, ymin, xmax, ymax])

            # ==================== 核心修改：写回对应的 JSON 文件 ====================
            # 4. 构造输出目标路径：output/t1/split_0_0_0_res.json
            # 这里的 'output/t1' 会根据你图片的层级自动匹配（如果固定是 t1，也可以硬编码）
            output_dir = os.path.join("output", image_base_name)
            os.makedirs(output_dir, exist_ok=True)  # 确保输出文件夹存在

            json_out_path = os.path.join(output_dir, f"{base_name}_res.json")

            # 5. 清理多余数据并执行安全格式化转换
            # 移除不可导出的字体类对象、或者包含图像矩阵的大字段防止 JSON 体积爆炸
            export_data = ocr_res.copy()
            if "vis_fonts" in export_data:
                export_data.pop("vis_fonts")
            if "doc_preprocessor_res" in export_data:
                export_data.pop("doc_preprocessor_res")

            cleaned_data = convert_numpy_to_list(export_data)

            # 6. 正式写入文件
            with open(json_out_path, "w", encoding="utf-8") as f:
                # indent=4 保持优美的缩进格式，ensure_ascii=False 确保中文正常不乱码
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            #
            # print(f"成功将更新结果写回至: {json_out_path}")
            # =====================================================================

        return ocr_list
    @staticmethod
    def count_non_bw_pixels_along_line(img, p1, p2, line_thickness=3, white_thresh=245, black_thresh=10, mask=None):
        if img is None or img.size == 0:
            return 0

        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, int(line_thickness))

        pixels = img[mask == 255]
        if pixels.size == 0:
            return 0

        white_mask = np.all(pixels >= int(white_thresh), axis=1)
        black_mask = np.all(pixels <= int(black_thresh), axis=1)
        # 灰色判断
        gray_thresh = 20

        gray_mask = (np.max(pixels, axis=1) - np.min(pixels, axis=1)
                    ) <= gray_thresh
        non_bw_count = int(np.sum(~(white_mask | black_mask | gray_mask)))
        debug_output = None
        if debug_output is not None:
            os.makedirs(debug_output, exist_ok=True)
            vis_img = img.copy()

            white_vis = np.zeros((h, w), dtype=np.uint8)
            black_vis = np.zeros((h, w), dtype=np.uint8)
            non_bw_vis = np.zeros((h, w), dtype=np.uint8)

            mask_indices = np.where(mask == 255)
            for idx in range(len(mask_indices[0])):
                y, x = mask_indices[0][idx], mask_indices[1][idx]
                if white_mask[idx]:
                    white_vis[y, x] = 255
                elif black_mask[idx]:
                    black_vis[y, x] = 255
                elif not (white_mask[idx] or black_mask[idx]):
                    non_bw_vis[y, x] = 255

            white_overlay = vis_img.copy()
            white_overlay[white_vis == 255] = [255, 255, 0]
            black_overlay = white_overlay.copy()
            black_overlay[black_vis == 255] = [0, 0, 255]
            final_overlay = black_overlay.copy()
            final_overlay[non_bw_vis == 255] = [0, 255, 0]

            cv2.imwrite(os.path.join(debug_output, "white_black_overlay.png"), final_overlay)
            cv2.imwrite(os.path.join(debug_output, "white_mask.png"), white_vis)
            cv2.imwrite(os.path.join(debug_output, "black_mask.png"), black_vis)
            cv2.imwrite(os.path.join(debug_output, "non_bw_mask.png"), non_bw_vis)

        return non_bw_count

    @staticmethod
    def choose_one_sided_extension(img, p1, p2, extend_length=220, line_thickness=3, white_thresh=245, black_thresh=5, min_non_bw_pixels=5, text=None):
        """选择线条的单侧延伸方向
        
        根据两侧延伸区域中的非黑白像素数量，决定线条应该向哪个方向延伸。
        返回延伸后的新端点，如果两侧都没有内容则返回None。
        
        Args:
            img: 输入图像
            p1, p2: 线段的两端点坐标
            extend_length: 延伸长度（像素）
            line_thickness: 线条粗细
            white_thresh: 白色阈值，高于此值视为白色
            black_thresh: 黑色阈值，低于此值视为黑色
            min_non_bw_pixels: 最少非黑白像素数阈值
            text: 输入文本，用于在两侧都无内容时判断延伸方向
            
        Returns:
            延伸后的新端点坐标对，或None
        """
        if img is None or img.size == 0:
            return None

        # 调试输出目录
        debug_dir = "debug_output"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # 计算线段的方向向量
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = float(np.sqrt(dx * dx + dy * dy))
        if length <= 1e-6:
            return None

        # 归一化方向向量
        ux = dx / length
        uy = dy / length

        h, w = img.shape[:2]
        # 创建原始线段的掩码
        base_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(base_mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, int(line_thickness))

        # 延伸方向1：从p1向相反方向延伸
        p1_ext = [p1[0] - ux * extend_length, p1[1] - uy * extend_length]
        cand1_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cand1_mask, (int(p1_ext[0]), int(p1_ext[1])), (int(p2[0]), int(p2[1])), 255, int(line_thickness))
        # 只保留延伸部分的掩码（排除原始线段）
        ext1_mask = cv2.bitwise_and(cand1_mask, cv2.bitwise_not(base_mask))

        if text == "XFS II":
            # 可视化：保存ext1_mask并在原图上显示
            cv2.imwrite(os.path.join(debug_dir, "ext1_mask.png"), ext1_mask)
            img_ext1_vis = img.copy()
            if len(img_ext1_vis.shape) == 2:
                img_ext1_vis = cv2.cvtColor(img_ext1_vis, cv2.COLOR_GRAY2BGR)
            img_ext1_vis[ext1_mask > 0] = [0, 255, 255]
            cv2.imwrite(os.path.join(debug_dir, "ext1_mask_overlay.png"), img_ext1_vis)
        
        # 统计延伸区域1中的总像素数和非黑白像素数量
        ext1_total = int(np.sum(ext1_mask > 0))
        ext1_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext1_mask
        )

        # 延伸方向2：从p2向相同方向延伸
        p2_ext = [p2[0] + ux * extend_length, p2[1] + uy * extend_length]
        cand2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cand2_mask, (int(p1[0]), int(p1[1])), (int(p2_ext[0]), int(p2_ext[1])), 255, int(line_thickness))
        # 只保留延伸部分的掩码（排除原始线段）
        ext2_mask = cv2.bitwise_and(cand2_mask, cv2.bitwise_not(base_mask))
        
        # # 可视化：保存ext2_mask并在原图上显示
        # cv2.imwrite(os.path.join(debug_dir, "ext2_mask.png"), ext2_mask)
        # img_ext2_vis = img.copy()
        # if len(img_ext2_vis.shape) == 2:
        #     img_ext2_vis = cv2.cvtColor(img_ext2_vis, cv2.COLOR_GRAY2BGR)
        # img_ext2_vis[ext2_mask > 0] = [0, 255, 255]
        # cv2.imwrite(os.path.join(debug_dir, "ext2_mask_overlay.png"), img_ext2_vis)
        
        # 统计延伸区域2中的总像素数和非黑白像素数量
        ext2_total = int(np.sum(ext2_mask > 0))
        ext2_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext2_mask
        )
        if text and "X" in text.upper() and 'S' in text.upper():
            return p1_ext, p2_ext
        if text and 'X' in text.upper():
            return p1, p2_ext
        elif text and 'S' in text.upper():
            return p1_ext, p2
        # 如果两侧都没有内容，根据输入文本判断延伸方向
        if ext1_count < int(min_non_bw_pixels) and ext2_count < int(min_non_bw_pixels):
            # 文本包含X则向右延伸，包含S则向左延伸
            if text and 'X' in text.upper():
                return p1, p2_ext
            elif text and 'S' in text.upper():
                return p1_ext, p2
            # return None

        # 返回非黑白像素较多的那个延伸方向
        if ext2_count > ext1_count:
            return p1, p2_ext
        elif ext2_count<ext1_count:
            return p1_ext, p2
        else:
            return p1_ext, p2_ext

    @staticmethod
    def extend_opposite_side_for_small_box(p1, p2, seg, box_width, width_threshold=100, opposite_extend=70):
        if not seg:
            return seg
        if box_width is None or float(box_width) >= float(width_threshold):
            return seg

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = float(np.sqrt(dx * dx + dy * dy))
        if length <= 1e-6:
            return seg

        ux = dx / length
        uy = dy / length

        s0, s1 = seg

        if abs(float(s0[0]) - float(p1[0])) < 1e-3 and abs(float(s0[1]) - float(p1[1])) < 1e-3:
            return [p1[0] - ux * opposite_extend, p1[1] - uy * opposite_extend], s1
        if abs(float(s1[0]) - float(p2[0])) < 1e-3 and abs(float(s1[1]) - float(p2[1])) < 1e-3:
            return s0, [p2[0] + ux * opposite_extend, p2[1] + uy * opposite_extend]

        d0 = (float(s0[0]) - float(p1[0])) ** 2 + (float(s0[1]) - float(p1[1])) ** 2
        d1 = (float(s1[0]) - float(p2[0])) ** 2 + (float(s1[1]) - float(p2[1])) ** 2
        if d0 <= d1:
            return [p1[0] - ux * opposite_extend, p1[1] - uy * opposite_extend], s1
        return s0, [p2[0] + ux * opposite_extend, p2[1] + uy * opposite_extend]
