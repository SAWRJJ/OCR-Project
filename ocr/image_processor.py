import os
import cv2
import numpy as np
import logging
from ocr.find_boundary_dark import find_drak_remove
logger = logging.getLogger("ocr_system")
import os
import cv2
import numpy as np


def detect_and_whiten_color_with_connected_dark(
    img, dark_threshold=165, debug_name=None
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

    # 1. 第一步：平滑蓝色 mask
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

    # ==================== 核心战略修改：动态向外生长触碰 ====================
    # 放弃死板的 15px 固定长线。我们使用一个稍大的核进行膨胀，让颜色区域主动“长”出去触碰黑块。
    # 如果你的外圈黑块距离颜色区域真的很远，可以把 iterations 改大（比如 5 或 8）
    growth_kernel = np.ones((5, 5), np.uint8)
    dilated_color_mask = cv2.dilate(
        combined_color_mask.astype(np.uint8), growth_kernel, iterations=3
    )

    # 寻找颜色外延后，与黑色区域的交集（即触碰点）
    touching_mask = (dilated_color_mask > 0) & (black_mask > 0)
    # =====================================================================

    # 3. 第三步：连通域追踪。
    # 只要黑块的任何一个像素碰到了 `touching_mask`，整个黑块（连通域）就会被全部标记。
    num_labels, labels, stats_cc, centroids = (
        cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    )
    connected_black_mask = np.zeros_like(black_mask, dtype=bool)

    for i in range(1, num_labels):
        # 如果当前连通块的面积太小（比如小于5个像素的噪点），可以选择跳过
        if stats_cc[i, cv2.CC_STAT_AREA] < 5:
            continue

        component_mask = labels == i
        if np.any(component_mask & touching_mask):
            connected_black_mask |= component_mask

    # Debug 导出
    if debug_name is not None and np.any(blue_mask):
        debug_path = "splits/debug"
        os.makedirs(debug_path, exist_ok=True)
        blue_mask_vis = (blue_mask * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(debug_path, f"{debug_name}_blue_mask.jpg"),
            blue_mask_vis,
        )

    # 4. 第四步：最终图像涂白渲染
    result_img = img.copy()

    # 联合颜色区 + 顺藤摸瓜找到的所有连通黑色区，全部涂白
    final_white_mask = combined_color_mask | connected_black_mask
    result_img[final_white_mask] = [255, 255, 255]

    output_stats = {
        'blue': int(np.sum(color_masks.get("blue", 0))),
        'green': int(np.sum(color_masks['green'])),
        'yellow': int(np.sum(color_masks['yellow'])),
        'red': int(np.sum(color_masks['red'])),
        'dark': int(np.sum(black_mask)),
        'connected_dark': int(np.sum(connected_black_mask)),
    }

    return result_img, output_stats
class ImageProcessor:
    @staticmethod
    def split_image(img_path, window=1000, overlap=300):
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
        os.makedirs("splits", exist_ok=True)
        debug_path = "splits/debug"
        os.makedirs(debug_path, exist_ok=True)

        infos = []
        valid_paths = []
        step = window - overlap
        idx = 0

        img_size = os.path.getsize(img_path)
        for y in range(overlap, bordered_h, step):
            for x in range(overlap, bordered_w, step):
                x2 = min(x + window, bordered_w)
                y2 = min(y + window, bordered_h)
                orig_x, orig_y = x - overlap, y - overlap

                patch = bordered_img[y:y2, x:x2]
                debug_name = f"split_{idx}_{orig_x}_{orig_y}"
                if img_size < 450 * 1024:
                    patch, _ = detect_and_whiten_color_with_connected_dark(patch, debug_name=debug_name)
                h_patch, w_patch = patch.shape[:2]

                # --- 正常图保存 ---
                path = f"splits/split_{idx}_{orig_x}_{orig_y}.jpg"
                cv2.imwrite(path, patch)

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
        return infos, valid_paths

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
