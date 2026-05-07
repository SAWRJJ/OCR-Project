import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("ocr_system")

class ImageProcessor:
    @staticmethod
    def split_image(img_path, window=1000, overlap=300):
        if cv2 is None:
            raise ModuleNotFoundError("未安装 cv2（opencv-python）")
            
        img = cv2.imread(img_path)
        if img is None:
            logger.error("无法读取图像")
            return [], []

        h, w = img.shape[:2]
        
        # Add border
        bordered_img = cv2.copyMakeBorder(
            img,
            overlap, overlap, overlap, overlap,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        bordered_h, bordered_w = bordered_img.shape[:2]
        os.makedirs("splits", exist_ok=True)
        
        infos = []
        valid_paths = []
        step = window - overlap
        idx = 0
        
        for y in range(overlap, bordered_h, step):
            for x in range(overlap, bordered_w, step):
                x2 = min(x + window, bordered_w)
                y2 = min(y + window, bordered_h)
                
                orig_x = x - overlap
                orig_y = y - overlap
                
                patch = bordered_img[y:y2, x:x2]
                path = f"splits/split_{idx}_{orig_x}_{orig_y}.jpg"
                cv2.imwrite(path, patch)
                
                is_white = np.all(patch >= 250)
                
                infos.append({
                    "path": path,
                    "offset": (orig_x, orig_y)
                })
                
                if not is_white:
                    valid_paths.append(path)
                
                idx += 1
                
        logger.info(f"图像切分完成，共 {len(infos)} 张，其中非白图 {len(valid_paths)} 张")
        return infos, valid_paths

    @staticmethod
    def count_non_bw_pixels_along_line(img, p1, p2, line_thickness=3, white_thresh=245, black_thresh=10, mask=None, debug_output=None):
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

        if text == "D48":
            # 可视化：保存ext1_mask并在原图上显示
            cv2.imwrite(os.path.join(debug_dir, "ext1_mask.png"), ext1_mask)
            img_ext1_vis = img.copy()
            if len(img_ext1_vis.shape) == 2:
                img_ext1_vis = cv2.cvtColor(img_ext1_vis, cv2.COLOR_GRAY2BGR)
            img_ext1_vis[ext1_mask > 0] = [0, 255, 255]
            cv2.imwrite(os.path.join(debug_dir, "ext1_mask_overlay.png"), img_ext1_vis)
        output = "debug_output"
        # 统计延伸区域1中的总像素数和非黑白像素数量
        ext1_total = int(np.sum(ext1_mask > 0))
        ext1_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext1_mask
        ,debug_output=debug_dir)

        # 延伸方向2：从p2向相同方向延伸
        p2_ext = [p2[0] + ux * extend_length, p2[1] + uy * extend_length]
        cand2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cand2_mask, (int(p1[0]), int(p1[1])), (int(p2_ext[0]), int(p2_ext[1])), 255, int(line_thickness))
        # 只保留延伸部分的掩码（排除原始线段）
        ext2_mask = cv2.bitwise_and(cand2_mask, cv2.bitwise_not(base_mask))
        
        # # 可视化：保存ext2_mask并在原图上显示
        if text == "D48":
            cv2.imwrite(os.path.join(debug_dir, "ext2_mask.png"), ext2_mask)
            img_ext2_vis = img.copy()
            if len(img_ext2_vis.shape) == 2:
                img_ext2_vis = cv2.cvtColor(img_ext2_vis, cv2.COLOR_GRAY2BGR)
            img_ext2_vis[ext2_mask > 0] = [0, 255, 255]
            cv2.imwrite(os.path.join(debug_dir, "ext2_mask_overlay.png"), img_ext2_vis)

        # 统计延伸区域2中的总像素数和非黑白像素数量
        ext2_total = int(np.sum(ext2_mask > 0))
        ext2_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext2_mask
        ,debug_output=debug_dir)
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
