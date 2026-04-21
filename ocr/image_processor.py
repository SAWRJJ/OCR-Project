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
        return int(np.sum(~(white_mask | black_mask)))

    @staticmethod
    def choose_one_sided_extension(img, p1, p2, extend_length=220, line_thickness=3, white_thresh=245, black_thresh=10, min_non_bw_pixels=5):
        if img is None or img.size == 0:
            return None

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        length = float(np.sqrt(dx * dx + dy * dy))
        if length <= 1e-6:
            return None

        ux = dx / length
        uy = dy / length

        h, w = img.shape[:2]
        base_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(base_mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, int(line_thickness))

        # Check extension 1 (from p1 away)
        p1_ext = [p1[0] - ux * extend_length, p1[1] - uy * extend_length]
        cand1_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cand1_mask, (int(p1_ext[0]), int(p1_ext[1])), (int(p2[0]), int(p2[1])), 255, int(line_thickness))
        ext1_mask = cv2.bitwise_and(cand1_mask, cv2.bitwise_not(base_mask))
        
        ext1_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext1_mask
        )

        # Check extension 2 (from p2 away)
        p2_ext = [p2[0] + ux * extend_length, p2[1] + uy * extend_length]
        cand2_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(cand2_mask, (int(p1[0]), int(p1[1])), (int(p2_ext[0]), int(p2_ext[1])), 255, int(line_thickness))
        ext2_mask = cv2.bitwise_and(cand2_mask, cv2.bitwise_not(base_mask))
        
        ext2_count = ImageProcessor.count_non_bw_pixels_along_line(
            img, (0, 0), (0, 0), line_thickness, white_thresh, black_thresh, mask=ext2_mask
        )

        if ext1_count < int(min_non_bw_pixels) and ext2_count < int(min_non_bw_pixels):
            return None

        if ext2_count >= ext1_count:
            return p1, p2_ext
        return p1_ext, p2

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
