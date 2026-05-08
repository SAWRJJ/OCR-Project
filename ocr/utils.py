import logging
import os

import cv2
import numpy as np
def calculate_shift_params(micro_poly, input_angle=None,extend_length = 55):
    """计算平移参数

    Args:
        micro_poly: micro_poly 坐标列表
        input_angle: 输入的倾斜角度（弧度），如果为None则从micro_poly计算

    Returns:
        dict: 包含计算结果的字典
    """
    micro_poly_arr = np.array(micro_poly)
    sorted_by_x = micro_poly_arr[micro_poly_arr[:, 0].argsort()]
    left_points = sorted_by_x[:2]
    right_points = sorted_by_x[-2:]

    left_center = np.mean(left_points, axis=0)
    right_center = np.mean(right_points, axis=0)
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]

    if input_angle is not None:
        angle = input_angle
        dx = 50 * np.cos(angle)
        dy = 50 * np.sin(angle)
    else:
        angle = np.arctan2(dy, dx)

    length = np.sqrt(dx ** 2 + dy ** 2)
    if length > 0:
        ux = dx / length
        uy = dy / length
    else:
        ux, uy = 1, 0

    p1_shifted = (int(left_points[0][0] + ux * extend_length), int(left_points[0][1] + uy * extend_length))
    p2_shifted = (int(left_points[1][0] + ux * extend_length), int(left_points[1][1] + uy * extend_length))

    left_poly_format = [
        [int(left_points[0][0]), int(left_points[0][1])],
        [int(left_points[1][0]), int(left_points[1][1])],
        [int(p2_shifted[0]), int(p2_shifted[1])],
        [int(p1_shifted[0]), int(p1_shifted[1])]
    ]

    return {
        'left_points': left_points,
        'right_points': right_points,
        'left_center': left_center,
        'right_center': right_center,
        'angle': angle,
        'dx': dx,
        'dy': dy,
        'ux': ux,
        'uy': uy,
        'extend_length': extend_length,
        'p1_shifted': p1_shifted,
        'p2_shifted': p2_shifted,
    },left_poly_format

import logging
import sys

class ColorFormatter(logging.Formatter):

    COLORS = {
        logging.DEBUG: "\033[36m",     # 青色
        logging.INFO: "\033[32m",      # 绿色
        logging.WARNING: "\033[33m",   # 黄色
        logging.ERROR: "\033[31m",     # 红色
        logging.CRITICAL: "\033[41m",  # 红底
    }

    RESET = "\033[0m"

    def format(self, record):

        color = self.COLORS.get(record.levelno, self.RESET)

        log_msg = super().format(record)

        return f"{color}{log_msg}{self.RESET}"


def setup_logging():

    logger = logging.getLogger("ocr_system")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.StreamHandler(sys.stdout)

        handler.setLevel(logging.INFO)

        formatter = ColorFormatter(
            "[%(asctime)s] - [%(levelname)s] - %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger

logger = setup_logging()

def normalize_text_for_match(text) -> str:
    if text is None:
        return ""
    return str(text).upper()

def should_keep_text(text, target_chars, exclude_substrings) -> bool:
    text_upper = normalize_text_for_match(text)
    if not text_upper:
        return False
    if exclude_substrings:
        for s in exclude_substrings:
            if s and str(s).upper() in text_upper:
                return False
    if not target_chars:
        return True
    return any(str(c).upper() in text_upper for c in target_chars)

def safe_filename_component(text: str, max_len: int = 32) -> str:
    if not text:
        return ""
    cleaned = "".join(c if c.isalnum() else "_" for c in str(text))
    cleaned = cleaned.strip("_")
    return cleaned[:max_len]

def calculate_iou(box1, box2):
    """
    计算两个矩形的IoU（Intersection over Union）
    box格式：[x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_centers_separate(mask,
                               min_area=10,
                               min_circularity=0.6,
                               min_radius=7):

    if cv2.countNonZero(mask) == 0:
        return [], np.zeros_like(mask)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    ))

    centers = []

    non_circular_mask = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        perimeter = cv2.arcLength(contours[0], True)

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        radius = np.sqrt(area / np.pi)

        if circularity <= min_circularity or radius <= min_radius:

            cx = int(centroids[i][0])
            cy = int(centroids[i][1])

            centers.append((cx, cy))

            non_circular_mask = cv2.bitwise_or(
                non_circular_mask,
                component_mask
            )

    return centers, non_circular_mask

def change_img_red(img_path):
    img = cv2.imread(img_path)

    original_dir = os.path.join(os.path.dirname(img_path), 'original')
    os.makedirs(original_dir, exist_ok=True)
    original_path = os.path.join(original_dir, os.path.basename(img_path))
    cv2.imwrite(original_path, img)

    # =========================
    # HSV转换
    # =========================

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # =========================
    # 红色mask
    # =========================

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])

    lower_red2 = np.array([150, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # =========================
    # 分析非圆形红色区域
    # =========================

    red_centers, non_circular_mask = calculate_centers_separate(mask_red)

    # =========================
    # 外扩mask
    # =========================

    if cv2.countNonZero(non_circular_mask) > 0:
        kernel = np.ones((3, 3), np.uint8)

        expanded_mask = cv2.dilate(
            non_circular_mask,
            kernel,
            iterations=1
        )

        # 白色覆盖
        img[expanded_mask > 0] = 255

    # =========================
    # 保存结果
    # =========================

    output_path = img_path

    cv2.imwrite(output_path, img)
    return output_path
