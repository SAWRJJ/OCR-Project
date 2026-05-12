import logging
import os

import cv2
import numpy as np
import unicodedata
import re

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

def calculate_angle_to_horizontal(point_a, point_b):
    """
    计算从点a到点b相对于水平线的偏移角度

    Args:
        point_a: 起点坐标 (x, y)
        point_b: 终点坐标 (x, y)

    Returns:
        float: 角度（弧度），正值表示向上偏，负值表示向下偏
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return np.arctan2(dx, dy)

def fullwidth_to_halfwidth(text):
    return unicodedata.normalize('NFKC', text)

def has_text_or_number(s):
    # \u4e00-\u9fa5 是汉字的范围
    # a-zA-Z0-9 是字母和数字
    pattern = re.compile(r'[a-zA-Z0-9\u4e00-\u9fa5]')
    return bool(pattern.search(s))