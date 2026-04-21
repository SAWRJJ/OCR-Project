import logging
import os
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("ocr_system")
    logging.getLogger("paddle").setLevel(logging.ERROR)
    logging.getLogger("ppocr").setLevel(logging.ERROR)
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
