import os

# Paddle / 系统环境
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CPU_NUM_THREADS"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"

class Config:
    OCR_DET_MODEL = "PP-OCRv5_mobile_det"
    OCR_REC_MODEL = "PP-OCRv5_mobile_rec"
    USE_DOC_ORIENTATION_CLASSIFY = False
    USE_DOC_UNWARPING = False
    USE_TEXTLINE_ORIENTATION = False
    
    DEFAULT_OUTPUT_DIR = "output"
    TEXT_REC_SCORE_THRESH = 0.0
    
    TARGET_CHARS = ["S", "X", "D"]
    EXCLUDE_SUBSTRINGS = ["DG"]

    SPLIT_WINDOW_SIZE = 1000
    SPLIT_BORDER_THRESH = 5
