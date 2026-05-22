import time
import os
import logging
from paddleocr import PaddleOCR
from .config import Config

logger = logging.getLogger("ocr_system")

class OCREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCREngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        logger.info("初始化 PaddleOCR 5.0（CPU）")
        start = time.time()
        
        self.ocr = PaddleOCR(
            text_detection_model_name=Config.OCR_DET_MODEL,
            text_recognition_model_name=Config.OCR_REC_MODEL,
            use_doc_orientation_classify=Config.USE_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=Config.USE_DOC_UNWARPING,
            use_textline_orientation=Config.USE_TEXTLINE_ORIENTATION
        )
        
        logger.info(f"OCR 初始化完成，耗时 {time.time() - start:.2f}s")
        self._initialized = True

    def predict(self, img_path, output_dir: str = Config.DEFAULT_OUTPUT_DIR,adjust_type=False,init=False):
        results = []
        res0 = []
        try:
            # PaddleOCR handles both single path and list of paths
            raw = self.ocr.predict(img_path)
        except Exception as e:
            logger.error(f"OCR predict failed: {e}")
            return results

        for index, res in enumerate(raw):
            os.makedirs(output_dir, exist_ok=True)
            try:
                res.save_to_img(output_dir)
                res.save_to_json(output_dir)
                res0.append(res.json["res"])
            except Exception as e:
                 logger.error(f"Error saving results for index {index}: {e}")

        # # If input is a list of paths, the function in original code returned None
        # if isinstance(img_path, list):
        #     return []
        if init:
            return res0
        if not raw or not raw[0]:
            return results

        for item in raw[0]:
            try:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                
                box, info = item
                
                # Validation logic
                if not (isinstance(box, (list, tuple)) and len(box) == 4 and 
                        all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box)):
                    continue

                if not isinstance(info, (list, tuple)) or len(info) < 2:
                    continue

                text = str(info[0])
                conf = float(info[1])
                results.append([box, (text, conf)])

            except Exception as e:
                logger.error(f"OCR 解析失败: {e}, 数据: {item}")
        first_res = raw[0]
        if len(results) == 0 and 'rec_texts' in first_res and first_res['rec_texts'] and adjust_type:
            for res in raw:
                results.append(res)

        return results
