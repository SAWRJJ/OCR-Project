import os
import time
import logging
from ocr.utils import setup_logging
from ocr.config import Config
from ocr.ocr_engine import OCREngine
from ocr.image_processor import ImageProcessor
from ocr.data_handler import DataHandler
from ocr.visualizer import Visualizer
from ocr.micro_ocr import process_micro_images, save_results_to_excel
from compare_xlsx_matched_keys import *
# Setup logging
logger = setup_logging()

def get_output_dir_for_image(image_path: str, output_root: str = "output") -> str:
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_root, base_name)

def main(image_paths):
    # Initialize OCR Engine
    ocr_engine = OCREngine()
    for image_path in image_paths:
        output_dir = get_output_dir_for_image(image_path, output_root="output")
        os.makedirs(output_dir, exist_ok=True)
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]

        logger.info("====== OCR 流程开始 ======")
        t0 = time.time()

        # Split Image
        splits, img_paths = ImageProcessor.split_image(image_path, window=1000)

        t2 = time.time()
        if not splits:
            raise SystemExit("切图失败")
        logger.info(f"{image_base_name} split time:{t2 - t0}")

        # Run OCR
        ocr_start_time = time.time()
        # Processing all splits in batch
        ocr_engine.predict(img_paths, output_dir=output_dir)

        ocr_end_time = time.time()
        ocr_total_time = ocr_end_time - ocr_start_time
        logger.info(f"OCR 处理总耗时: {ocr_total_time:.2f}s")
        logger.info(f"{image_base_name} OCR 处理总耗时: {ocr_total_time:.2f}s")

        t1 = time.time()
        # Merge Results
        if os.path.exists(output_dir):
            DataHandler.merge_all_json(
                output_dir,
                output_file=os.path.join(output_dir, f"{image_base_name}_merged.json")
            )
        t2 = time.time()
        logger.info(f"{image_base_name} merge json time:{t2 - t1}")

        t1 = time.time()
        # Draw Red Lines
        if os.path.exists(output_dir):
            rec_polys = DataHandler.load_rec_polys_from_json(output_dir)
            t2 = time.time()
            logger.info(f"{image_base_name} rec_polys 加载完成，耗时：:{t2 - t1}")
            if rec_polys:
                Visualizer.draw_red_lines(
                    image_path,
                    rec_polys,
                    out_path=os.path.join(output_dir, f"{image_base_name}_ocr_result_with_red_lines.jpg"),
                    save_boxes_dir=os.path.join(output_dir, "micro_img"),
                )
        t2 = time.time()
        logger.info(f"{image_base_name} draw time:{t2 - t1}")

        t1 = time.time()
        # Micro Image OCR
        micro_img_dir = os.path.join(output_dir, "micro_img")
        if os.path.exists(micro_img_dir):
            all_results = process_micro_images(micro_img_dir)
            # 保存到 Excel
            excel_path = os.path.join(output_dir, f"{image_base_name}_ocr_report.xlsx")
            save_results_to_excel(all_results, excel_path, micro_img_dir)
        t2 = time.time()
        logger.info(f"{image_base_name} micro ocr time:{t2 - t1}")

        # Cleanup
        DataHandler.cleanup_output_dir(output_dir, delete_empty=True)
        tn = time.time()
        logger.info(f"OCR {image_base_name} 完成，文字结果已保存 time:{tn-t0}")

if __name__ == "__main__":
    # img_path1 =[]
    img_path1 = ["img/t7.jpg"]
    img_path = ["img/t3.jpg", "img/t6.jpg", "img/t4.jpg", "img/t1.jpg"]
    img_path = ["img/t9.jpg", "img/t8.jpg", "img/t5.jpg", "img/t2.jpg"]
    # "img/t7.jpg"
    img_path= ["img/t9.jpg","img/t8.jpg","img/t5.jpg","img/t2.jpg","img/t3.jpg","img/t6.jpg","img/t4.jpg","img/t1.jpg"]+img_path1
    img_path = ["img/t7.jpg"]# "img/t9.jpg","img/t8.jpg"," "img/t9.jpg","img/t8.jpg","img/t5.jpg"
    # img_path = ["img/元龙站.jpg","img/凤阁岭站.jpg","img/武功站.jpg","img/杨陵站场.jpg","img/新建河站.jpg"]#
    t1 = time.time()
    main(img_path)
    t2 = time.time()
    logger.info(f"Total time:{t2 - t1}")
    mainc()
