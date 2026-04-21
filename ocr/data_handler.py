import os
import json
import logging
from .utils import calculate_iou
from .config import Config

logger = logging.getLogger("ocr_system")

class DataHandler:
    @staticmethod
    def load_rec_polys_from_json(output_dir):
        rec_polys_with_text = []

        for filename in os.listdir(output_dir):
            if not filename.endswith(".json"):
                continue
                
            json_path = os.path.join(output_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if "rec_polys" in data and "rec_texts" in data:
                    rec_polys = data["rec_polys"]
                    rec_texts = data["rec_texts"]
                    rec_scores = data.get("rec_scores", [])
                    
                    if len(rec_polys) != len(rec_texts):
                        continue
                    if "X" in rec_texts:
                        print(1)
                    input_path = data.get("input_path", "")
                    if input_path and "split_" in input_path:
                        parts = input_path.split("_")
                        if len(parts) >= 4:
                            try:
                                ox = int(parts[-2])
                                oy = int(parts[-1].split(".")[0])
                                for i, (poly, text) in enumerate(zip(rec_polys, rec_texts)):
                                    if rec_scores and i < len(rec_scores):
                                        if rec_scores[i] < Config.TEXT_REC_SCORE_THRESH:
                                            continue

                                    poly_xs = [p[0] for p in poly]
                                    poly_ys = [p[1] for p in poly]
                                    min_x, max_x = min(poly_xs), max(poly_xs)
                                    min_y, max_y = min(poly_ys), max(poly_ys)

                                    window_size = Config.SPLIT_WINDOW_SIZE
                                    border_thresh = Config.SPLIT_BORDER_THRESH
                                    if min_x <= border_thresh or max_x >= window_size - border_thresh or min_y <= border_thresh or max_y >= window_size - border_thresh:
                                        continue

                                    offset_poly = [[p[0] + ox, p[1] + oy] for p in poly]
                                    rec_polys_with_text.append((offset_poly, text, input_path, poly,rec_scores[i]))
                            except ValueError:
                                continue
            except Exception as e:
                logger.error(f"读取 JSON 文件失败: {e}, 文件: {json_path}")

        return rec_polys_with_text

    @staticmethod
    def merge_all_json(output_dir, output_file="merged_output.json", iou_threshold=0.5):
        output_file_basename = os.path.basename(output_file) if output_file else None

        merged_data = {
            "dt_polys": [],
            "textline_orientation_angles": [],
            "rec_texts": [],
            "rec_scores": [],
            "rec_polys": [],
            "rec_boxes": []
        }
        
        # Accessors for cleaner code
        m_dt_polys = merged_data["dt_polys"]
        m_angles = merged_data["textline_orientation_angles"]
        m_texts = merged_data["rec_texts"]
        m_scores = merged_data["rec_scores"]
        m_polys = merged_data["rec_polys"]
        m_boxes = merged_data["rec_boxes"]

        for filename in os.listdir(output_dir):
            if not filename.endswith(".json"):
                continue
            if output_file_basename and filename == output_file_basename:
                continue
                
            json_path = os.path.join(output_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Parsing offset
                input_path = data.get("input_path", "")
                ox, oy = 0, 0
                if input_path and "split_" in input_path:
                    parts = input_path.split("_")
                    if len(parts) >= 4:
                        try:
                            ox = int(parts[-2])
                            oy = int(parts[-1].split(".")[0])
                        except ValueError:
                            continue

                if not all(key in data for key in ["dt_polys", "rec_texts", "rec_scores", "rec_polys", "rec_boxes"]):
                    continue

                dt_polys = data["dt_polys"]
                rec_texts = data["rec_texts"]
                rec_scores = data["rec_scores"]
                rec_polys = data["rec_polys"]
                rec_boxes = data["rec_boxes"]
                
                angles = data.get("textline_orientation_angles", [])

                if not (len(dt_polys) == len(rec_texts) == len(rec_scores) == len(rec_polys) == len(rec_boxes)):
                    continue

                for i in range(len(dt_polys)):
                    offset_dt_poly = [[p[0] + ox, p[1] + oy] for p in dt_polys[i]]
                    text = rec_texts[i]
                    score = rec_scores[i]
                    offset_rec_poly = [[p[0] + ox, p[1] + oy] for p in rec_polys[i]]
                    offset_rec_box = [rec_boxes[i][0] + ox, rec_boxes[i][1] + oy,
                                     rec_boxes[i][2] + ox, rec_boxes[i][3] + oy]
                    
                    angle = angles[i] if i < len(angles) else None

                    overlap_indices = []
                    for j, existing_box in enumerate(m_boxes):
                        iou = calculate_iou(offset_rec_box, existing_box)
                        if iou > iou_threshold:
                            overlap_indices.append(j)

                    if overlap_indices:
                        best_existing_index = max(
                            overlap_indices,
                            key=lambda idx: len(str(m_texts[idx])) if m_texts[idx] is not None else 0,
                        )
                        
                        existing_text = m_texts[best_existing_index]
                        if len(str(text)) > len(str(existing_text)):
                            # Replace existing
                            m_dt_polys[best_existing_index] = offset_dt_poly
                            m_texts[best_existing_index] = text
                            m_scores[best_existing_index] = score
                            m_polys[best_existing_index] = offset_rec_poly
                            m_boxes[best_existing_index] = offset_rec_box
                            if angle is not None and best_existing_index < len(m_angles):
                                m_angles[best_existing_index] = angle
                        
                        # Remove other overlaps
                        remove_indices = [idx for idx in overlap_indices if idx != best_existing_index]
                        for idx in sorted(remove_indices, reverse=True):
                            del m_dt_polys[idx]
                            del m_texts[idx]
                            del m_scores[idx]
                            del m_polys[idx]
                            del m_boxes[idx]
                            if idx < len(m_angles):
                                del m_angles[idx]
                    else:
                        m_dt_polys.append(offset_dt_poly)
                        m_texts.append(text)
                        m_scores.append(score)
                        m_polys.append(offset_rec_poly)
                        m_boxes.append(offset_rec_box)
                        m_angles.append(angle)

            except Exception as e:
                logger.error(f"读取 JSON 文件失败: {e}, 文件: {json_path}")

        # Final structure
        final_data = {
            "input_path": "merged",
            "page_index": None,
            "model_settings": {},
            "doc_preprocessor_res": {},
            "dt_polys": m_dt_polys,
            "text_det_params": {},
            "text_type": "general",
            "textline_orientation_angles": m_angles,
            "text_rec_score_thresh": Config.TEXT_REC_SCORE_THRESH,
            "rec_texts": m_texts,
            "rec_scores": m_scores,
            "rec_polys": m_polys,
            "rec_boxes": m_boxes
        }

        # Recover meta from first file
        for filename in os.listdir(output_dir):
            if filename.endswith(".json") and (not output_file_basename or filename != output_file_basename):
                try:
                    with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "model_settings" in data: final_data["model_settings"] = data["model_settings"]
                        if "doc_preprocessor_res" in data: final_data["doc_preprocessor_res"] = data["doc_preprocessor_res"]
                        if "text_det_params" in data: final_data["text_det_params"] = data["text_det_params"]
                    break
                except:
                    pass

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        logger.info(f"整合后的JSON文件已保存: {output_file}")
        logger.info(f"合并后的dt_polys数量: {len(m_dt_polys)}")
        return final_data
    
    @staticmethod
    def cleanup_output_dir(output_dir, delete_empty=True):
        if not os.path.exists(output_dir):
            return

        # Delete 'processed' images
        for filename in os.listdir(output_dir):
            if "processed" in filename.lower():
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"已删除文件: {file_path}")

        if delete_empty:
            for filename in os.listdir(output_dir):
                if filename.endswith(".json"):
                    json_path = os.path.join(output_dir, filename)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data.get("rec_texts"), list) and len(data["rec_texts"]) == 0:
                            base_name = filename.replace("_res.json", "")
                            img_path = os.path.join(output_dir, f"{base_name}_ocr_res_img.jpg")
                            if os.path.isfile(img_path):
                                os.remove(img_path)
                            os.remove(json_path)
                            logger.info(f"已删除空rec_texts的记录: {json_path}")
                    except Exception as e:
                        logger.error(f"清理文件失败: {e}")
