import cv2
import json
import os
import math
import numpy as np
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ==================== 完美复用你原脚本的图像基础算子 ====================

def has_x_or_s(text):
    if not text: return False
    return 'X' in text.upper() or 'S' in text.upper()


def extract_color_mask(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    valid_color = (s >= 50) & (v >= 50)
    if color == 'blue':
        mask = (h >= 90) & (h <= 130) & valid_color
    elif color == 'green':
        mask = (h >= 35) & (h <= 85) & valid_color
    elif color == 'yellow':
        mask = (h >= 20) & (h <= 35) & valid_color
    elif color == 'red':
        mask = ((h >= 0) & (h <= 10) | (h >= 170) & (h <= 179)) & valid_color
    else:
        mask = np.zeros_like(valid_color)
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(cv2.erode(mask_uint8, kernel, iterations=1), kernel, iterations=1)


def get_color_cluster_centers(mask_uint8):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    return [(float(centroids[i][0]), float(centroids[i][1])) for i in range(1, num_labels) if
            stats[i, cv2.CC_STAT_AREA] >= 4]


def find_cluster_centers(points, distance_threshold=25):
    if not points: return []
    points = np.array(points)
    n = len(points)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]: continue
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            current = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    if np.hypot(points[current][0] - points[j][0],
                                points[current][1] - points[j][1]) < distance_threshold:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)
    center_points = []
    for cluster in clusters:
        c_pts = points[cluster]
        centroid = np.mean(c_pts, axis=0)
        min_idx = np.argmin([np.hypot(p[0] - centroid[0], p[1] - centroid[1]) for p in c_pts])
        center_points.append(tuple(c_pts[min_idx]))
    return center_points


# ==================== 额外要求的两张图可视化渲染函数 ====================

def draw_connections_to_centers(img_final_base, final_assignments, text_boxes, output_path):
    vis_img = img_final_base.copy()
    vis_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}

    for box_key, box in text_boxes.items():
        poly_pts = np.array(box["precise_global_poly"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [poly_pts], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        text_str = box["text"]
        x_min = int(min([pt[0] for pt in box["precise_global_poly"]]))
        y_min = int(min([pt[1] for pt in box["precise_global_poly"]]))
        cv2.putText(vis_img, text_str, (x_min, max(y_min - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    for box_key, box in text_boxes.items():
        poly_center = box["poly_center"]
        cv2.circle(vis_img, (int(poly_center[0]), int(poly_center[1])), 5, (255, 128, 0), 1, cv2.LINE_AA)
        if 'D' in box["text"].upper():
            cv2.drawMarker(vis_img, (int(poly_center[0]), int(poly_center[1])), (0, 165, 255),
                           markerType=cv2.MARKER_STAR, markerSize=8, thickness=1)

    for assign in final_assignments:
        c_name = assign["color_name"]
        cc_pt = assign["cc"]
        box_key = assign["box_key"]

        box = text_boxes[box_key]
        poly_center = box["poly_center"]
        color_rgb = vis_colors.get(c_name, (255, 255, 255))

        cv2.circle(vis_img, (int(poly_center[0]), int(poly_center[1])), 4, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.line(vis_img, (int(cc_pt[0]), int(cc_pt[1])), (int(poly_center[0]), int(poly_center[1])), color_rgb, 1,
                 cv2.LINE_AA)
        cv2.circle(vis_img, (int(cc_pt[0]), int(cc_pt[1])), 5, color_rgb, -1, cv2.LINE_AA)
        cv2.circle(vis_img, (int(cc_pt[0]), int(cc_pt[1])), 2, (255, 255, 255), -1, cv2.LINE_AA)

        if c_name == 'blue':
            coord_text = f"[{int(cc_pt[0])},{int(cc_pt[1])}]"
            cv2.putText(vis_img, coord_text, (int(cc_pt[0]) + 8, int(cc_pt[1]) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 100), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, vis_img)
    print(f"[SUCCESS] 升级版全包络归属连线图已保存至: {output_path}")


def draw_detected_vs_json_centers(img_final_base, text_boxes, color_clusters, output_path):
    vis_img = img_final_base.copy()
    vis_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}

    for color_name, centers in color_clusters.items():
        color_rgb = vis_colors.get(color_name, (255, 255, 255))
        for cc in centers:
            cv2.circle(vis_img, (int(cc[0]), int(cc[1])), 6, color_rgb, -1, cv2.LINE_AA)
            cv2.circle(vis_img, (int(cc[0]), int(cc[1])), 2, (255, 255, 255), -1, cv2.LINE_AA)

    for box_key, box in text_boxes.items():
        for color_name, json_pts in box["global_json_centers"].items():
            color_rgb = vis_colors.get(color_name, (255, 255, 255))
            for j_pt in json_pts:
                cv2.circle(vis_img, (int(j_pt[0]), int(j_pt[1])), 10, color_rgb, 2, cv2.LINE_AA)
                cv2.drawMarker(vis_img, (int(j_pt[0]), int(j_pt[1])), color_rgb, markerType=cv2.MARKER_CROSS,
                               markerSize=6, thickness=1)

    cv2.imwrite(output_path, vis_img)
    print(f"[SUCCESS] 额外图2（检测 vs JSON对比图）已保存至: {output_path}")


# ==================== 核心：同步计算与无引线可视化函数 ====================

def generate_accurate_micro_relations_vis(img_path, json_dir, output_vis_dir="visualization_output"):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] 无法加载图像: {img_path}")
        return {}

    image_base_name = os.path.splitext(os.path.basename(img_path))[0]
    colors = ['red', 'blue', 'green', 'yellow']
    vis_colors = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}

    # 1. 建立全颜色联合二值化底图
    masks = {}
    combined_binary = np.zeros(img.shape[:2], dtype=np.uint8)
    for color in colors:
        masks[color] = extract_color_mask(img, color)
        combined_binary = cv2.bitwise_or(combined_binary, masks[color])
    vis_img_base = cv2.cvtColor(combined_binary, cv2.COLOR_GRAY2BGR)
    vis_img_final = vis_img_base.copy()

    # 2. 图像实际提取点与黄色二次聚类
    color_clusters = {c: get_color_cluster_centers(masks[c]) for c in colors}
    color_clusters['yellow'] = find_cluster_centers(color_clusters['yellow'], distance_threshold=25)

    # 3. 解析并构建全局文本框基础映射矩阵
    json_files = list(Path(json_dir).glob("*.json"))
    text_boxes = {}

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        patch_poly = data.get('patch_poly', [])
        if not patch_poly: continue
        patch_left, patch_top = patch_poly[0][0], patch_poly[0][1]

        for detail_idx, detail in enumerate(data.get('re_ocr_details', [])):
            text = detail.get('text', '').strip()
            micro_poly = detail.get('micro_poly', [])
            if not micro_poly: continue

            precise_global_poly = [[float(pt[0] + patch_left), float(pt[1] + patch_top)] for pt in micro_poly[:4]]
            g_x = [pt[0] for pt in precise_global_poly[:4]]
            g_y = [pt[1] for pt in precise_global_poly[:4]]
            poly_center = (sum(g_x) / 4.0, sum(g_y) / 4.0)

            global_json_centers = {"red": [], "yellow": [], "green": [], "blue": []}
            centers_sep = detail.get('color_centers_separate', {})
            for c_name in global_json_centers.keys():
                for lp in centers_sep.get(c_name, []):
                    global_json_centers[c_name].append((float(lp[0] + patch_left), float(lp[1] + patch_top)))

            global_white_centers = [(float(w[0] + patch_left), float(w[1] + patch_top)) for w in
                                    centers_sep.get('white', [])]

            box_key = f"{json_file.name}_detail_{detail_idx}"
            text_boxes[box_key] = {
                "text": text, "is_xs": has_x_or_s(text), "patch_origin": (patch_left, patch_top),
                "precise_global_poly": precise_global_poly, "poly_center": poly_center,
                "global_white_centers": global_white_centers, "global_json_centers": global_json_centers,
                "target_colors": ['red', 'yellow', 'green'] if has_x_or_s(text) else ['red', 'blue'],
                "assigned_results": {}
            }

    # ✨✨ 新增核心逻辑：白灯（white）初筛与动态再分配处理 ✨✨
    orphan_white_centers = []
    # 步骤 A：对仅由 D+数字 组成的文本框执行独占初筛，多余的剥离进自由池
    for box_key, box in text_boxes.items():
        text_str = box["text"].upper()
        # 严格匹配仅含D和数字的文本框
        if re.match(r"^D\d+$", text_str) and box["global_white_centers"]:
            # 计算所有白灯点到该文本框几何中心的距离
            p_center = box["poly_center"]
            distances = [math.hypot(w_pt[0] - p_center[0], w_pt[1] - p_center[1]) for w_pt in
                         box["global_white_centers"]]
            min_idx = np.argmin(distances)

            # 提取保留唯一的最近白灯，其余扔入孤立无主池
            closest_white = box["global_white_centers"][min_idx]
            for idx, w_pt in enumerate(box["global_white_centers"]):
                if idx != min_idx:
                    orphan_white_centers.append(w_pt)

            # 更新该框内白灯列表，只保留最近的一个
            box["global_white_centers"] = [closest_white]

    # 步骤 B：将剥离出来的白灯，按就近原则重新塞回任意含有 'D' 字符的文本框中
    for w_pt in orphan_white_centers:
        best_d_box_key = None
        min_d_box_dist = float('inf')

        for box_key, box in text_boxes.items():
            if 'D' in box["text"].upper():
                p_center = box["poly_center"]
                dist = math.hypot(w_pt[0] - p_center[0], w_pt[1] - p_center[1])
                if dist < min_d_box_dist:
                    min_d_box_dist = dist
                    best_d_box_key = box_key

        # 找到最近的D文本框后塞给它
        if best_d_box_key:
            text_boxes[best_d_box_key]["global_white_centers"].append(w_pt)

    # 100% 同步核心：执行你原脚本中独特的双限阈值过滤逻辑
    for box_key, box in text_boxes.items():
        for color_name, json_pts in list(box["global_json_centers"].items()):
            valid_json_pts = []
            collected_pts = color_clusters.get(color_name, [])
            for j_pt in json_pts:
                min_dist_found = float('inf')
                for c_pt in collected_pts:
                    dist = math.hypot(j_pt[0] - c_pt[0], j_pt[1] - c_pt[1])
                    if dist < min_dist_found: min_dist_found = dist

                if 5.0 < min_dist_found <= 20.0:
                    continue
                valid_json_pts.append(j_pt)
            box["global_json_centers"][color_name] = valid_json_pts

    # 4. 三阶段精确混合匹配逻辑
    final_assignments = []
    matched_points_registry = set()

    for color_name, centers in color_clusters.items():
        for cc in centers:
            cc_round = (round(cc[0], 2), round(cc[1], 2))
            matched_in_stage1 = False
            stage1_box_key, stage1_dist = None, float('inf')

            # ---- 【阶段一：专属靶向筛查】 ----
            for box_key, box in text_boxes.items():
                if not box["is_xs"]: continue
                if color_name not in box["target_colors"]: continue

                for j_pt in box["global_json_centers"].get(color_name, []):
                    dist = math.hypot(cc[0] - j_pt[0], cc[1] - j_pt[1])
                    if dist <= 3.0:
                        stage1_dist = dist
                        stage1_box_key = box_key
                        matched_in_stage1 = True
                        break
                if matched_in_stage1: break

            if matched_in_stage1:
                final_assignments.append(
                    {"color_name": color_name, "cc": cc, "box_key": stage1_box_key, "type": "Targeted"})
                matched_points_registry.add((color_name, cc_round[0], cc_round[1]))
                text_boxes[stage1_box_key]["assigned_results"][color_name] = {
                    "cluster_center": [cc_round[0], cc_round[1]], "type": "Targeted"}
                continue

            # ---- 【阶段二：漏网之鱼就近收编】 ----
            best_stage2_match = None
            global_min_dist = float('inf')
            for box_key, box in text_boxes.items():
                if color_name not in box["target_colors"]: continue

                reference_points = [box["poly_center"]] + box["precise_global_poly"] + box["global_white_centers"]
                for pt_coords in reference_points:
                    dist = math.hypot(cc[0] - pt_coords[0], cc[1] - pt_coords[1])
                    if dist > 450: continue
                    if dist < global_min_dist:
                        global_min_dist = dist
                        best_stage2_match = {"assigned_box_key": box_key, "type": "Nearest"}

            if best_stage2_match:
                bk = best_stage2_match["assigned_box_key"]
                final_assignments.append({"color_name": color_name, "cc": cc, "box_key": bk, "type": "Nearest"})
                matched_points_registry.add((color_name, cc_round[0], cc_round[1]))
                text_boxes[bk]["assigned_results"][color_name] = {"cluster_center": [cc_round[0], cc_round[1]],
                                                                  "type": "Nearest"}
                continue

    # ---- 【阶段三：无条件全局就近强制收编】 ----
    for color_name, centers in color_clusters.items():
        for cc in centers:
            cc_round = (round(cc[0], 2), round(cc[1], 2))
            if (color_name, cc_round[0], cc_round[1]) not in matched_points_registry:
                best_forced_match = None
                forced_min_dist = float('inf')

                for box_key, box in text_boxes.items():
                    reference_points = [box["poly_center"]] + box["precise_global_poly"] + box["global_white_centers"]
                    for pt_coords in reference_points:
                        dist = math.hypot(cc[0] - pt_coords[0], cc[1] - pt_coords[1])
                        if dist < forced_min_dist:
                            forced_min_dist = dist
                            best_forced_match = {"assigned_box_key": box_key, "type": "Forced_Nearest"}

                if best_forced_match:
                    bk = best_forced_match["assigned_box_key"]
                    final_assignments.append(
                        {"color_name": color_name, "cc": cc, "box_key": bk, "type": "Forced_Nearest"})
                    matched_points_registry.add((color_name, cc_round[0], cc_round[1]))

                    save_key = color_name if color_name not in text_boxes[bk][
                        "assigned_results"] else f"{color_name}_forced"
                    text_boxes[bk]["assigned_results"][save_key] = {"cluster_center": [cc_round[0], cc_round[1]],
                                                                    "type": "Forced_Nearest"}

    # ==================== 5. 渲染最终结果主图（带文本内容） ====================
    for box_key, box in text_boxes.items():
        poly_pts = np.array(box["precise_global_poly"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img_final, [poly_pts], isClosed=True, color=(255, 100, 0), thickness=2)

        text_str = box["text"]
        if text_str:
            px = [pt[0] for pt in box["precise_global_poly"]]
            py = [pt[1] for pt in box["precise_global_poly"]]
            text_pos = (int(min(px)), int(min(py)) - 8)

            try:
                font_paths = ["msyh.ttc", "simsun.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                              "/usr/share/fonts/fonts-go/Go.ttf"]
                font_path = next((p for p in font_paths if os.path.exists(p)), None)

                if font_path:
                    pil_img = Image.fromarray(cv2.cvtColor(vis_img_final, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype(font_path, 14)
                    draw.text((text_pos[0] - 1, text_pos[1] - 1), text_str, fill=(0, 0, 0), font=font)
                    draw.text((text_pos[0], text_pos[1]), text_str, fill=(0, 255, 255), font=font)
                    vis_img_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    raise FileNotFoundError
            except Exception:
                cv2.putText(vis_img_final, text_str, (text_pos[0], text_pos[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    for assign in final_assignments:
        c_name = assign["color_name"]
        cc_pt = assign["cc"]
        core_color = vis_colors.get(c_name, (255, 255, 255))
        cv2.circle(vis_img_final, (int(cc_pt[0]), int(cc_pt[1])), 7, core_color, -1, cv2.LINE_AA)
        cv2.circle(vis_img_final, (int(cc_pt[0]), int(cc_pt[1])), 2, (255, 255, 255), -1, cv2.LINE_AA)

    os.makedirs(output_vis_dir, exist_ok=True)
    final_out_path = os.path.join(output_vis_dir, f"{image_base_name}_final_matching_result.jpg")
    cv2.imwrite(final_out_path, vis_img_final)
    print(f"[SUCCESS] 零偏差高保真二值化结果图已保存至: {final_out_path}")

    path_conn = os.path.join(output_vis_dir, f"{image_base_name}_diag_connections.jpg")
    path_compare = os.path.join(output_vis_dir, f"{image_base_name}_diag_detected_vs_json.jpg")

    draw_connections_to_centers(vis_img_base, final_assignments, text_boxes, path_conn)
    draw_detected_vs_json_centers(vis_img_base, text_boxes, color_clusters, path_compare)

    # ==================== 返回关系字典 ====================
    output_report_dict = {}
    report_colors = colors + ['white']
    for box_key, box in text_boxes.items():
        text_content = box["text"]
        if text_content not in output_report_dict:
            output_report_dict[text_content] = {
                "text": text_content,
                "color_centers": {c: [] for c in report_colors}
            }

    for assign in final_assignments:
        box_key = assign["box_key"]
        color_name = assign["color_name"]
        global_cc = assign["cc"]

        box = text_boxes[box_key]
        text_content = box["text"]
        patch_left, patch_top = box["patch_origin"]

        micro_x = round(global_cc[0] - patch_left, 2)
        micro_y = round(global_cc[1] - patch_top, 2)

        point_payload = {
            "micro_coord": [micro_x, micro_y],
            "global_coord": [round(global_cc[0], 2), round(global_cc[1], 2)],
            "match_stage": assign["type"]
        }
        output_report_dict[text_content]["color_centers"][color_name].append(point_payload)

    for box_key, box in text_boxes.items():
        text_content = box["text"]
        patch_left, patch_top = box["patch_origin"]

        for w_pt in box["global_white_centers"]:
            micro_wx = round(w_pt[0] - patch_left, 2)
            micro_wy = round(w_pt[1] - patch_top, 2)

            white_payload = {
                "micro_coord": [micro_wx, micro_wy],
                "global_coord": [round(w_pt[0], 2), round(w_pt[1], 2)],
                "match_stage": "JSON_Preset"
            }
            output_report_dict[text_content]["color_centers"]["white"].append(white_payload)

    return output_report_dict


if __name__ == "__main__":
    final_mapped_data = generate_accurate_micro_relations_vis("t15.jpg", "results")
    print("\n[INFO] 任务完成，数据已成功 return。")