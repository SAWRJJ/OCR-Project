import cv2
import json
import os
import math
import numpy as np
from pathlib import Path


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
    points = np.array(points);
    n = len(points);
    visited = [False] * n;
    clusters = []
    for i in range(n):
        if visited[i]: continue
        cluster = [i];
        visited[i] = True;
        queue = [i]
        while queue:
            current = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    if np.hypot(points[current][0] - points[j][0],
                                points[current][1] - points[j][1]) < distance_threshold:
                        visited[j] = True;
                        cluster.append(j);
                        queue.append(j)
        clusters.append(cluster)
    center_points = []
    for cluster in clusters:
        c_pts = points[cluster];
        centroid = np.mean(c_pts, axis=0)
        min_idx = np.argmin([np.hypot(p[0] - centroid[0], p[1] - centroid[1]) for p in c_pts])
        center_points.append(tuple(c_pts[min_idx]))
    return center_points


# ==================== 核心：同步计算与无引线可视化函数 ====================

def generate_accurate_micro_relations_vis(img_path, json_dir, output_vis_dir="visualization_output"):
    """
    完全对齐原始双限阈值与三阶段混合匹配算法，
    生成不带引线的全颜色二值化最终结果图，并返回带有解算好的 micro 局部坐标的关系字典。
    """
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
    vis_img_final = cv2.cvtColor(combined_binary, cv2.COLOR_GRAY2BGR)

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
            text = detail.get('text', '')
            micro_poly = detail.get('micro_poly', [])
            if not micro_poly: continue

            # 平移映射回大图全局坐标
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

    # ✨✨ 100% 同步核心：执行你原脚本中独特的双限阈值过滤逻辑 ✨✨
    for box_key, box in text_boxes.items():
        for color_name, json_pts in list(box["global_json_centers"].items()):
            valid_json_pts = []
            collected_pts = color_clusters.get(color_name, [])
            for j_pt in json_pts:
                min_dist_found = float('inf')
                for c_pt in collected_pts:
                    dist = math.hypot(j_pt[0] - c_pt[0], j_pt[1] - c_pt[1])
                    if dist < min_dist_found: min_dist_found = dist

                # 如果落在 5 到 20 像素之间，触发 should_filter 丢弃该点
                if 5.0 < min_dist_found <= 20.0:
                    continue
                valid_json_pts.append(j_pt)
            box["global_json_centers"][color_name] = valid_json_pts

    # 4. 三阶段精确混合匹配逻辑（一比一高保真还原）
    final_assignments = []
    matched_points_registry = set()

    for color_name, centers in color_clusters.items():
        for cc in centers:
            cc_round = (round(cc[0], 2), round(cc[1], 2))
            matched_in_stage1 = False
            stage1_box_key, stage1_dist = None, float('inf')

            # ---- 【阶段一：专属靶向筛查】（严格核对 is_xs 边界） ----
            for box_key, box in text_boxes.items():
                if not box["is_xs"]: continue  # 核心限制：不含 X/S 的方框在此处直接跳过
                if color_name not in box["target_colors"]: continue

                for j_pt in box["global_json_centers"].get(color_name, []):
                    dist = math.hypot(cc[0] - j_pt[0], cc[1] - j_pt[1])
                    if dist <= 15.0:  # SEARCH_RADIUS = 15.0
                        stage1_dist = dist;
                        stage1_box_key = box_key;
                        matched_in_stage1 = True;
                        break
                if matched_in_stage1: break

            if matched_in_stage1:
                final_assignments.append(
                    {"color_name": color_name, "cc": cc, "box_key": stage1_box_key, "type": "Targeted"})
                matched_points_registry.add((color_name, cc_round[0], cc_round[1]))
                text_boxes[stage1_box_key]["assigned_results"][color_name] = {
                    "cluster_center": [cc_round[0], cc_round[1]], "type": "Targeted"}
                continue

            # ---- 【阶段二：漏网之鱼就近收编】（多特征点中心+顶点+白心矩阵比对） ----
            best_stage2_match = None;
            global_min_dist = float('inf')
            for box_key, box in text_boxes.items():
                if color_name not in box["target_colors"]: continue

                # 完美还原你脚本里的参考点序列组合
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
                best_forced_match = None;
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

                    # 复制原脚本主键保护逻辑，防止覆盖高精度点
                    save_key = color_name if color_name not in text_boxes[bk][
                        "assigned_results"] else f"{color_name}_forced"
                    text_boxes[bk]["assigned_results"][save_key] = {"cluster_center": [cc_round[0], cc_round[1]],
                                                                    "type": "Forced_Nearest"}

    # ==================== 渲染最终结果图（无引线、无文字标签） ====================
    for box_key, box in text_boxes.items():
        poly_pts = np.array(box["precise_global_poly"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img_final, [poly_pts], isClosed=True, color=(255, 100, 0), thickness=2)

    # ✨ 核心视觉调整：先不要画距离线，只画出最终出现在匹配结果里的实体物理圆心
    for assign in final_assignments:
        c_name = assign["color_name"]
        cc_pt = assign["cc"]
        core_color = vis_colors.get(c_name, (255, 255, 255))
        # 绘制实体点：对应颜色实心球 + 白色核心提示
        cv2.circle(vis_img_final, (int(cc_pt[0]), int(cc_pt[1])), 7, core_color, -1, cv2.LINE_AA)
        cv2.circle(vis_img_final, (int(cc_pt[0]), int(cc_pt[1])), 2, (255, 255, 255), -1, cv2.LINE_AA)

    os.makedirs(output_vis_dir, exist_ok=True)
    final_out_path = os.path.join(output_vis_dir, f"{image_base_name}_final_matching_result.jpg")
    cv2.imwrite(final_out_path, vis_img_final)
    print(f"[SUCCESS] 零偏差高保真二值化结果图已保存至: {final_out_path}")

    # ==================== ✨ 返回关系字典（内含精准逆向 Micro 局部坐标） ====================
    output_report_dict = {}

    for box_key, box in text_boxes.items():
        text_content = box["text"]
        # 确保每个文本拥有一个唯一的字典条目进行数据承载
        if text_content not in output_report_dict:
            output_report_dict[text_content] = {
                "text": text_content,
                "color_centers": {c: [] for c in colors}
            }

    # 遍历最终的分配关系，将其逆向解算为 micro 局部坐标并打包
    for assign in final_assignments:
        box_key = assign["box_key"]
        color_name = assign["color_name"]
        global_cc = assign["cc"]

        box = text_boxes[box_key]
        text_content = box["text"]
        patch_left, patch_top = box["patch_origin"]

        # 核心逆向转换：大图全局坐标减去切片左上角原点 = micro 局部坐标系坐标
        micro_x = round(global_cc[0] - patch_left, 2)
        micro_y = round(global_cc[1] - patch_top, 2)

        point_payload = {
            "micro_coord": [micro_x, micro_y],  # 满足计算并输出 micro poly 关系的要求
            "global_coord": [round(global_cc[0], 2), round(global_cc[1], 2)],
            "match_stage": assign["type"]
        }
        output_report_dict[text_content]["color_centers"][color_name].append(point_payload)

    return output_report_dict


if __name__ == "__main__":
    # 调用示例
    final_mapped_data = generate_accurate_micro_relations_vis("t6.jpg", "results")
    print("\n[INFO] 任务完成，数据已成功 return。")