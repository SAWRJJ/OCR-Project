import cv2
import json
import os
import math
import numpy as np
import time
from pathlib import Path

# 可视化结果输出目录
VIS_DIR = "visualization_output"


def has_x_or_s(text):
    if not text:
        return False
    text_upper = text.upper()
    return 'X' in text_upper or 'S' in text_upper


def extract_color_mask(img, color):
    """提取指定颜色的平滑面具"""
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
    mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    return mask_uint8


def get_color_cluster_centers(mask_uint8):
    """通过连通域提取每个颜色集群的中心点 (Centroids)"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cluster_centers = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])
        cluster_centers.append((cx, cy))
    return cluster_centers


def find_cluster_centers(points, distance_threshold=25):
    """根据距离聚集点集，并返回每个聚集点集中最靠近几何中心的真实点"""
    if not points:
        return []
    points = np.array(points)
    n = len(points)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    dist = np.sqrt((points[current][0] - points[j][0]) ** 2 + (points[current][1] - points[j][1]) ** 2)
                    if dist < distance_threshold:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)

    center_points = []
    for cluster in clusters:
        cluster_points_arr = points[cluster]
        centroid = np.mean(cluster_points_arr, axis=0)
        distances = [np.sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2) for p in cluster_points_arr]
        min_idx = np.argmin(distances)
        center_points.append(tuple(cluster_points_arr[min_idx]))

    return center_points


def map_micro_to_global_pt(micro_pt, patch_poly):
    """通过 patch_poly 的左上角原点，将单点从 micro 局部坐标系平移映射回全局大图坐标系"""
    patch_left = patch_poly[0][0]
    patch_top = patch_poly[0][1]
    return [float(micro_pt[0] + patch_left), float(micro_pt[1] + patch_top)]


def map_micro_poly_to_global_poly(detail_micro_poly, patch_poly):
    """将 re_ocr_details 里的具体文本 micro_poly 整体平移映射回全局大图坐标"""
    patch_left = patch_poly[0][0]
    patch_top = patch_poly[0][1]
    return [[float(pt[0] + patch_left), float(pt[1] + patch_top)] for pt in detail_micro_poly[:4]]


# ==================== 可视化函数 ====================

def save_filtered_json_pts_visualization(base_img, filtered_details_list):
    """专门可视化被过滤掉的 JSON 点"""
    vis_img = base_img.copy()

    for item in filtered_details_list:
        j_pt = item["json_pt"]
        c_pt = item["collected_pt"]
        color_name = item["color_name"]
        dist = item["distance"]

        pt_j = (int(j_pt[0]), int(j_pt[1]))
        pt_c = (int(c_pt[0]), int(c_pt[1]))

        cv2.line(vis_img, pt_j, pt_c, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(vis_img, pt_j, 5, (0, 0, 255), 1)
        cv2.circle(vis_img, pt_c, 3, (128, 128, 128), -1)

        label_text = f"Filtered {color_name}: {dist}px"
        cv2.putText(vis_img, label_text, (pt_j[0] + 6, pt_j[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    os.makedirs(VIS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    output_path = f"{VIS_DIR}/filtered_json_points_{ts}.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"[INFO] 已生成过滤点单独可视化文件: {output_path}")


def save_global_distance_visualization(base_img, text_boxes_dict, final_assignments,image_base_name):
    vis_img = base_img.copy()
    vis_colors = {
        'red': (0, 0, 255), 'blue': (255, 0, 0),
        'green': (0, 255, 0), 'yellow': (0, 255, 255), 'white': (200, 200, 200)
    }

    for box_key, box in text_boxes_dict.items():
        precise_poly = box["precise_global_poly"]
        poly_pts = np.array(precise_poly[:4], np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [poly_pts], True, (255, 255, 255), 2)
        for pt in poly_pts:
            cv2.circle(vis_img, tuple(pt[0]), 5, (0, 255, 255), -1)
        cv2.circle(vis_img, (int(box["poly_center"][0]), int(box["poly_center"][1])), 8, (255, 255, 0), -1)

        for c_name, g_centers in box["global_json_centers"].items():
            c_color = vis_colors.get(c_name, (255, 255, 255))
            for gc in g_centers:
                cv2.circle(vis_img, (int(gc[0]), int(gc[1])), 15, c_color, 1)

    for assign in final_assignments:
        color_name = assign["color_name"]
        cc = assign["cluster_center"]
        dist = assign["min_distance"]
        match_type = assign["match_type"]
        box_key = assign["assigned_box_key"]

        box_center = text_boxes_dict[box_key]["poly_center"]
        start_pt = (int(cc[0]), int(cc[1]))
        end_pt = (int(box_center[0]), int(box_center[1]))

        # 根据匹配阶段切换连线颜色：Targeted(绿), Nearest(品红), Forced_Nearest(橘黄)
        if match_type == "Targeted":
            line_color = (0, 255, 0)
        elif match_type == "Nearest":
            line_color = (255, 0, 255)
        else:
            line_color = (0, 165, 255)  # Orange for Forced_Nearest

        cv2.line(vis_img, start_pt, end_pt, line_color, 2, cv2.LINE_AA)

        label_text = f"{color_name}({match_type}: {dist}px)"
        cv2.putText(vis_img, label_text, (start_pt[0] + 8, start_pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1, cv2.LINE_AA)
        cv2.circle(vis_img, start_pt, 6, vis_colors.get(color_name, (255, 255, 255)), -1)

    os.makedirs(VIS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f"{VIS_DIR}/{image_base_name}_strict_hybrid_matching_{ts}.jpg", vis_img)


import cv2
import json
import math
import os
from pathlib import Path


def process_distances(img_path, results_dir, output_dir="output_reports"):
    img = cv2.imread(img_path)
    if img is None: return {}

    # ✨ 新增：提取当前大图的基础文件名（不含路径和后缀），例如 "big_scene_01"
    image_base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 1. 提取大图所有收集到的颜色集群中心
    colors_to_extract = ['red', 'blue', 'green', 'yellow']
    color_clusters = {}
    for color in colors_to_extract:
        mask = extract_color_mask(img, color)
        color_clusters[color] = get_color_cluster_centers(mask)

    color_clusters['yellow'] = find_cluster_centers(color_clusters['yellow'], distance_threshold=25)

    results_path = Path(results_dir)
    json_files = list(results_path.glob("*.json"))

    text_boxes = {}

    # 2. 解析文本框并建立全局映射
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        patch_poly = data.get('patch_poly', [])
        if not patch_poly: continue

        details = data.get('re_ocr_details', [])
        for detail_idx, detail in enumerate(details):
            text = detail.get('text', '')
            detail_micro_poly = detail.get('micro_poly', [])
            if not detail_micro_poly: continue

            precise_global_poly = map_micro_poly_to_global_poly(detail_micro_poly, patch_poly)

            x_coords = [pt[0] for pt in precise_global_poly[:4]]
            y_coords = [pt[1] for pt in precise_global_poly[:4]]
            poly_center = (sum(x_coords) / 4.0, sum(y_coords) / 4.0)
            poly_vertices = [[pt[0], pt[1]] for pt in precise_global_poly[:4]]

            global_json_centers = {"red": [], "yellow": [], "green": [], "blue": []}
            centers_sep = detail.get('color_centers_separate', {})

            for c_name in global_json_centers.keys():
                local_centers = centers_sep.get(c_name, [])
                for lp in local_centers:
                    g_x, g_y = map_micro_to_global_pt(lp, patch_poly)
                    global_json_centers[c_name].append((g_x, g_y))

            global_white_centers = []
            for w_pt in centers_sep.get('white', []):
                g_w_x, g_w_y = map_micro_to_global_pt(w_pt, patch_poly)
                global_white_centers.append((g_w_x, g_w_y))

            is_xs = has_x_or_s(text)
            target_colors = ['red', 'yellow', 'green'] if is_xs else ['red', 'blue']

            box_key = f"{json_file.name}_detail_{detail_idx}"
            text_boxes[box_key] = {
                "json_file_name": json_file.name, "text": text, "is_xs": is_xs,
                "precise_global_poly": precise_global_poly, "poly_center": poly_center,
                "poly_vertices": poly_vertices, "global_white_centers": global_white_centers,
                "global_json_centers": global_json_centers, "target_colors": target_colors,
                "assigned_results": {}
            }

    # 利用图像收集到的点对 JSON 映射点进行双限阈值筛选过滤
    filtered_records = []
    for box_key, box in text_boxes.items():
        for color_name, json_pts in list(box["global_json_centers"].items()):
            valid_json_pts = []
            collected_pts = color_clusters.get(color_name, [])

            for j_pt in json_pts:
                should_filter = False
                closest_collected_pt = None
                min_dist_found = float('inf')

                for c_pt in collected_pts:
                    dist = math.hypot(j_pt[0] - c_pt[0], j_pt[1] - c_pt[1])
                    if dist < min_dist_found:
                        min_dist_found = dist
                        closest_collected_pt = c_pt

                if 5.0 < min_dist_found <= 20.0:
                    should_filter = True
                    filtered_records.append({
                        "box_key": box_key, "color_name": color_name, "json_pt": j_pt,
                        "collected_pt": closest_collected_pt, "distance": round(min_dist_found, 2)
                    })

                if not should_filter:
                    valid_json_pts.append(j_pt)

            box["global_json_centers"][color_name] = valid_json_pts

    if filtered_records:
        print(f"[INFO] 触发筛选限制：共过滤掉 {len(filtered_records)} 个不满足要求的 JSON 点")
        save_filtered_json_pts_visualization(img, filtered_records)

    # 3. 混合匹配核心
    final_assignments = []
    SEARCH_RADIUS = 15.0
    matched_points_registry = set()

    for color_name, centers in color_clusters.items():
        for cc_idx, cc in enumerate(centers):
            matched_in_stage1 = False
            stage1_box_key = None
            stage1_dist = float('inf')

            # ----------------- 【阶段一：专属靶向筛查】 -----------------
            for box_key, box in text_boxes.items():
                if not box["is_xs"]: continue
                if color_name not in box["target_colors"]: continue

                my_json_pts = box["global_json_centers"].get(color_name, [])
                for j_pt in my_json_pts:
                    dist = math.hypot(cc[0] - j_pt[0], cc[1] - j_pt[1])
                    if dist <= SEARCH_RADIUS:
                        stage1_dist = dist
                        stage1_box_key = box_key
                        matched_in_stage1 = True
                        break
                if matched_in_stage1: break

            if matched_in_stage1:
                assignment_entry = {
                    "color_name": color_name, "cluster_center": [round(cc[0], 2), round(cc[1], 2)],
                    "assigned_box_key": stage1_box_key, "closest_to": "json_original_belonging",
                    "min_distance": round(stage1_dist, 2), "match_type": "Targeted"
                }
                final_assignments.append(assignment_entry)
                matched_points_registry.add((color_name, round(cc[0], 2), round(cc[1], 2)))

                text_boxes[stage1_box_key]["assigned_results"][color_name] = {
                    "min_distance": round(stage1_dist, 2),
                    "cluster_center": [round(cc[0], 2), round(cc[1], 2)],
                    "closest_to": "json_original_belonging"
                }
                continue

            # ----------------- 【阶段二：漏网之鱼就近收编】 -----------------
            best_stage2_match = None
            global_min_dist = float('inf')

            for box_key, box in text_boxes.items():
                if color_name not in box["target_colors"]: continue

                reference_points = [("poly_center", box["poly_center"])]
                for v_idx, v in enumerate(box["poly_vertices"]):
                    reference_points.append((f"vertex_{v_idx}", v))
                for w_idx, w in enumerate(box["global_white_centers"]):
                    reference_points.append((f"white_center_{w_idx}", w))

                for pt_type, pt_coords in reference_points:
                    dist = math.hypot(cc[0] - pt_coords[0], cc[1] - pt_coords[1])
                    if dist > 350: continue

                    if dist < global_min_dist:
                        global_min_dist = dist
                        best_stage2_match = {
                            "assigned_box_key": box_key, "closest_to": pt_type,
                            "min_distance": round(dist, 2), "match_type": "Nearest"
                        }

            if best_stage2_match:
                assignment_entry = {
                    "color_name": color_name, "cluster_center": [round(cc[0], 2), round(cc[1], 2)],
                    **best_stage2_match
                }
                final_assignments.append(assignment_entry)
                matched_points_registry.add((color_name, round(cc[0], 2), round(cc[1], 2)))

                bk = best_stage2_match["assigned_box_key"]
                text_boxes[bk]["assigned_results"][color_name] = {
                    "min_distance": best_stage2_match["min_distance"],
                    "cluster_center": [round(cc[0], 2), round(cc[1], 2)],
                    "closest_to": best_stage2_match["closest_to"]
                }

    # ----------------- 【阶段三：无条件全局就近强制收编】 -----------------
    forced_count = 0
    for color_name, centers in color_clusters.items():
        for cc in centers:
            cc_round = (round(cc[0], 2), round(cc[1], 2))

            if (color_name, cc_round[0], cc_round[1]) not in matched_points_registry:
                best_forced_match = None
                forced_min_dist = float('inf')

                for box_key, box in text_boxes.items():
                    reference_points = [("poly_center", box["poly_center"])]
                    for v_idx, v in enumerate(box["poly_vertices"]):
                        reference_points.append((f"vertex_{v_idx}", v))
                    for w_idx, w in enumerate(box["global_white_centers"]):
                        reference_points.append((f"white_center_{w_idx}", w))

                    for pt_type, pt_coords in reference_points:
                        dist = math.hypot(cc[0] - pt_coords[0], cc[1] - pt_coords[1])
                        if dist < forced_min_dist:
                            forced_min_dist = dist
                            best_forced_match = {
                                "assigned_box_key": box_key,
                                "closest_to": f"forced_{pt_type}",
                                "min_distance": round(dist, 2),
                                "match_type": "Forced_Nearest"
                            }

                if best_forced_match:
                    forced_count += 1
                    assignment_entry = {
                        "color_name": color_name,
                        "cluster_center": [cc_round[0], cc_round[1]],
                        **best_forced_match
                    }
                    final_assignments.append(assignment_entry)
                    matched_points_registry.add((color_name, cc_round[0], cc_round[1]))

                    bk = best_forced_match["assigned_box_key"]
                    save_key = color_name if color_name not in text_boxes[bk]["assigned_results"] else f"{color_name}_forced"
                    text_boxes[bk]["assigned_results"][save_key] = {
                        "min_distance": best_forced_match["min_distance"],
                        "cluster_center": [cc_round[0], cc_round[1]],
                        "closest_to": best_forced_match["closest_to"]
                    }

    print(f"\n[INFO] 阶段三无条件匹配完毕：共对 {forced_count} 个未匹配点执行了强行就近文本框归属分配。")

    # 核对打印
    print("\n" + "=" * 40 + " 剩余未匹配彩色中心点报告 " + "=" * 40)
    unmatched_count = 0
    for color_name, centers in color_clusters.items():
        for cc in centers:
            cc_round = (round(cc[0], 2), round(cc[1], 2))
            if (color_name, cc_round[0], cc_round[1]) not in matched_points_registry:
                unmatched_count += 1
                print(f"[未匹配] 颜色: {color_name:<6} -> 图像全局坐标: X={cc_round[0]:<8}, Y={cc_round[1]:<8}")

    if unmatched_count == 0:
        print("[INFO] 极其完美！所有提取到的像素检测中心点已 100% 强制分配入相应的文本框。")
    else:
        print(f"[INFO] 警告：仍有 {unmatched_count} 个像素检测中心未匹配。")
    print("=" * 104 + "\n")

    # 4. 可视化与输出报告
    if len(final_assignments) > 0:
        # ✨ 修改：这里将提取出的 image_base_name 作为第 4 个参数传入可视化函数
        save_global_distance_visualization(img, text_boxes, final_assignments, image_base_name)

    has_xs_report, no_xs_report = [], []
    for box_key, box in text_boxes.items():
        distance_summary = {}
        for k, v in box["assigned_results"].items():
            distance_summary[k] = v

        entry = {
            "json_file": box["json_file_name"], "text": box["text"],
            "poly_center": [round(box["poly_center"][0], 2), round(box["poly_center"][1], 2)],
            "distances": distance_summary
        }
        has_xs_report.append(entry) if box["is_xs"] else no_xs_report.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "report_with_XS.json"), "w", encoding="utf-8") as f:
        json.dump(has_xs_report, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "report_without_XS.json"), "w", encoding="utf-8") as f:
        json.dump(no_xs_report, f, indent=4, ensure_ascii=False)

    # ==================== 构建全局返回映射字典 ====================
    text_to_centers_map = {}
    for box_key, box in text_boxes.items():
        text_content = box["text"]
        unique_key = f"{text_content}"  # 保持纯文本作为 key

        color_points = {}
        for key_name, info in box["assigned_results"].items():
            color_points[key_name] = info["cluster_center"]

        # 同步把白色中心点也带入返回字典中，以便回填
        if box.get("global_white_centers"):
            color_points["white"] = [[round(pt[0], 2), round(pt[1], 2)] for pt in box["global_white_centers"]]

        text_to_centers_map[unique_key] = {
            "text": text_content,
            "json_file": box["json_file_name"],
            "poly_center": [round(box["poly_center"][0], 2), round(box["poly_center"][1], 2)],
            "color_centers": color_points
        }

    return text_to_centers_map

def update_and_return_all_results0(all_results, text_to_centers_map):
    """
    根据全局映射字典，替换 all_results 中对应的 color_centers_separate 局部数据。
    替换完成后，直接 return 修改完的 all_results。
    """
    for item in all_results:
        details = item.get("details", [])
        for detail in details:
            text_content = detail.get("text", "")

            # 通过纯文本 key 寻找该文本在大图中计算出的全局中心点
            if text_content in text_to_centers_map:
                global_info = text_to_centers_map[text_content]
                global_color_centers = global_info.get("color_centers", {})

                # 初始化一个干净的、用于承载全局坐标的容器
                new_centers_separate = {
                    "blue": [], "green": [], "red": [], "white": [], "yellow": []
                }

                # 开始分拣映射坐标
                for color_key, coords in global_color_centers.items():
                    # 剥离可能存在的强绑尾缀（例如阶段三产生的 'red_forced' -> 'red'）
                    base_color = color_key.split("_")[0]

                    if base_color in new_centers_separate:
                        # 如果 coords 是多级列表（例如白色的多个点 [[x1,y1], [x2,y2]]）
                        if isinstance(coords[0], list):
                            for single_pt in coords:
                                new_centers_separate[base_color].append((single_pt[0], single_pt[1]))
                        else:
                            # 单个点情况 [x, y] -> 转为元组 (x, y)
                            new_centers_separate[base_color].append((coords[0], coords[1]))

                # 覆盖原有的局部坐标数据
                detail["color_centers_separate"] = new_centers_separate

    # 按照需求：直接将修改完毕的整个对象完整 return 出来
    return all_results


def update_and_return_all_results(all_results, text_to_centers_map):
    """
    根据全新的全局映射字典结构，替换 all_results 中对应的 color_centers_separate 局部数据。
    支持新版结构：每个颜色对应一个包含 dict 的 list: [{'micro_coord': [x, y], 'global_coord': [x, y], 'match_stage': ...}]
    替换完成后，直接 return 修改完的 all_results。
    """
    for item in all_results:
        details = item.get("details", [])
        for detail in details:
            text_content = detail.get("text", "")

            # 通过纯文本 key 寻找该文本在大图中计算出的全局中心点
            if text_content in text_to_centers_map:
                global_info = text_to_centers_map[text_content]
                global_color_centers = global_info.get("color_centers", {})

                # 初始化一个干净的、用于承载计算后局部坐标的容器
                new_centers_separate = {
                    "blue": [], "green": [], "red": [], "white": [], "yellow": []
                }

                # 开始分拣映射坐标
                for color_key, point_list in global_color_centers.items():
                    # 剥离可能存在的强绑尾缀（例如阶段三产生的 'red_forced' -> 'red'）
                    base_color = color_key.split("_")[0]

                    if base_color in new_centers_separate and point_list:
                        # 💡 适配新数据结构：point_list 此时是包含多个点字典的列表
                        for pt_info in point_list:
                            if isinstance(pt_info, dict) and "micro_coord" in pt_info:
                                # 提取计算好的 micro_coord 局部坐标
                                m_coord = pt_info["micro_coord"]
                                # 转换为元组 (x, y) 追加进对应的颜色列表中
                                new_centers_separate[base_color].append((m_coord[0], m_coord[1]))

                            # 兼容性备用边界：万一某些历史遗留数据还是老格式的纯列表 [x, y]
                            elif isinstance(pt_info, list):
                                new_centers_separate[base_color].append((pt_info[0], pt_info[1]))

                # 覆盖原有的局部坐标数据
                detail["color_centers_separate"] = new_centers_separate

    # 直接将修改完毕的整个对象完整 return 出来
    return all_results

if __name__ == "__main__":
    res = process_distances("t6.jpg", "results")
    print(res)