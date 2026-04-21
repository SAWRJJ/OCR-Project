import cv2
import numpy as np
import json
import os


def find_left_to_right_dark_region(img, dark_threshold=125):
    """
    找到从左侧边到右侧边的深色连通区域
    返回: 中线行号，如果没找到返回None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    labeled = np.zeros_like(dark_mask, dtype=np.int32)
    label = 0
    regions = []

    from collections import deque

    for row in range(h):
        for col in range(w):
            if dark_mask[row, col] == 1 and labeled[row, col] == 0:
                label += 1
                region_pixels = []
                queue = deque([(row, col)])
                labeled[row, col] = label

                while queue:
                    r, c = queue.popleft()
                    region_pixels.append((r, c))

                    neighbors = [
                        (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
                        (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1),
                    ]

                    for nr, nc in neighbors:
                        if 0 <= nr < h and 0 <= nc < w:
                            if dark_mask[nr, nc] == 1 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = label
                                queue.append((nr, nc))

                if region_pixels:
                    regions.append({
                        'label': label,
                        'pixels': region_pixels,
                        'min_col': min(p[1] for p in region_pixels),
                        'max_col': max(p[1] for p in region_pixels),
                        'min_row': min(p[0] for p in region_pixels),
                        'max_row': max(p[0] for p in region_pixels),
                    })

    left_to_right_regions = [r for r in regions if r['min_col'] == 0 and r['max_col'] == w - 1]

    if left_to_right_regions:
        largest_region = max(left_to_right_regions, key=lambda r: len(r['pixels']))
        center_row = (largest_region['min_row'] + largest_region['max_row']) // 2
        print(f"找到左到右连通区域: 像素数={len(largest_region['pixels'])}, "
              f"行范围=[{largest_region['min_row']}, {largest_region['max_row']}], "
              f"中线行号={center_row}")
        return center_row, largest_region

    return None, None


def crop_by_centerline(img, center_row):
    """
    根据中线位置裁剪图片
    如果中线在图片下方，保留中线上方的图片
    如果中线在图片上方，保留中线下方的图片
    """
    h, w = img.shape[:2]
    mid = h // 2

    if center_row > mid:
        print(f"中线({center_row})在图片下方({mid})，保留中线上方 [0, {center_row})")
        cropped = img[0:center_row, :]
    else:
        print(f"中线({center_row})在图片上方({mid})，保留中下方 [{center_row}, {h})")
        cropped = img[center_row:h, :]

    return cropped


def visualize_crop(img, center_row, output_path):
    """
    可视化裁剪位置
    """
    h, w = img.shape[:2]
    vis_img = img.copy()

    mid = h // 2

    color = (0, 255, 0) if center_row > mid else (0, 0, 255)
    cv2.line(vis_img, (0, center_row), (w - 1, center_row), color, 2)

    cv2.line(vis_img, (0, mid), (w - 1, mid), (255, 255, 0), 1)

    label = "中线"
    cv2.putText(vis_img, f"{label}: {center_row}", (10, center_row - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(vis_img, f"图中线: {mid}", (10, mid - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    direction = "保留上方" if center_row > mid else "保留下方"
    cv2.putText(vis_img, f"裁剪方向: {direction}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite(output_path, vis_img)
    print(f"裁剪可视化已保存: {output_path}")

    return vis_img


def remove_dark_region(img, region):
    """
    将连通区域的像素全部转换为白色
    """
    result = img.copy()
    for r, c in region['pixels']:
        result[r, c] = (255, 255, 255)
    return result


def calculate_angle(poly):
    """
    计算文本框的水平倾斜角度
    poly: 四边形四个顶点坐标列表
    返回: 角度（度）
    """
    points = np.array(poly)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min_idx = np.argmin(x_coords)
    x_max_idx = np.argmax(x_coords)

    p1 = points[x_min_idx]
    p2 = points[x_max_idx]

    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]

    angle = np.arctan2(dy, dx) * 180 / np.pi

    return angle


def get_bounding_box(poly):
    """获取四边形的边界框"""
    points = np.array(poly)
    x_min = int(points[:, 0].min())
    x_max = int(points[:, 0].max())
    y_min = int(points[:, 1].min())
    y_max = int(points[:, 1].max())
    return x_min, y_min, x_max, y_max


def scan_rows_with_dark_pixel_ratio(img, angle, step=2, direction='top_to_bottom', bounding_box=None):
    """
    按行扫描深色像素占比
    img: 输入图像
    angle: 文本框倾斜角度
    step: 扫描步长（每2像素检查一次）
    direction: 'top_to_bottom' 或 'bottom_to_top'
    bounding_box: 可选的边界框 (x_min, y_min, x_max, y_max)
    返回: 所有行的占比信息列表
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if bounding_box:
        x_min, y_min, x_max, y_max = bounding_box
    else:
        x_min, y_min, x_max, y_max = 0, 0, w, h

    x_min, y_min = 0, 0
    x_max, y_max = w, h

    dark_mask = (gray < 125).astype(np.uint8)

    rotated_gray = gray
    rotated_dark_mask = dark_mask
    rotated_bbox = (x_min, y_min, x_max, y_max)

    if abs(angle) > 0.5:
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_gray = cv2.warpAffine(gray, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        rotated_dark_mask = cv2.warpAffine(dark_mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    all_row_ratios = []

    if direction == 'top_to_bottom':
        row_indices = range(y_min, y_max, step)
    else:
        row_indices = range(y_max - 1, y_min - 1, -step)

    for row in row_indices:
        if row < 0 or row >= h:
            continue

        row_pixels = rotated_dark_mask[row, x_min:x_max]
        dark_count = np.sum(row_pixels > 0)
        total_count = x_max - x_min

        dark_ratio = dark_count / total_count if total_count > 0 else 0

        all_row_ratios.append((row, dark_ratio))

    return all_row_ratios, rotated_gray, rotated_dark_mask


def visualize_all_rows(img, rows_info, angle, direction, threshold, output_path):
    """
    可视化所有扫描的行，用颜色区分不同占比
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    vis_img = img.copy()

    if abs(angle) > 0.5:
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_vis = cv2.warpAffine(vis_img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    else:
        rotated_vis = vis_img

    max_ratio = max(ratio for _, ratio in rows_info) if rows_info else 1.0
    if max_ratio == 0:
        max_ratio = 1.0

    high_ratio_rows = []

    for row, ratio in rows_info:
        if ratio >= threshold:
            high_ratio_rows.append((row, ratio))

        normalized_ratio = ratio / max_ratio

        if ratio >= threshold:
            color = (0, 255, 0)
        elif ratio >= threshold * 0.7:
            color = (0, 255, 255)
        elif ratio >= threshold * 0.5:
            color = (0, 165, 255)
        else:
            b = int(255 * normalized_ratio)
            color = (0, 0, b)

        cv2.line(rotated_vis, (0, row), (w - 1, row), color, 1)

    if abs(angle) > 0.5:
        inv_rotation = cv2.getRotationMatrix2D(center, -angle, 1.0)
        final_vis = cv2.warpAffine(rotated_vis, inv_rotation, (w, h), flags=cv2.INTER_LINEAR)
    else:
        final_vis = rotated_vis

    direction_label = "上到下" if direction == 'top_to_bottom' else "下到上"

    cv2.putText(final_vis, f"{direction_label} 倾斜角:{angle:.2f}度", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(final_vis, f"阈值:{threshold:.0%} 深色行:{len(high_ratio_rows)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    legend_y = 80
    cv2.putText(final_vis, ">=80%: 绿色", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(final_vis, ">=56%: 黄色", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(final_vis, ">=40%: 橙色", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    cv2.putText(final_vis, "<40%: 蓝色渐变", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imwrite(output_path, final_vis)
    print(f"可视化结果已保存: {output_path}")

    return final_vis, high_ratio_rows


def process_image_with_json(image_path, json_path, output_dir="output"):
    """
    处理单张图像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    micro_poly = data.get('micro_poly', [])
    if not micro_poly:
        print("JSON中未找到micro_poly字段")
        return

    vis_img = img.copy()
    poly_points = np.array(micro_poly, dtype=np.int32)
    cv2.polylines(vis_img, [poly_points], isClosed=True, color=(0, 255, 0), thickness=2)

    for i, point in enumerate(micro_poly):
        cv2.circle(vis_img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
        cv2.putText(vis_img, str(i), (int(point[0]) + 5, int(point[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    poly_vis_path = os.path.join(output_dir, f"{base_name}_poly_vis.png")
    cv2.imwrite(poly_vis_path, vis_img)
    print(f"micro_poly可视化已保存: {poly_vis_path}")

    angle = calculate_angle(micro_poly)
    angle = 0
    print(f"文本框倾斜角度: {angle:.2f} 度")

    bbox = get_bounding_box(micro_poly)
    print(f"文本框边界框: x={bbox[0]}-{bbox[2]}, y={bbox[1]}-{bbox[3]}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    threshold = 0.8

    for direction in ['top_to_bottom', 'bottom_to_top']:
        rows_info, rotated_gray, rotated_dark_mask = scan_rows_with_dark_pixel_ratio(
            img, angle, step=2, direction=direction, bounding_box=bbox
        )

        print(f"\n{direction}方向扫描结果:")
        print(f"总共扫描 {len(rows_info)} 行")

        high_rows = [(row, ratio) for row, ratio in rows_info if ratio >= threshold]
        print(f"深色像素占比 >= {threshold:.0%} 的行: {len(high_rows)} 行")

        top_10 = sorted(rows_info, key=lambda x: x[1], reverse=True)[:10]
        print(f"深色像素占比Top 10的行:")
        for row, ratio in top_10:
            status = "✓" if ratio >= threshold else ""
            print(f"  行 {row}: {ratio:.4f} ({ratio*100:.2f}%) {status}")

        output_path = os.path.join(output_dir, f"{base_name}_{direction}_scan.png")
        visualize_all_rows(img, rows_info, angle, direction, threshold, output_path)


if __name__ == "__main__":
    image_path = r"d:\work\ocr+Transformer\little_light\micro_0045_XI.jpg"
    json_path = r"d:\work\ocr+Transformer\little_light\micro_0045_XI.json"

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    print("=" * 50)
    print("步骤1: 找到左到右深色连通区域")
    print("=" * 50)

    center_row, region = find_left_to_right_dark_region(img, dark_threshold=125)

    if region is not None:
        print("=" * 50)
        print("步骤2: 将连通区域转换为白色")
        print("=" * 50)
        result = remove_dark_region(img, region)
        result_path = r"d:\work\ocr+Transformer\little_light\micro_0045_XI_white.png"
        cv2.imwrite(result_path, result)
        print(f"处理后的图片已保存: {result_path}")
    else:
        print("未找到左到右贯通的深色区域")
