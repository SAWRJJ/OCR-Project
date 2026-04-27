import cv2
import numpy as np
from collections import deque
import os


def find_closed_dark_regions(img_path, dark_threshold=125, min_circularity=0.7):
    """
    找出完全封闭的深色区域（闭合圆环）
    通过检查深色区域的边界是否完全被深色像素包围（即内部有空洞）
    只有圆度 >= min_circularity 的才被认为是闭合圆环
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    labeled = np.zeros_like(dark_mask, dtype=np.int32)
    label = 0
    regions = []

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

    closed_regions = []
    for r in regions:
        touches_boundary = (
            r['min_row'] == 0 or
            r['max_row'] == h - 1 or
            r['min_col'] == 0 or
            r['max_col'] == w - 1
        )

        if not touches_boundary:
            region_mask = np.zeros((h, w), dtype=np.uint8)
            for pixel in r['pixels']:
                region_mask[pixel[0], pixel[1]] = 1

            x_min, y_min = r['min_col'], r['min_row']
            x_max, y_max = r['max_col'], r['max_row']

            hole_pixels = 0
            total_pixels = 0

            for row in range(y_min + 1, y_max):
                for col in range(x_min + 1, x_max):
                    if region_mask[row, col] == 0:
                        hole_pixels += 1

            bounding_area = (x_max - x_min) * (y_max - y_min)
            dark_area = len(r['pixels'])

            if hole_pixels > 0:
                circularity, largest_contour = calculate_circularity(r, (h, w))
                if circularity >= min_circularity:
                    center, radius = calculate_circle_from_contour(largest_contour)
                    r['center'] = center
                    r['radius'] = radius
                    print(center, radius)
                    closed_regions.append(r)

    return closed_regions


def calculate_circularity(region, img_shape):
    """
    计算闭合圆环的真圆度
    使用公式: 4 * π * Area / Perimeter²
    值越接近1越圆
    """
    pixels = region['pixels']
    if not pixels:
        return 0.0, None

    h, w = img_shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    for pixel in pixels:
        region_mask[pixel[0], pixel[1]] = 255

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)

    if perimeter == 0:
        return 0.0, None

    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity, largest_contour


def calculate_circle_from_contour(contour):
    """
    从轮廓计算最小外接圆的圆心和半径

    参数:
        contour: OpenCV轮廓

    返回:
        tuple: (center, radius) - center为(x, y)元组，radius为半径
    """
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def find_boundary_connected_dark_pixels(img_path, dark_threshold=125):
    """
    找出与边界相连的深色像素连通区域
    使用BFS进行连通区域标记
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    labeled = np.zeros_like(dark_mask, dtype=np.int32)
    label = 0
    regions = []

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

    boundary_regions = []
    for r in regions:
        touches_boundary = (
            r['min_row'] == 0 or
            r['max_row'] == h - 1 or
            r['min_col'] == 0 or
            r['max_col'] == w - 1
        )
        if touches_boundary:
            boundary_regions.append(r)

    print(f"图像尺寸: {w}x{h}")
    print(f"深色阈值: {dark_threshold}")
    print(f"总连通区域数: {len(regions)}")
    print(f"与边界相连的深色区域数: {len(boundary_regions)}")

    for i, r in enumerate(boundary_regions):
        boundary_sides = []
        if r['min_row'] == 0:
            boundary_sides.append('上')
        if r['max_row'] == h - 1:
            boundary_sides.append('下')
        if r['min_col'] == 0:
            boundary_sides.append('左')
        if r['max_col'] == w - 1:
            boundary_sides.append('右')

        print(f"  区域{i+1}: 像素数={len(r['pixels'])}, "
              f"行范围=[{r['min_row']}, {r['max_row']}], "
              f"列范围=[{r['min_col']}, {r['max_col']}], "
              f"接触边界: {', '.join(boundary_sides)}")

    return img, boundary_regions


def remove_dark_regions(img, boundary_regions, closed_regions, output_path):
    """
    将所有被判定为深色像素的区域设置为白色
    包括边界连通区域和闭合圆环
    """
    result = img.copy()

    all_pixels = set()
    for r in boundary_regions:
        for pixel in r['pixels']:
            all_pixels.add(pixel)

    for r in closed_regions:
        for pixel in r['pixels']:
            all_pixels.add(pixel)

    for r, c in all_pixels:
        result[r, c] = (255, 255, 255)

    cv2.imwrite(output_path, result)
    print(f"已将 {len(all_pixels)} 个深色像素设置为白色")
    print(f"处理后的图片已保存: {output_path}")

    return result


def visualize_all_dark_regions(img, boundary_regions, closed_regions, output_path):
    """
    在原图上叠加显示所有需要关注的深色像素
    - 与边界相连的深色像素（绿色）
    - 闭合圆环（红色）
    """
    h, w = img.shape[:2]
    vis_img = img.copy()

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for r in boundary_regions:
        for pixel in r['pixels']:
            overlay[pixel[0], pixel[1]] = [0, 255, 0]

    for r in closed_regions:
        for pixel in r['pixels']:
            overlay[pixel[0], pixel[1]] = [0, 0, 255]

    y_offset = 30
    cv2.putText(vis_img, f"边界连通: {len(boundary_regions)}个区域, {sum(len(r['pixels']) for r in boundary_regions)}像素",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    y_offset += 25
    cv2.putText(vis_img, f"闭合圆环: {len(closed_regions)}个区域, {sum(len(r['pixels']) for r in closed_regions)}像素",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    cv2.imwrite(output_path, vis_img)
    print(f"综合可视化结果已保存: {output_path}")

    return vis_img


def visualize_boundary_dark_pixels(img, regions, output_path):
    """
    可视化与边界相连的深色像素连通区域
    """
    h, w = img.shape[:2]
    vis_img = img.copy()

    if regions:
        all_boundary_pixels = set()
        for r in regions:
            for pixel in r['pixels']:
                all_boundary_pixels.add(pixel)

        print(f"总共标记 {len(all_boundary_pixels)} 个与边界相连的深色像素")

        for r, c in all_boundary_pixels:
            vis_img[r, c] = [0, 255, 0]

        for r in regions:
            if 'center' in r and 'radius' in r:
                cv2.circle(vis_img, r['center'], r['radius'], (0, 255, 255), 2)
                cv2.circle(vis_img, r['center'], 3, (0, 0, 255), -1)

        total_pixels = len(all_boundary_pixels)
        cv2.putText(vis_img, f"边界连通深色像素: {total_pixels}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis_img, "未找到与边界相连的深色像素", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(output_path, vis_img)
    print(f"可视化结果已保存: {output_path}")

    return vis_img


def visualize_with_original(img, regions, output_path):
    """
    在原图上叠加显示与边界相连的深色像素
    """
    h, w = img.shape[:2]
    vis_img = img.copy()

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, r in enumerate(regions):
        color = colors[i % len(colors)]
        for pixel in r['pixels']:
            overlay[pixel[0], pixel[1]] = color

        boundary_sides = []
        if r['min_row'] == 0:
            boundary_sides.append('上')
        if r['max_row'] == h - 1:
            boundary_sides.append('下')
        if r['min_col'] == 0:
            boundary_sides.append('左')
        if r['max_col'] == w - 1:
            boundary_sides.append('右')

        label_text = f"区域{i+1}: {len(r['pixels'])}像素 ({', '.join(boundary_sides)}边界)"
        cv2.putText(vis_img, label_text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

    cv2.imwrite(output_path, vis_img)
    print(f"叠加可视化结果已保存: {output_path}")

    return vis_img


if __name__ == "__main__":
    test_dir = r"./test/test5"
    image_path = "micro_0008_XII0.jpg"
    json_path = image_path.replace(".jpg", ".json")
    output_dir = test_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("查找与边界相连的深色像素连通区域")
    print("=" * 60)

    img, boundary_regions = find_boundary_connected_dark_pixels(image_path, dark_threshold=200)

    print("\n" + "=" * 60)
    print("查找闭合圆环（完全封闭的深色区域）")
    print("=" * 60)

    closed_regions = find_closed_dark_regions(image_path, dark_threshold=200)

    if closed_regions:
        print(f"找到 {len(closed_regions)} 个闭合圆环:")
        h, w = img.shape[:2]
        for i, r in enumerate(closed_regions):
            circularity, _ = calculate_circularity(r, img.shape)
            print(f"  区域{i+1}: 像素数={len(r['pixels'])}, "
                  f"行范围=[{r['min_row']}, {r['max_row']}], "
                  f"列范围=[{r['min_col']}, {r['max_col']}], "
                  f"圆度={circularity:.4f}")
    else:
        print("未找到闭合圆环")

    if img is not None:
        vis_path = os.path.join(output_dir, "boundary_dark_pixels.png")
        visualize_boundary_dark_pixels(img, closed_regions, vis_path)

        overlay_path = os.path.join(output_dir, "boundary_dark_pixels_overlay.png")
        visualize_with_original(img, boundary_regions, overlay_path)

        all_vis_path = os.path.join(output_dir, "all_dark_regions.png")
        visualize_all_dark_regions(img, boundary_regions, closed_regions, all_vis_path)

        removed_path = os.path.join(output_dir, "removed_dark_pixels.png")
        remove_dark_regions(img, boundary_regions, closed_regions, removed_path)

        print("\n处理完成!")
