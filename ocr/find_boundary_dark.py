import cv2
import numpy as np
from collections import deque
import os

def find_closed_dark_regions(img_path, dark_threshold=125, min_circularity=0.7,path=None):
    """
    找出完全封闭的深色区域（闭合圆环）
    通过检查深色区域的边界是否完全被深色像素包围（即内部有空洞）
    只有圆度 >= min_circularity 的才被认为是闭合圆环
    img_path: 可以是图片路径(str)或图片数组(numpy.ndarray)
    """
    if isinstance(img_path, np.ndarray):
        img = img_path
    else:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    if isinstance(path, str):
        dark_mask_vis = (dark_mask * 255).astype(np.uint8)
        name, _ = os.path.splitext(os.path.basename(path))
        output_dir = os.path.join(os.path.dirname(path), 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join('output', f"{name}_dark_mask.jpg")
        cv2.imwrite(output_path, dark_mask_vis)

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
                # print(f"区域 {r['label']} 的圆度: {circularity:.4f}")
                if circularity >= min_circularity:
                    center, radius = calculate_circle_from_contour(largest_contour)
                    r['center'] = center
                    r['radius'] = radius
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


def find_boundary_connected_dark_pixels(img_path, dark_threshold=125, remove_color_adjacent=False, find_adjacent_color_regions=False):
    """
    找出与边界相连的深色像素连通区域
    使用BFS进行连通区域标记
    img_path: 可以是图片路径(str)或图片数组(numpy.ndarray)
    remove_color_adjacent: 如果为True，则排除与彩色像素相邻的深色区域
    find_adjacent_color_regions: 如果为True，在现有基础上额外找到与黑色像素相连的彩色像素区域
    """
    if isinstance(img_path, np.ndarray):
        img = img_path
    else:
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

    if remove_color_adjacent:
        filtered_regions = []
        white_threshold = 200
        for r in boundary_regions:
            touches_color = False
            for pixel in r['pixels']:
                row, col = pixel
                for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                    if 0 <= nr < h and 0 <= nc < w:
                        if dark_mask[nr, nc] == 0:
                            pixel_color = img[nr, nc]
                            if not (pixel_color[0] >= white_threshold and pixel_color[1] >= white_threshold and pixel_color[2] >= white_threshold):
                                touches_color = True
                                break
                if touches_color:
                    break
            if not touches_color:
                filtered_regions.append(r)
        boundary_regions = filtered_regions

    adjacent_color_regions = []
    if find_adjacent_color_regions:
        dark_pixels_set = set()
        for r in boundary_regions:
            for pixel in r['pixels']:
                dark_pixels_set.add(pixel)

        white_threshold = 200
        non_white_mask = ~(
            (img[:, :, 0] >= white_threshold) &
            (img[:, :, 1] >= white_threshold) &
            (img[:, :, 2] >= white_threshold)
        )
        visited = set(dark_pixels_set)
        queue = deque(dark_pixels_set)
        adjacent_pixels = set()

        while queue:
            row, col = queue.popleft()
            for nr, nc in [
                (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
                (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1),
            ]:
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in visited:
                    continue
                if not non_white_mask[nr, nc]:
                    continue

                visited.add((nr, nc))
                queue.append((nr, nc))
                if (nr, nc) not in dark_pixels_set:
                    adjacent_pixels.add((nr, nc))

        if adjacent_pixels:
            adjacent_color_regions.append({
                'label': label + 1,
                'pixels': list(adjacent_pixels),
                'min_col': min(p[1] for p in adjacent_pixels),
                'max_col': max(p[1] for p in adjacent_pixels),
                'min_row': min(p[0] for p in adjacent_pixels),
                'max_row': max(p[0] for p in adjacent_pixels),
            })

        boundary_regions = boundary_regions + adjacent_color_regions

    # print(f"图像尺寸: {w}x{h}")
    # print(f"深色阈值: {dark_threshold}")
    # print(f"总连通区域数: {len(regions)}")
    # print(f"与边界相连的深色区域数: {len(boundary_regions)}")
    # if find_adjacent_color_regions:
    #     print(f"与黑色像素相连的非白色区域数: {len(adjacent_color_regions)}")

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

        # print(f"  区域{i+1}: 像素数={len(r['pixels'])}, "
        #       f"行范围=[{r['min_row']}, {r['max_row']}], "
        #       f"列范围=[{r['min_col']}, {r['max_col']}], "
        #       f"接触边界: {', '.join(boundary_sides)}")

    return img, boundary_regions


def remove_dark_regions(img, boundary_regions, closed_regions, output_path=None):
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
    if output_path:
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

    for r in closed_regions:
        if 'center' in r and 'radius' in r:
            cv2.circle(vis_img, r['center'], r['radius'], (0, 255, 255), 2)
            cv2.circle(vis_img, r['center'], 3, (0, 0, 255), -1)

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


def find_color_and_adjacent_black_regions(image_path, dark_threshold=200, detect_green=False, detect_blue=False, detect_red=False, detect_yellow=False):
    """
    查找图中指定彩色像素及其相连的黑色像素区域
    image_path: 可以是图片路径(str)或图片数组(numpy.ndarray)
    dark_threshold: 黑色像素阈值
    detect_green/detect_blue/detect_red/detect_yellow: 是否检测对应颜色
    """
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    color_mask = np.zeros((h, w), dtype=np.uint8)

    if detect_green:
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        color_mask = cv2.bitwise_or(color_mask, green_mask)

    if detect_blue:
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        color_mask = cv2.bitwise_or(color_mask, blue_mask)

    if detect_red:
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        color_mask = cv2.bitwise_or(color_mask, red_mask)

    if detect_yellow:
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        color_mask = cv2.bitwise_or(color_mask, yellow_mask)

    labeled = np.zeros_like(dark_mask, dtype=np.int32)
    label = 0
    all_regions = []

    for row in range(h):
        for col in range(w):
            if (dark_mask[row, col] == 1 or color_mask[row, col] == 1) and labeled[row, col] == 0:
                label += 1
                region_pixels = []
                queue = deque([(row, col)])
                labeled[row, col] = label
                is_dark_region = dark_mask[row, col] == 1

                while queue:
                    r, c = queue.popleft()
                    region_pixels.append((r, c))

                    neighbors = [
                        (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
                        (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1),
                    ]

                    for nr, nc in neighbors:
                        if 0 <= nr < h and 0 <= nc < w:
                            if labeled[nr, nc] == 0:
                                if dark_mask[nr, nc] == 1 or color_mask[nr, nc] == 1:
                                    labeled[nr, nc] = label
                                    queue.append((nr, nc))
                                    if dark_mask[nr, nc] == 1:
                                        is_dark_region = True

                if is_dark_region and region_pixels:
                    all_regions.append({
                        'label': label,
                        'pixels': region_pixels,
                        'min_col': min(p[1] for p in region_pixels),
                        'max_col': max(p[1] for p in region_pixels),
                        'min_row': min(p[0] for p in region_pixels),
                        'max_row': max(p[0] for p in region_pixels),
                    })

    return all_regions


def find_drak_remove(image_path, dark_threshold=200, output_path=None, save_circle=True, remove_light_white=False, remove_color_adjacent=False, find_adjacent_color_regions=False, not_save_boundary=False,min_circularity=0.8, remove_color_and_adjacent_black=False, detect_green=False, detect_blue=False, detect_red=False, detect_yellow=False):
    """
    找出并移除深色像素（边界连通 + 闭合圆环）
    image_path: 可以是图片路径(str)或图片数组(numpy.ndarray)
    remove_color_adjacent: 如果为True，则排除与彩色像素相邻的深色区域
    find_adjacent_color_regions: 如果为True，额外找到与黑色像素相连的彩色像素区域
    remove_color_and_adjacent_black: 如果为True，查找图中指定彩色像素及其相连的黑色像素并去除
    detect_green/detect_blue/detect_red/detect_yellow: 指定要去除的彩色颜色
    """
    from ocr.detect_white_circles import find_all_white_regions, detect_circular_white_regions
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None, None
    boundary_regions = []
    if not_save_boundary == False:
        img, boundary_regions = find_boundary_connected_dark_pixels(
            img, dark_threshold=dark_threshold, remove_color_adjacent=remove_color_adjacent,
            find_adjacent_color_regions=find_adjacent_color_regions)
    closed_regions = []
    if not save_circle:
        closed_regions = find_closed_dark_regions(image_path, dark_threshold=dark_threshold)
        if remove_light_white:
            regions, white_mask = find_all_white_regions(img, white_threshold=200)
            _, closed_regions = detect_circular_white_regions(regions, img.shape, closed_circles=closed_regions,
                                                    min_circularity=min_circularity)
    if remove_color_and_adjacent_black:
        color_and_black_regions = find_color_and_adjacent_black_regions(
            image_path, dark_threshold=dark_threshold,
            detect_green=detect_green, detect_blue=detect_blue,
            detect_red=detect_red, detect_yellow=detect_yellow)
        boundary_regions = boundary_regions + color_and_black_regions

    result = remove_dark_regions(img, boundary_regions, closed_regions, output_path)
    return result

if __name__ == "__main__":
    test_dir = r"test/test5/"
    image_path = os.path.join(test_dir, "micro_0005_S15_cropped0.jpg")

    output_dir = test_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("查找与边界相连的深色像素连通区域")
    print("=" * 60)

    img, boundary_regions = find_boundary_connected_dark_pixels(image_path, dark_threshold=200,find_adjacent_color_regions=True)

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

    removed_path = os.path.join(output_dir, "removed_dark_pixels.png")
    remove_dark_regions(img, boundary_regions, closed_regions, removed_path)
    # if img is not None:
    #     vis_path = os.path.join(output_dir, "boundary_dark_pixels.png")
    #     visualize_boundary_dark_pixels(img, boundary_regions, vis_path)
    #
    #     overlay_path = os.path.join(output_dir, "boundary_dark_pixels_overlay.png")
    #     visualize_with_original(img, boundary_regions, overlay_path)
    #
    #     all_vis_path = os.path.join(output_dir, "all_dark_regions.png")
    #     visualize_all_dark_regions(img, boundary_regions, closed_regions, all_vis_path)
    #
    #     removed_path = os.path.join(output_dir, "removed_dark_pixels.png")
    #     remove_dark_regions(img, boundary_regions, closed_regions, removed_path)
    #
    #     print("\n处理完成!")
