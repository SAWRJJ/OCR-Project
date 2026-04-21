import cv2
import numpy as np
from collections import deque


def find_left_to_right_dark_connections(image_path, output_path, dark_threshold=125):
    """
    找到从左侧边到右侧边的深色像素连接路径
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dark_mask = (gray < dark_threshold).astype(np.uint8)

    visited = np.zeros_like(dark_mask, dtype=np.uint8)

    connections = []

    def bfs(start_row):
        """BFS找到从该行左侧到右侧的路径"""
        queue = deque()
        found_right = False
        path = []

        for col in range(w):
            if dark_mask[start_row, col] == 1:
                queue.append((start_row, col, [(start_row, col)]))
                break

        if len(queue) == 0:
            return None

        visited_local = np.zeros_like(dark_mask, dtype=np.uint8)

        while queue:
            row, col, path_so_far = queue.popleft()

            if row < 0 or row >= h or col < 0 or col >= w:
                continue

            if visited_local[row, col] == 1:
                continue
            visited_local[row, col] = 1

            if col >= w - 1:
                return path_so_far

            neighbors = [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
                (row - 1, col - 1),
                (row - 1, col + 1),
                (row + 1, col - 1),
                (row + 1, col + 1),
            ]

            for nr, nc in neighbors:
                if 0 <= nr < h and 0 <= nc < w:
                    if dark_mask[nr, nc] == 1 and visited_local[nr, nc] == 0:
                        queue.append((nr, nc, path_so_far + [(nr, nc)]))

        return None

    print(f"图像尺寸: {w}x{h}")
    print(f"深色像素阈值: < {dark_threshold}")
    print(f"深色像素总数: {np.sum(dark_mask > 0)}")

    for start_row in range(h):
        if dark_mask[start_row, 0] == 1 and visited[start_row, 0] == 0:
            path = bfs(start_row)
            if path:
                connections.append(path)
                for r, c in path:
                    visited[r, c] = 1

    print(f"找到 {len(connections)} 条从左到右的连通路径")

    vis_img = img.copy()

    for row in range(h):
        for col in range(w):
            if dark_mask[row, col] == 1:
                vis_img[row, col] = [50, 50, 50]

    np.random.seed(42)
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    for i, path in enumerate(connections):
        color = colors[i % len(colors)]
        for j, (r, c) in enumerate(path):
            if j % 3 == 0:
                cv2.circle(vis_img, (c, r), 2, color, -1)

    cv2.putText(vis_img, f"左到右连通路径: {len(connections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis_img, f"深色阈值: < {dark_threshold}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, vis_img)
    print(f"可视化结果已保存: {output_path}")

    return connections, vis_img


def find_horizontal_dark_bridges(image_path, output_path, dark_threshold=125):
    """
    找到所有横向跨越图像的深色连通区域
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

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

    left_to_right_regions = [r for r in regions if r['min_col'] == 0 and r['max_col'] == w - 1]

    print(f"找到 {len(regions)} 个连通区域")
    print(f"其中从左到右贯穿的: {len(left_to_right_regions)} 个")

    for i, region in enumerate(left_to_right_regions):
        print(f"  区域 {i+1}: 像素数={len(region['pixels'])}, "
              f"行范围=[{region['min_row']}, {region['max_row']}], "
              f"列范围=[{region['min_col']}, {region['max_col']}]")

    vis_img = img.copy()

    for row in range(h):
        for col in range(w):
            if dark_mask[row, col] == 1:
                vis_img[row, col] = [60, 60, 60]

    np.random.seed(42)
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
    ]

    for i, region in enumerate(left_to_right_regions):
        color = colors[i % len(colors)]
        for r, c in region['pixels']:
            vis_img[r, c] = color

    cv2.putText(vis_img, f"左到右连通区域: {len(left_to_right_regions)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(output_path, vis_img)
    print(f"可视化结果已保存: {output_path}")

    return left_to_right_regions, vis_img


if __name__ == "__main__":
    image_path = r"d:\work\ocr+Transformer\little_light\micro_0045_XI.jpg"
    output_path = r"d:\work\ocr+Transformer\little_light\micro_0045_XI_left_right_conn.png"

    print("=" * 50)
    print("方法1: BFS寻找路径")
    print("=" * 50)
    connections, _ = find_left_to_right_dark_connections(image_path, output_path, dark_threshold=125)

    print("\n" + "=" * 50)
    print("方法2: 连通区域分析")
    print("=" * 50)
    regions, _ = find_horizontal_dark_bridges(image_path, output_path, dark_threshold=125)
