import cv2
import numpy as np
from collections import deque

def find_all_dark_regions(img, dark_threshold=125):
    """
    找到所有深色像素连通区域
    """
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

    return regions, dark_mask


def fit_circle_least_squares(pixels):
    """
    使用最小二乘法拟合圆
    返回: (center_x, center_y), radius
    """
    if len(pixels) < 3:
        return None, None

    points = np.array(pixels, dtype=np.float64)

    x = points[:, 1]
    y = points[:, 0]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x = x - x_mean
    y = y - y_mean

    d2 = x**2 + y**2
    d4 = d2**2

    denominator = np.sum(d2)
    if denominator == 0:
        return None, None

    numerator_x = np.sum(x * d4)
    numerator_y = np.sum(y * d4)

    xc = numerator_x / (2 * denominator)
    yc = numerator_y / (2 * denominator)

    u = x - xc
    v = y - yc

    r_est = np.sqrt(np.mean(u**2 + v**2))

    center_x = xc + x_mean
    center_y = yc + y_mean

    return (float(center_x), float(center_y)), float(r_est)


def fit_circle_min_enclosing(pixels):
    """
    使用最小外接圆方法拟合圆
    返回: (center_x, center_y), radius
    """
    if len(pixels) < 3:
        return None, None

    points = np.array([[p[1], p[0]] for p in pixels], dtype=np.int32)

    contours = [points]
    hull = cv2.convexHull(points)
    (x, y), radius = cv2.minEnclosingCircle(hull)

    return (float(x), float(y)), float(radius)


def fit_circle_contour_method(region, img_shape):
    """
    使用轮廓方法拟合圆
    返回: (center_x, center_y), radius, circularity
    """
    h, w = img_shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    for pixel in region['pixels']:
        region_mask[pixel[0], pixel[1]] = 255

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)

    if perimeter == 0:
        return None, None, 0.0

    circularity = 4 * np.pi * area / (perimeter ** 2)

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    return (float(x), float(y)), float(radius), float(circularity)


def visualize_regions_with_circles(img, regions, output_path, min_circularity=0.75):
    """
    可视化所有深色区域及其拟合圆（只标记圆度>=min_circularity的区域）
    """
    h, w = img.shape[:2]
    vis_img = img.copy()

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    results = []

    for i, region in enumerate(regions):
        center, radius, circularity = fit_circle_contour_method(region, img.shape)
        if center is not None and circularity >= min_circularity:
            color = (0, 255, 0)

            for pixel in region['pixels']:
                overlay[pixel[0], pixel[1]] = color

            cv2.circle(vis_img, (int(center[0]), int(center[1])), int(radius), color, 2)
            cv2.circle(vis_img, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

            results.append({
                'region_id': i + 1,
                'pixel_count': len(region['pixels']),
                'center': center,
                'radius': radius,
                'circularity': circularity,
                'bbox': (region['min_row'], region['min_col'], region['max_row'], region['max_col'])
            })

    for i, result in enumerate(results):
        text = f"区域{result['region_id']}: 圆心=({result['center'][0]:.1f}, {result['center'][1]:.1f}), " \
               f"半径={result['radius']:.1f}, 圆度={result['circularity']:.3f}"
        cv2.putText(vis_img, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, vis_img)
    print(f"可视化结果已保存: {output_path}")

    return results


def convert_high_circularity_region_to_white(img, regions, min_circularity=0.75):
    """
    将圆度>=min_circularity的区域的像素转为白色
    """
    result = img.copy()

    for region in regions:
        center, radius, circularity = fit_circle_contour_method(region, img.shape)
        if center is not None and circularity >= min_circularity:
            for pixel in region['pixels']:
                result[pixel[0], pixel[1]] = [255, 255, 255]

    return result


def process_image_high_circularity_to_white(img, dark_threshold=125, min_circularity=0.75, binary_output_path=None):
    """
    完整处理流程：
    1. 找到所有深色像素连通区域
    2. 对每个区域拟合圆并计算圆度
    3. 将圆度>=min_circularity的区域转为白色
    4. 对结果进行二值化
    5. 可选保存二值化结果

    参数:
        img: 输入图像 (numpy.ndarray)
        dark_threshold: 深色阈值，默认125
        min_circularity: 最小圆度阈值，默认0.75
        binary_output_path: 二值化结果保存路径，如果为None则不保存

    返回:
        tuple: (去除了高圆度区域的图像, 二值化图像, 处理结果列表)
    """
    regions, _ = find_all_dark_regions(img, dark_threshold=dark_threshold)

    result_img = img.copy()
    result_list = []

    for region in regions:
        center, radius, circularity = fit_circle_contour_method(region, img.shape)
        if center is not None and circularity >= min_circularity:
            for pixel in region['pixels']:
                result_img[pixel[0], pixel[1]] = [255, 255, 255]
            result_list.append({
                'pixel_count': len(region['pixels']),
                'center': center,
                'radius': radius,
                'circularity': circularity
            })

    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)

    if binary_output_path is not None:
        cv2.imwrite(binary_output_path, binary)

    return result_img, binary, result_list


def main():
    image_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test9/micro_0093_XL_I_HO_00_cropped.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    print("\n使用组合函数处理图像...")
    result_img, binary, results = process_image_high_circularity_to_white(
        img,
        dark_threshold=200,
        min_circularity=0.75,
        binary_output_path="/Users/saw/WorkSpace/work/OCR-Project/test/test9/black_region_binary.jpg"
    )

    output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test9/black_region_to_white.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"已保存: {output_path}")

    print(f"\n处理了 {len(results)} 个高圆度区域")


if __name__ == "__main__":
    main()