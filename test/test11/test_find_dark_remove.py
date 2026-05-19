import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/saw/WorkSpace/work/OCR-Project')
from ocr.find_boundary_dark import find_boundary_connected_dark_pixels, find_closed_dark_regions, remove_dark_regions

img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test11/micro_0108_FSPN.jpg"

img = cv2.imread(img_path)
if img is None:
    print(f"无法读取图片: {img_path}")
    exit(1)

print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

dark_threshold = 85
print("\n=== 不排除与彩色区域相邻的深色区域 ===")
_, boundary_regions = find_boundary_connected_dark_pixels(img, dark_threshold=dark_threshold, remove_color_adjacent=True)
closed_regions = find_closed_dark_regions(img, dark_threshold=dark_threshold,path = r"/Users/saw/WorkSpace/work/OCR-Project/test/test11/micro_0108_FSPN.jpg")

print(f"\n边界连通区域数: {len(boundary_regions)}")
print(f"闭合圆环区域数: {len(closed_regions)}")

h, w = img.shape[:2]

for i, region in enumerate(closed_regions):
    pixels = region['pixels']
    min_dist = float('inf')

    for pixel in pixels:
        row, col = pixel
        dist_to_top = row
        dist_to_bottom = h - 1 - row
        dist_to_left = col
        dist_to_right = w - 1 - col
        min_pixel_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)
        if min_pixel_dist < min_dist:
            min_dist = min_pixel_dist

    print(f"  闭合区域{i+1}: {len(pixels)}像素, 边界最近距离: {min_dist}像素, 区域范围: row[{region['min_row']}, {region['max_row']}], col[{region['min_col']}, {region['max_col']}]")

output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test11/micro_0029_X6_processed.jpg"
result_img = remove_dark_regions(img, boundary_regions, closed_regions, output_path)
print(f"\n处理后的图像已保存至: {output_path}")

# print("\n=== 排除与彩色区域相邻的深色区域 ===")
# _, boundary_regions_filtered = find_boundary_connected_dark_pixels(img, dark_threshold=dark_threshold, remove_color_adjacent=True)
# print(f"\n过滤后边界连通区域数: {len(boundary_regions_filtered)}")