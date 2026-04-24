import json
import cv2
import os
import math
import numpy as np

def expand_poly_vertical(poly, expand_pixels=5):
    '''
    将文本框沿上下方向外扩指定像素
    
    Args:
        poly: 多边形坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        expand_pixels: 外扩像素数，上下各扩展这么多像素
    
    Returns:
        扩展后的多边形坐标
    '''
    poly = np.array(poly, dtype=np.float64)
    
    y_coords = poly[:, 1]
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    new_y_min = y_min - expand_pixels
    new_y_max = y_max + expand_pixels
    
    new_poly = poly.copy()
    
    for i in range(len(poly)):
        if poly[i][1] == y_min:
            new_poly[i][1] = new_y_min
        elif poly[i][1] == y_max:
            new_poly[i][1] = new_y_max
    
    return new_poly.tolist()

def count_dark_pixels_in_expanded_region(image, original_poly, expanded_poly, dark_threshold=128):
    '''
    统计外扩新增区域内的深色像素数量（外扩矩形内但不在原始矩形内的区域）
    
    Args:
        image: OpenCV图像对象 (BGR格式)
        original_poly: 原始多边形坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        expanded_poly: 外扩后多边形坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dark_threshold: 深色像素阈值，灰度值低于此值为深色 (默认128)
    
    Returns:
        dark_pixel_count: 新增区域内深色像素数量
        total_pixel_count: 新增区域总像素数量
        dark_ratio: 新增区域深色像素比例
    '''
    if image is None:
        return 0, 0, 0.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    original_poly = np.array(original_poly, dtype=np.int32)
    expanded_poly = np.array(expanded_poly, dtype=np.int32)
    
    mask_original = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask_original, [original_poly], 255)
    
    mask_expanded = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask_expanded, [expanded_poly], 255)
    
    mask_new_region = cv2.subtract(mask_expanded, mask_original)
    
    total_pixel_count = np.sum(mask_new_region > 0)
    
    dark_pixel_count = np.sum((gray < dark_threshold) & (mask_new_region > 0))
    
    dark_ratio = dark_pixel_count / total_pixel_count if total_pixel_count > 0 else 0.0
    
    return dark_pixel_count, total_pixel_count, dark_ratio

def draw_poly_comparison(json_path, output_dir, expand_pixels=5):
    '''
    可视化外扩前后的文本框对比
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_name = data['micro_image_name']
    image_path = os.path.join(os.path.dirname(json_path), image_name)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f'无法读取图片: {image_path}')
        return
    
    original_poly = data['micro_poly']
    
    expanded_poly = expand_poly_vertical(original_poly, expand_pixels)
    
    orig_points = [(int(x), int(y)) for x, y in original_poly]
    for i in range(4):
        cv2.line(img, orig_points[i], orig_points[(i+1)%4], (0, 255, 0), 2)
    
    exp_points = [(int(x), int(y)) for x, y in expanded_poly]
    for i in range(4):
        cv2.line(img, exp_points[i], exp_points[(i+1)%4], (0, 0, 255), 2)
    
    cv2.putText(img, 'Original (Green)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f'Expanded (Red, +{expand_pixels}px)', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'expanded_{image_name}')
    cv2.imwrite(output_path, img)
    
    print(f'处理完成: {output_path}')
    print(f'原始多边形: {original_poly}')
    print(f'外扩后多边形: {expanded_poly}')

if __name__ == '__main__':
    json_path = r'/test/test4/micro_0060_X.json'
    output_dir = r'/test/test4/output'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_name = data['micro_image_name']
    image_path = os.path.join(os.path.dirname(json_path), image_name)
    img = cv2.imread(image_path)
    
    original_poly = data['micro_poly']
    expanded_poly = expand_poly_vertical(original_poly, expand_pixels=5)
    
    dark_count, total_count, dark_ratio = count_dark_pixels_in_expanded_region(img, original_poly, expanded_poly, dark_threshold=128)
    print(f'外扩新增区域深色像素统计:')
    print(f'  新增区域深色像素数量: {dark_count}')
    print(f'  新增区域总像素数量: {total_count}')
    print(f'  新增区域深色像素比例: {dark_ratio:.4f} ({dark_ratio*100:.2f}%)')
    
    draw_poly_comparison(json_path, output_dir, expand_pixels=5)
