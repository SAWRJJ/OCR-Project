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

def find_first_non_white_column_along_tilt(poly, gray_img, angle, debug_img=None, output_path=None,ex_p = 70):
    """
    利用文本框的横向倾斜角度，从图片最左侧开始沿着倾斜方向扫描
    找到第一列（有非白色像素的列），然后向右外扩像素
    
    参数:
        poly: 文本框四边形顶点
        gray_img: 灰度图像
        angle: 文本框倾斜角度
        debug_img: 用于可视化的BGR图像，如果为None则从gray_img创建
        output_path: 可视化图片保存路径
    """
    poly = np.array(poly, dtype=np.float64)

    top_left = poly[0]
    top_right = poly[1]

    tilt_angle = angle

    print(f"文本框倾斜角度: {math.degrees(tilt_angle):.2f} 度")
    tilt_angle = -tilt_angle
    poly_y_center = (np.min(poly[:, 1]) + np.max(poly[:, 1])) / 2
    poly_x_min = np.min(poly[:, 0])

    img_height = gray_img.shape[0]
    y_min = 15
    y_max = img_height - 15

    first_non_white_col = None
    non_white_pixels = []

    start_x = 0
    start_y = poly_y_center

    step_size = 1
    num_steps = max(gray_img.shape[1] * 2, 3000)

    vis_img = None
    if debug_img is not None:
        vis_img = debug_img.copy()
    else:
        if len(gray_img.shape) == 2:
            vis_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = gray_img.copy()

    scan_points = []
    found_col_x = None
    found_scan_line_start = None
    found_scan_line_end = None

    perp_angle = tilt_angle + math.pi / 2
    perp_dx = math.cos(perp_angle)
    perp_dy = math.sin(perp_angle)
    scan_line_length = 52

    for step in range(num_steps):
        x = start_x + step * math.cos(tilt_angle) * step_size
        y = start_y + step * math.sin(tilt_angle) * step_size

        check_x = int(x)

        if check_x < 0 or check_x >= gray_img.shape[1]:
            continue

        scan_line_start = (int(x - perp_dx * scan_line_length), int(y - perp_dy * scan_line_length))
        scan_line_end = (int(x + perp_dx * scan_line_length), int(y + perp_dy * scan_line_length))

        col_has_non_white = False

        line_points = []
        dx = scan_line_end[0] - scan_line_start[0]
        dy = scan_line_end[1] - scan_line_start[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            steps = 1
        for i in range(steps + 1):
            px = int(scan_line_start[0] + dx * i / steps)
            py = int(scan_line_start[1] + dy * i / steps)
            if 0 <= py < gray_img.shape[0] and 0 <= px < gray_img.shape[1]:
                line_points.append((px, py))
                if gray_img[py, px] < 128:
                    col_has_non_white = True

        if step % 10 == 0 and vis_img is not None:
            cv2.line(vis_img, scan_line_start, scan_line_end, (200, 200, 200), 1)
            cv2.circle(vis_img, (check_x, int(y)), 1, (0, 255, 255), -1)
            scan_points.append((check_x, int(y)))

        if first_non_white_col is None:
            if col_has_non_white:
                first_non_white_col = check_x
                found_col_x = check_x
                found_scan_line_start = scan_line_start
                found_scan_line_end = scan_line_end
                found_y = y
                for px, py in line_points:
                    if gray_img[py, px] < 128:
                        non_white_pixels.append((px, py))
                print(f"找到第一列有非白色像素: x={check_x}, 共{len(non_white_pixels)}个非白像素")
                break

        if x > poly_x_min + 50:
            break

    expand_line_start = None
    expand_line_end = None
    expand_x = None
    expand_y = None
    if first_non_white_col is not None:
        expand_x = check_x + ex_p * math.cos(tilt_angle)
        expand_y = y + ex_p * math.sin(tilt_angle)
        expand_line_start = (int(expand_x - perp_dx * scan_line_length), int(expand_y - perp_dy * scan_line_length))
        expand_line_end = (int(expand_x + perp_dx * scan_line_length), int(expand_y + perp_dy * scan_line_length))
        print(f"向右外扩像素: x={expand_x}, y={expand_y}")

    scan_end_point = (int(x), int(y)) if step > 0 else None

    if vis_img is not None and output_path is not None:
        cv2.polylines(vis_img, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
        
        if scan_points and len(scan_points) > 1:
            cv2.polylines(vis_img, [np.array(scan_points)], False, (255, 255, 0), 1)
        
        if found_scan_line_start is not None and found_scan_line_end is not None:
            cv2.line(vis_img, found_scan_line_start, found_scan_line_end, (0, 0, 255), 2)
        
        if expand_line_start is not None and expand_line_end is not None:
            cv2.line(vis_img, expand_line_start, expand_line_end, (255, 0, 0), 2)
        
        for px, py in non_white_pixels:
            cv2.circle(vis_img, (px, py), 1, (0, 255, 0), -1)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"可视化图片已保存到: {output_path}")

    return first_non_white_col, found_scan_line_start, found_scan_line_end, non_white_pixels, expand_x, expand_y, expand_line_start, expand_line_end


def shift_poly_along_angle(poly, angle, shift_distance=100, debug_img=None, output_path=None):
    """
    沿文本框倾斜角度向右平移多边形左侧边界，组合成新的多边形

    参数:
        poly: 文本框四边形顶点 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        angle: 文本框倾斜角度（弧度），正值表示右倾斜，负值表示左倾斜
        shift_distance: 平移距离（像素），默认100
        debug_img: 用于可视化的BGR图像，如果为None则创建
        output_path: 可视化图片保存路径

    返回:
        new_poly: 新的多边形顶点（平移后的左侧边 + 原始上下右边）
        shift_line_start: 平移线起点
        shift_line_end: 平移线终点
    """
    poly = np.array(poly, dtype=np.float64)

    tilt_angle = -angle
    print(f"平移角度: {math.degrees(tilt_angle):.2f} 度, 平移距离: {shift_distance} 像素")

    left_y_center = (poly[0, 1] + poly[3, 1]) / 2
    left_x = poly[0, 0]

    shift_dx = shift_distance * math.cos(tilt_angle)
    shift_dy = shift_distance * math.sin(tilt_angle)

    shifted_left_poly = poly.copy()
    shifted_left_poly[0, 0] += shift_dx
    shifted_left_poly[0, 1] += shift_dy
    shifted_left_poly[3, 0] += shift_dx
    shifted_left_poly[3, 1] += shift_dy

    new_poly = np.array([
        shifted_left_poly[0],  # 新左上
        poly[0],  # 原始左上 (变更为新右上)
        poly[3],  # 原始左下 (变更为新右下)
        shifted_left_poly[3]  # 新左下
    ], dtype=np.float64)

    vis_img = None
    if debug_img is not None:
        vis_img = debug_img.copy()
    else:
        vis_img = np.zeros((500, 500, 3), dtype=np.uint8)
        print("警告: 未提供debug_img，可视化可能不准确")

    cv2.polylines(vis_img, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(vis_img, [np.array(new_poly, dtype=np.int32)], True, (255, 0, 0), 2)

    shift_line_start = (int(left_x), int(left_y_center))
    shift_line_end = (int(left_x + shift_dx), int(left_y_center + shift_dy))
    cv2.arrowedLine(vis_img, shift_line_start, shift_line_end, (0, 255, 255), 2)

    cv2.putText(vis_img, f"Original (Green)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_img, f"New Poly (Blue)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(vis_img, f"Angle: {math.degrees(tilt_angle):.2f}deg, Dist: {shift_distance}px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"可视化图片已保存到: {output_path}")

    print(f"原始多边形: {poly.tolist()}")
    print(f"新多边形: {new_poly.tolist()}")
    print(f"平移向量: dx={shift_dx:.2f}, dy={shift_dy:.2f}")

    return new_poly, shift_line_start, shift_line_end,shifted_left_poly


def shift_poly_along_angle_step(poly, angle, step_size=5, debug_img=None, output_path=None, target_first_black_index=None, shift_after_black=5):
    """
    沿文本框倾斜角度方向,每5像素逐步平移左侧边直到右侧边位置

    参数:
        poly: 文本框四边形顶点 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        angle: 文本框倾斜角度（弧度），正值表示右倾斜，负值表示左倾斜
        step_size: 每步平移像素数，默认5
        debug_img: 用于可视化的BGR图像，如果为None则创建
        output_path: 可视化图片保存路径
        target_first_black_index: 当找黑检测线的个数等于此值时，用该检测线后再平移shift_after_black次的检测线和右侧边组成新poly
        shift_after_black: 在找到黑检测线后再平移的次数，默认5

    返回:
        all_polys: 所有平移过程中的多边形列表
        final_poly: 最终多边形（左侧边到达右侧边位置）
        shift_info: 每步的平移信息列表
        black_pixel_positions: 检测到黑色像素的位置列表
        first_black_lines: 首次出现黑色的检测线索引列表
        target_poly: 目标新poly（当target_first_black_index匹配时）
        target_crop_img: 根据target_poly截取的图像
    """
    poly = np.array(poly, dtype=np.float64)

    tilt_angle = -angle
    print(f"平移角度: {math.degrees(tilt_angle):.2f} 度, 步长: {step_size} 像素")

    left_top = poly[0]
    left_bottom = poly[3]
    right_top = poly[1]
    right_bottom = poly[2]

    left_center_y = (left_top[1] + left_bottom[1]) / 2
    right_center_y = (right_top[1] + right_bottom[1]) / 2

    dx_per_step = step_size * math.cos(tilt_angle)
    dy_per_step = step_size * math.sin(tilt_angle)

    current_left_top = left_top.copy()
    current_left_bottom = left_bottom.copy()

    right_top_x = right_top[0]
    right_bottom_x = right_bottom[0]

    all_polys = []
    shift_info = []
    black_pixel_positions = []

    step_count = 0
    while True:
        if current_left_top[0] >= right_top_x and current_left_bottom[0] >= right_bottom_x:
            break

        shifted_poly = np.array([
            current_left_top,
            current_left_bottom,
            right_bottom,
            right_top
        ], dtype=np.float64)

        all_polys.append(shifted_poly.copy())
        shift_info.append({
            'step': step_count,
            'left_top': tuple(current_left_top),
            'left_bottom': tuple(current_left_bottom)
        })

        current_left_top[0] += dx_per_step
        current_left_top[1] += dy_per_step
        current_left_bottom[0] += dx_per_step
        current_left_bottom[1] += dy_per_step
        step_count += 1

        if step_count > 1000:
            print("警告: 达到最大迭代次数1000，可能存在无限循环问题")
            break

    final_poly = np.array([
        current_left_top,
        current_left_bottom,
        right_bottom,
        right_top
    ], dtype=np.float64)

    vis_img = None
    if debug_img is not None:
        vis_img = debug_img.copy()
    else:
        h, w = 500, 500
        if debug_img is None:
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
            print("警告: 未提供debug_img，可视化可能不准确")

    cv2.polylines(vis_img, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(vis_img, [np.array(final_poly, dtype=np.int32)], True, (255, 0, 0), 2)

    original_left_top = tuple(map(int, left_top))
    original_left_bottom = tuple(map(int, left_bottom))

    first_black_lines = []
    prev_had_black = False

    for i, info in enumerate(shift_info):
        left_edge_start = tuple(map(int, info['left_top']))
        left_edge_end = tuple(map(int, info['left_bottom']))

        has_black = False

        num_steps = max(1, int(np.sqrt((left_edge_end[0] - left_edge_start[0])**2 + (left_edge_end[1] - left_edge_start[1])**2)))

        for j in range(num_steps):
            t = j / num_steps
            x = int(left_edge_start[0] + t * (left_edge_end[0] - left_edge_start[0]))
            y = int(left_edge_start[1] + t * (left_edge_end[1] - left_edge_start[1]))

            if 0 <= y < debug_img.shape[0] and 0 <= x < debug_img.shape[1]:
                pixel_val = debug_img[y, x]
                if np.mean(pixel_val) < 100:
                    has_black = True
                    black_pixel_positions.append((x, y))

        if has_black and not prev_had_black:
            color = (255, 0, 0)
            first_black_lines.append(i)
        elif has_black:
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)

        prev_had_black = has_black
        cv2.line(vis_img, left_edge_start, left_edge_end, color, 1)

    cv2.putText(vis_img, f"Original (Green)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_img, f"First Black (Blue)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(vis_img, f"Steps: {step_count}, Size: {step_size}px, First Black: {len(first_black_lines)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    for bx, by in black_pixel_positions:
        cv2.circle(vis_img, (bx, by), 1, (0, 0, 255), -1)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"可视化图片已保存到: {output_path}")

    print(f"原始多边形: {poly.tolist()}")
    print(f"最终多边形: {final_poly.tolist()}")
    print(f"总共平移步数: {step_count}")
    print(f"检测到黑色像素数量: {len(black_pixel_positions)}")
    print(f"首次出现黑色的检测线索引: {first_black_lines}")

    target_poly = None
    target_crop_img = None

    if target_first_black_index is not None and len(first_black_lines) >= target_first_black_index:
        target_line_idx = first_black_lines[target_first_black_index - 1]
        shifted_line_idx = target_line_idx + shift_after_black

        if shifted_line_idx >= len(shift_info):
            shifted_line_idx = len(shift_info) - 1
            print(f"警告: shift_after_black={shift_after_black}超出范围，使用最后一条检测线(索引={shifted_line_idx})")

        shifted_info = shift_info[shifted_line_idx]

        left_top = shifted_info['left_top']
        left_bottom = shifted_info['left_bottom']

        left_top_x = left_top[0]
        left_bottom_x = left_bottom[0]

        left_edge_near_right_x = max(left_top_x, left_bottom_x)

        new_left_top = (left_edge_near_right_x, left_top[1])
        new_left_bottom = (left_edge_near_right_x, left_bottom[1])

        target_poly = np.array([
            new_left_top,
            new_left_bottom,
            right_bottom,
            right_top
        ], dtype=np.int32)
        print(f"使用第 {target_first_black_index} 个找黑检测线(索引={target_line_idx})后平移{shift_after_black}次(索引={shifted_line_idx})创建新poly")
        print(f"  原始检测线: left_top={left_top}, left_bottom={left_bottom}")
        print(f"  左侧靠近右侧边的x坐标: {left_edge_near_right_x}")
        print(f"  新poly: {target_poly.tolist()}")

        x_min = max(0, int(np.min(target_poly[:, 0])) - 5)
        x_max = min(debug_img.shape[1], int(np.max(target_poly[:, 0])) + 5)
        y_min = max(0, int(np.min(target_poly[:, 1])) - 5)
        y_max = min(debug_img.shape[0], int(np.max(target_poly[:, 1])) + 5)

        target_crop_img = debug_img[y_min:y_max, x_min:x_max]
        cv2.polylines(vis_img, [target_poly], True, (0, 255, 255), 2)

    return all_polys, final_poly, shift_info, black_pixel_positions, first_black_lines, target_poly, target_crop_img
def shift_poly_along_angle1(poly, img, shift_point=None, target_char=None, output_path=None):
    """
    根据target_char选择左侧或右侧边，以nearest_center为中心，
    沿边的倾斜角度上下外扩55px得到110px线段，
    再将线段平移回被平移边位置，组合成新的多边形

    参数:
        poly: 文本框四边形顶点 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        img: 原始图像
        shift_point: 平移目标点/圆心 (nearest_center)
        target_char: 目标字符，'X'选右侧边，'S'选左侧边
        output_path: 可视化图片保存路径

    返回:
        new_poly: 新的多边形顶点
        shift_line_start: 平移线起点
        shift_line_end: 平移线终点
    """
    poly = np.array(poly, dtype=np.float64)
    h, w = img.shape[:2]

    if target_char and 'X' in target_char.upper():
        edge_start = poly[1].tolist()
        edge_end = poly[2].tolist()
        edge_name = "右侧"
    else:
        edge_start = poly[0].tolist()
        edge_end = poly[3].tolist()
        edge_name = "左侧"

    print(f"选择{edge_name}边: {edge_start} -> {edge_end}")

    dx = float(edge_end[0]) - float(edge_start[0])
    dy = float(edge_end[1]) - float(edge_start[1])
    edge_length = np.sqrt(dx**2 + dy**2)
    if edge_length == 0:
        edge_length = 1

    tilt_angle = np.arctan2(dy, dx)
    print(f"边倾斜角度: {np.degrees(tilt_angle):.2f} 度")

    cos_a = np.cos(tilt_angle)
    sin_a = np.sin(tilt_angle)

    extend_length = 55
    if shift_point is not None:
        center_x, center_y = float(shift_point[0]), float(shift_point[1])

        line_start = (center_x - cos_a * extend_length, center_y - sin_a * extend_length)
        line_end = (center_x + cos_a * extend_length, center_y + sin_a * extend_length)

        print(f"以nearest_center为中点的110px线段: {line_start} -> {line_end}")

        shift_dx = float(edge_start[0]) - line_start[0]
        shift_dy = float(edge_start[1]) - line_start[1]

        shifted_line_start = (line_start[0] + shift_dx, line_start[1] + shift_dy)
        shifted_line_end = (line_end[0] + shift_dx, line_end[1] + shift_dy)
    else:
        shift_dx = 0
        shift_dy = 0
        line_start = (float(edge_start[0]), float(edge_start[1]))
        line_end = (float(edge_end[0]), float(edge_end[1]))
        shifted_line_start = line_start
        shifted_line_end = line_end

    print(f"平移回被平移边后: {shifted_line_start} -> {shifted_line_end}")

    new_poly = np.array([
        [shifted_line_start[0], shifted_line_start[1]],
        [shifted_line_end[0], shifted_line_end[1]],
        [line_end[0], line_end[1]],
        [line_start[0], line_start[1]]
    ], dtype=np.int32)

    vis_img = img.copy()

    cv2.line(vis_img, (int(line_start[0]), int(line_start[1])), (int(line_end[0]), int(line_end[1])), (255, 255, 0), 2)
    cv2.line(vis_img, (int(shifted_line_start[0]), int(shifted_line_start[1])), (int(shifted_line_end[0]), int(shifted_line_end[1])), (255, 0, 0), 2)

    if shift_point is not None:
        cv2.circle(vis_img, (int(center_x), int(center_y)), 5, (0, 255, 255), -1)

    cv2.putText(vis_img, f"110px Line (Yellow)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(vis_img, f"Shifted Line (Blue)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(vis_img, f"Center Point (Cyan)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"可视化图片已保存到: {output_path}")

    print(f"原始边起点: {edge_start}, 终点: {edge_end}")
    print(f"新多边形: {new_poly.tolist()}")

    return new_poly, (int(shifted_line_start[0]), int(shifted_line_start[1])), (int(shifted_line_end[0]), int(shifted_line_end[1]))


def calculate_horizontal_tilt_angle(poly):
    """
    计算文本框的水平倾斜角度
    poly: 四边形四个点的坐标列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
          顺序为：左上、右上、右下、左下
    返回：倾斜角度（度），正值表示右倾斜，负值表示左倾斜
    """
    poly = np.array(poly)
    top_left = poly[0]
    top_right = poly[1]
    bottom_right = poly[2]
    bottom_left = poly[3]

    top_angle = math.degrees(math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
    bottom_angle = math.degrees(math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0]))

    avg_angle = (top_angle + bottom_angle) / 2
    return round(avg_angle, 2)


def expand_poly(poly, expand_x=10, expand_y=5, angle=0):
    """
    扩展多边形
    poly: 四边形四个点的坐标列表 [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
          顺序为：左上、右上、右下、左下
    expand_x: 水平方向右侧扩展像素数（左侧不变）
    expand_y: 垂直方向上下扩展像素数
    angle: 倾斜角度（度），正值右倾斜，负值左倾斜
    返回：扩展后的多边形
    """
    poly = np.array(poly, dtype=np.float32)
    angle_rad = math.radians(angle)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rotate_point(point, center, cos_a, sin_a):
        x, y = point[0] - center[0], point[1] - center[1]
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        return [new_x + center[0], new_y + center[1]]

    center_x = (poly[0][0] + poly[2][0]) / 2
    center_y = (poly[0][1] + poly[2][1]) / 2
    center = [center_x, center_y]

    rotated_poly = []
    for p in poly:
        rotated_poly.append(rotate_point(p, center, cos_a, -sin_a))

    rotated_poly = np.array(rotated_poly, dtype=np.float32)

    expanded = rotated_poly.copy()
    expanded[0][0] = rotated_poly[0][0]
    expanded[0][1] -= expand_y
    expanded[1][0] = rotated_poly[1][0] + expand_x
    expanded[1][1] -= expand_y
    expanded[2][0] = rotated_poly[2][0] + expand_x
    expanded[2][1] = rotated_poly[2][1] + expand_y
    expanded[3][0] = rotated_poly[3][0]
    expanded[3][1] = rotated_poly[3][1] + expand_y

    final_poly = []
    for p in expanded:
        final_poly.append(rotate_point(p, center, cos_a, sin_a))

    return [[int(p[0]), int(p[1])] for p in final_poly]

def count_vertical_strokes(image, debug=False,json_path=None):
    """
    输入:
        image: BGR / 灰度图 (np.ndarray)
    输出:
        竖线数量（int）
    """

    # 1. 灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. 二值化（建议自适应更稳）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 垂直投影（按列求和）
    projection = np.sum(binary // 255, axis=0)

    # 4. 平滑（去噪）
    kernel_size = 5
    projection_smooth = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')

    # 5. 峰值检测
    threshold = np.max(projection_smooth) * 0.7
    peaks = projection_smooth > threshold

    # 6. 统计连续峰段数量
    count = 0
    in_peak = False
    for val in peaks:
        if val and not in_peak:
            count += 1
            in_peak = True
        elif not val:
            in_peak = False

    # if debug:
    #     import matplotlib.pyplot as plt
    #     plt.plot(projection_smooth)
    #     plt.title(f"Peaks: {count}")
    #     plt.show()

    return count


if __name__ == '__main__':
    json_path = r'd:\work\ocr+Transformer\test4\micro_0060_X.json'
    output_dir = r'd:\work\ocr+Transformer\test4\output'
    
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
