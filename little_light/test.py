import time
import cv2
import numpy as np
import json
import os

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def draw_target_char_edge(img, json_path, target_char, direction='left', color_line_info=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_index = None
    for i, text in enumerate(data.get('rec_texts', [])):
        if text == target_char:
            target_index = i
            break
    
    if target_index is None:
        print(f"未找到'{target_char}'文本")
        return img
    
    dt_polys = data.get('dt_polys', [])
    if target_index >= len(dt_polys):
        print("索引超出范围")
        return img
    
    poly = dt_polys[target_index]
    print(f"{target_char}对应的文本框轮廓: {poly}")
    
    cv2.polylines(img, [np.array(poly)], isClosed=True, color=(147, 20, 255), thickness=2)
    
    use_adjusted_edge = False
    adjusted_edge = None
    
    x_coords = [point[0] for point in poly]
    textbox_length = max(x_coords) - min(x_coords)
    print(f"文本框长度: {textbox_length} 像素")
    
    if textbox_length > 300:
        y_coords = [point[1] for point in poly]
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        if direction == 'right':
            rightmost_x = max(x_coords)
            yellow_box_left = rightmost_x - 40
            yellow_contour = [(yellow_box_left, min_y), (rightmost_x, min_y), (rightmost_x, max_y), (yellow_box_left, max_y)]
            cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
            print(f"已绘制最右侧40长度的黄色轮廓: {yellow_contour}")
            
            x1 = int(yellow_box_left)
            y1 = int(min_y)
            x2 = int(rightmost_x)
            y2 = int(max_y)
        elif direction == 'left':
            leftmost_x = min(x_coords)
            yellow_box_right = leftmost_x + 40
            yellow_contour = [(leftmost_x, min_y), (yellow_box_right, min_y), (yellow_box_right, max_y), (leftmost_x, max_y)]
            cv2.polylines(img, [np.array(yellow_contour)], isClosed=True, color=(0, 255, 255), thickness=2)
            print(f"已绘制最左侧40长度的黄色轮廓: {yellow_contour}")
            
            x1 = int(leftmost_x)
            y1 = int(min_y)
            x2 = int(yellow_box_right)
            y2 = int(max_y)
        
        height, width = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        if x2 > x1 and y2 > y1:
            cropped_img = img[y1:y2, x1:x2]
            
            if not os.path.exists('output'):
                os.makedirs('output')
            
            filename = os.path.basename(json_path).replace('_res.json', '')
            crop_output_path = os.path.join('output', f'{filename}_yellow_crop.png')
            cv2.imwrite(crop_output_path, cropped_img)
            print(f"已保存裁剪后的黄色轮廓区域: {crop_output_path}")
            
            crop_ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)
            
            crop_result = crop_ocr.predict([crop_output_path])
            
            ocr_success = False
            print("黄色轮廓区域的OCR识别结果:")
            for res in crop_result:
                if isinstance(res, dict) and 'rec_texts' in res and len(res['rec_texts']) > 0:
                    ocr_success = True
                    for line in res["rec_texts"]:
                        print(f"识别文本: {line}")
            
            for res in crop_result:
                if hasattr(res, 'save_to_img'):
                    res.save_to_img("output")
                    res.save_to_json("output")
            
            if ocr_success:
                print("OCR识别成功，将调整边缘")
                if direction == 'right':
                    adjusted_edge = [(yellow_box_left, min_y), (yellow_box_left, max_y)]
                elif direction == 'left':
                    adjusted_edge = [(yellow_box_right, min_y), (yellow_box_right, max_y)]
                use_adjusted_edge = True
                print(f"已设置调整后的边缘: {adjusted_edge}")
    
    poly_sorted = sorted(poly, key=lambda point: point[0])
    if direction == 'right':
        edge_points = poly_sorted[-2:]
    elif direction == 'left':
        edge_points = poly_sorted[:2]
    edge_points.sort(key=lambda point: point[1])
    
    if use_adjusted_edge and adjusted_edge:
        edge_points = adjusted_edge
        print(f"已使用调整后的边缘: {edge_points}")
    
    edge = tuple(map(tuple, edge_points))
    print(f"{direction}侧边坐标: {edge}")
    
    extended_start = edge[0]
    extended_end = edge[1]
    
    adjusted_color_line = None
    if color_line_info:
        color_start, color_end = color_line_info
        
        color_dx = abs(color_end[0] - color_start[0])
        color_dy = abs(color_end[1] - color_start[1])
        
        if color_dy < color_dx * 0.5:
            if direction == 'right':
                pos_x = min(color_start[0], color_end[0])
            elif direction == 'left':
                pos_x = max(color_start[0], color_end[0])
            adjusted_color_line = [(pos_x, edge[0][1]), (pos_x, edge[1][1])]
            print(f"颜色线水平排列，已调整为垂直线条: {adjusted_color_line}")
            color_start, color_end = adjusted_color_line
        
        color_dx = color_end[0] - color_start[0]
        color_dy = color_end[1] - color_start[1]
        color_length = ((color_dx ** 2) + (color_dy ** 2)) ** 0.5
        
        edge_dx = edge[1][0] - edge[0][0]
        edge_dy = edge[1][1] - edge[0][1]
        edge_length = ((edge_dx ** 2) + (edge_dy ** 2)) ** 0.5
        
        if edge_length > 0 and color_length > 0:
            unit_dx = edge_dx / edge_length
            unit_dy = edge_dy / edge_length
            scale_factor = color_length / edge_length
            extended_start = (int(edge[0][0] - unit_dx * edge_length * (scale_factor - 1) / 2), 
                              int(edge[0][1] - unit_dy * edge_length * (scale_factor - 1) / 2))
            extended_end = (int(edge[1][0] + unit_dx * edge_length * (scale_factor - 1) / 2), 
                            int(edge[1][1] + unit_dy * edge_length * (scale_factor - 1) / 2))
    
    cv2.line(img, extended_start, extended_end, (0, 0, 255), 2)
    
    if color_line_info:
        if adjusted_color_line:
            color_start, color_end = adjusted_color_line
        else:
            color_start, color_end = color_line_info
        
        polygon = [color_start, color_end, extended_end, extended_start]
        cv2.polylines(img, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=1)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon)], 255)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        black_pixels = cv2.countNonZero(cv2.bitwise_and(mask, black_mask))
        print(f"多边形内的黑色像素数量: {black_pixels}")
        
        data['black_pixels_in_polygon'] = int(black_pixels)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已将黑色像素数量更新到: {json_path}")
    
    return img

def detect_colors(image_path, target_char, direction):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([70, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    
    total_pixels = img.shape[0] * img.shape[1]
    
    print(f"图像: {image_path}")
    print(f"红色像素: {red_pixels} ({red_pixels/total_pixels*100:.2f}%)")
    print(f"黄色像素: {yellow_pixels} ({yellow_pixels/total_pixels*100:.2f}%)")
    print(f"绿色像素: {green_pixels} ({green_pixels/total_pixels*100:.2f}%)")
    
    def calculate_center(mask):
        if cv2.countNonZero(mask) == 0:
            return None
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    def calculate_extreme(mask, extreme_type='left'):
        if cv2.countNonZero(mask) == 0:
            return None
        coords = np.column_stack(np.where(mask > 0))
        if extreme_type == 'left':
            coords_sorted = coords[coords[:,1].argsort()]
            extreme = coords_sorted[0]
        else:  # right
            coords_sorted = coords[coords[:,1].argsort()[::-1]]
            extreme = coords_sorted[0]
        return (extreme[1], extreme[0])
    
    red_center = calculate_center(mask_red)
    yellow_center = calculate_center(mask_yellow)
    green_center = calculate_center(mask_green)
    
    if direction == 'right':
        extreme_type = 'left'
    else:
        extreme_type = 'right'
    
    red_extreme = calculate_extreme(mask_red, extreme_type)
    yellow_extreme = calculate_extreme(mask_yellow, extreme_type)
    green_extreme = calculate_extreme(mask_green, extreme_type)
    
    if red_center:
        print(f"红色像素中心坐标: {red_center}")
    if yellow_center:
        print(f"黄色像素中心坐标: {yellow_center}")
    if green_center:
        print(f"绿色像素中心坐标: {green_center}")
    
    if red_extreme:
        print(f"红色像素最{'左' if extreme_type == 'left' else '右'}端坐标: {red_extreme}")
    if yellow_extreme:
        print(f"黄色像素最{'左' if extreme_type == 'left' else '右'}端坐标: {yellow_extreme}")
    if green_extreme:
        print(f"绿色像素最{'左' if extreme_type == 'left' else '右'}端坐标: {green_extreme}")
    
    # 位置关系分析 (common)
    print("\n颜色位置关系分析:")
    color_centers = {}
    if red_center: color_centers['红色'] = red_center
    if yellow_center: color_centers['黄色'] = yellow_center
    if green_center: color_centers['绿色'] = green_center
    
    colors = list(color_centers.keys())
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1 = colors[i]
            c2 = colors[j]
            center1 = color_centers[c1]
            center2 = color_centers[c2]
            x_diff = abs(center1[0] - center2[0])
            if x_diff <= 5:
                left_right = f"{c1}和{c2}在同一垂直位置（偏差{int(x_diff)}像素）"
            elif center1[0] < center2[0]:
                left_right = f"{c1}在{c2}的左侧（偏差{int(x_diff)}像素）"
            else:
                left_right = f"{c1}在{c2}的右侧（偏差{int(x_diff)}像素）"
            y_diff = abs(center1[1] - center2[1])
            if y_diff <= 5:
                up_down = f"{c1}和{c2}在同一水平位置（偏差{int(y_diff)}像素）"
            elif center1[1] < center2[1]:
                up_down = f"{c1}在{c2}的上方（偏差{int(y_diff)}像素）"
            else:
                up_down = f"{c1}在{c2}的下方（偏差{int(y_diff)}像素）"
            print(f"{left_right}，{up_down}")
    
    if len(color_centers) >= 2:
        sorted_by_x = sorted(color_centers.items(), key=lambda item: item[1][0])
        left_to_right = [color for color, _ in sorted_by_x]
        print(f"从左到右的颜色顺序: {left_to_right}")
        sorted_by_y = sorted(color_centers.items(), key=lambda item: item[1][1])
        top_to_bottom = [color for color, _ in sorted_by_y]
        print(f"从上到下的颜色顺序: {top_to_bottom}")
    
    vis_img = img.copy()
    
    if red_center:
        cv2.circle(vis_img, red_center, 2, (255, 0, 0), -1)
    if yellow_center:
        cv2.circle(vis_img, yellow_center, 2, (255, 0, 0), -1)
    if green_center:
        cv2.circle(vis_img, green_center, 2, (255, 0, 0), -1)
    
    extreme_points = []
    if red_extreme: extreme_points.append(red_extreme)
    if yellow_extreme: extreme_points.append(yellow_extreme)
    if green_extreme: extreme_points.append(green_extreme)
    
    color_line_info = None
    if extreme_points:
        if direction == 'right':
            extreme_dir = 'left'
            extreme_pos = min(p[0] for p in extreme_points)
            far_points = [p for p in extreme_points if abs(p[0] - extreme_pos) <= 5]
        else:
            extreme_dir = 'right'
            extreme_pos = max(p[0] for p in extreme_points)
            far_points = [p for p in extreme_points if abs(p[0] - extreme_pos) <= 5]
        
        if len(far_points) == 1:
            center_point = far_points[0]
            extended_start = (center_point[0], center_point[1] - 10)
            extended_end = (center_point[0], center_point[1] + 10)
            print(f"{'左侧' if extreme_dir == 'left' else '右侧'}只有一种颜色，扩展像素: {center_point}")
            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)
        elif len(far_points) >= 2:
            far_points.sort(key=lambda p: p[1])
            point1 = far_points[0]
            point2 = far_points[-1]
            print(f"{'左侧' if extreme_dir == 'left' else '右侧'}有两种或以上颜色，使用原始方法连接: {point1} 和 {point2}")
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = ((dx ** 2) + (dy ** 2)) ** 0.5
            if length > 0:
                unit_dx = dx / length
                unit_dy = dy / length
            else:
                unit_dx, unit_dy = 0, 0
            extend_length = length / 2
            extended_start = (int(point1[0] - unit_dx * extend_length), 
                              int(point1[1] - unit_dy * extend_length))
            extended_end = (int(point2[0] + unit_dx * extend_length), 
                            int(point2[1] + unit_dy * extend_length))
            cv2.line(vis_img, extended_start, extended_end, (255, 0, 0), 1)
            color_line_info = (extended_start, extended_end)
    
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    json_path = os.path.join("output", f"{name}_res.json")
    vis_img = draw_target_char_edge(vis_img, json_path, target_char, direction, color_line_info)
    
    if not os.path.exists('output'):
        os.makedirs('output')
    vis_output_path = os.path.join('output', f'{name}_color_centers{os.path.splitext(filename)[1]}')
    cv2.imwrite(vis_output_path, vis_img)
    print(f"颜色中心可视化结果已保存到: {vis_output_path}")

if __name__ == "__main__":
    input_images = [r"D:\jjr\ocr_work\xbd\data\image.png"]
    target_char = input("Enter the target text: ")
    direction = 'right' if target_char == 'X' else 'left'
    print(f"Using direction: {direction} for target: {target_char}")
    
    time1 = time.time()
    result = ocr.predict(input_images)
    for i, res in enumerate(result):
        res.save_to_img("output")
        res.save_to_json("output")
        detect_colors(input_images[i], target_char, direction)
    time2 = time.time()
    print((time2 - time1)/len(input_images))
