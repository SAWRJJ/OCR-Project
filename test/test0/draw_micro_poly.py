import json
import cv2
import os
import math

def draw_micro_poly(json_path, output_dir):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图片路径
    image_name = data['micro_image_name']
    image_path = os.path.join(os.path.dirname(json_path), image_name)
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 获取micro_poly坐标
    micro_poly = data['micro_poly']
    
    # 转换为整数坐标
    points = [(int(x), int(y)) for x, y in micro_poly]
    
    # 绘制矩形
    for i in range(4):
        cv2.line(img, points[i], points[(i+1)%4], (0, 255, 0), 2)
    
    # 计算长度和宽度
    # 计算相邻顶点之间的距离
    width = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
    height = math.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2)
    
    # 计算中心位置用于标注
    center_x = (points[0][0] + points[2][0]) // 2
    center_y = (points[0][1] + points[2][1]) // 2
    
    # 标注长宽
    cv2.putText(img, f"W: {width:.1f}", (center_x - 50, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, f"H: {height:.1f}", (center_x - 50, center_y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"annotated_{image_name}")
    cv2.imwrite(output_path, img)
    
    print(f"处理完成: {output_path}")
    print(f"宽度: {width:.1f}, 高度: {height:.1f}")

if __name__ == "__main__":
    # 输入路径
    json_path = r"/test/test0/micro_0005_S.json"
    # 输出目录
    output_dir = r"/test/test0/output"
    
    draw_micro_poly(json_path, output_dir)
