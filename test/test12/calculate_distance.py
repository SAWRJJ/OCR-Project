import math

def calculate_textbox_center(poly):
    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    textbox_center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
    return textbox_center

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

data = [
    {
        "green_points": [(369, 69), (140, 70)],
        "textbox_contour": [[0, 40], [104, 40], [104, 88], [0, 88]],
        "label": "FXⅡ"
    },
    {
        "green_points": [(360, 72), (502, 72)],
        "textbox_contour": [[7, 44], [107, 39], [107, 93], [7, 97]],
        "label": "XLⅠ"
    }
]

for i, item in enumerate(data):
    print(f"\n=== 组{i+1}: {item['label']} ===")
    textbox_center = calculate_textbox_center(item["textbox_contour"])
    print(f"文本框轮廓: {item['textbox_contour']}")
    print(f"文本框中心坐标: {textbox_center}")
    print(f"绿色独立区域中心坐标: {item['green_points']}")

    for j, green_point in enumerate(item["green_points"]):
        distance = calculate_distance(textbox_center, green_point)
        print(f"  绿色点{j+1} {green_point} 到文本框中心的距离: {distance:.2f}")