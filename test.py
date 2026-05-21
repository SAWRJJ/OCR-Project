import math

# 定义数据
polygon = [[0, 40], [134, 40], [134, 114], [0, 114]]
target_point = (191, 86)

# 1. 计算多边形中心 (所有顶点坐标的平均值)
num_points = len(polygon)
center_x = sum(p[0] for p in polygon) / num_points
center_y = sum(p[1] for p in polygon) / num_points
center_point = (center_x, center_y)

# 2. 计算给定点到中心的欧几里得距离
distance = math.sqrt((target_point[0] - center_x) ** 2 + (target_point[1] - center_y) ** 2)

print(f"多边形中心点: {center_point}")
print(f"点 {target_point} 到中心的距离: {distance:.4f}")