import math

# 定义数据 # [[5044.0, 974.0], [5134.0, 921.0], [5172.0, 974.0], [5081.0, 1026.0]]
polygon = [[4705.0, 1442.0], [4805.0, 1442.0], [4805.0, 1494.0], [4705.0, 1494.0]]
target_point = (5046,1090)

# 1. 计算多边形中心 (所有顶点坐标的平均值)
num_points = len(polygon)
center_x = sum(p[0] for p in polygon) / num_points
center_y = sum(p[1] for p in polygon) / num_points
center_point = (center_x, center_y)

# 2. 计算给定点到中心的欧几里得距离
distance = math.sqrt((target_point[0] - center_x) ** 2 + (target_point[1] - center_y) ** 2)

print(f"多边形中心点: {center_point}")
print(f"点 {target_point} 到中心的距离: {distance:.4f}")

print("\n" + "="*60)
print("点到多边形各顶点的距离：")
print("="*60)

vertex_distances = []
for i, vertex in enumerate(polygon):
    dist = math.sqrt((target_point[0] - vertex[0]) ** 2 + (target_point[1] - vertex[1]) ** 2)
    vertex_distances.append((i+1, vertex, dist))
    print(f"顶点{i+1} {vertex} 到点 {target_point} 的距离: {dist:.4f}")

print(f"\n最短距离: {min(vertex_distances, key=lambda x: x[2])[2]:.4f}")
print(f"最长距离: {max(vertex_distances, key=lambda x: x[2])[2]:.4f}")
print(f"平均距离: {sum(v[2] for v in vertex_distances) / len(vertex_distances):.4f}")