import cv2
import numpy as np

def find_cluster_centers(points, distance_threshold=30):
    """
    根据距离聚集点集，并返回每个聚集点集的中心点

    参数:
        points: 点集列表，如 [(x1,y1), (x2,y2), ...]
        distance_threshold: 聚集距离阈值，默认30

    返回:
        center_points: 中心点列表，如 [(cx1,cy1), (cx2,cy2), ...]
    """
    points = np.array(points)
    n = len(points)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]

        while queue:
            current = queue.pop(0)
            for j in range(n):
                if not visited[j]:
                    dist = np.sqrt((points[current][0] - points[j][0])**2 + (points[current][1] - points[j][1])**2)
                    if dist < distance_threshold:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)

    center_points = []
    for cluster in clusters:
        cluster_points_arr = points[cluster]
        centroid = np.mean(cluster_points_arr, axis=0)
        distances = [np.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2) for p in cluster_points_arr]
        min_idx = np.argmin(distances)
        center_points.append(tuple(cluster_points_arr[min_idx]))

    return center_points


if __name__ == "__main__":
    img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test11/micro_0023_SF.jpg"
    output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test11/micro_0023_SF0.jpg"

    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        exit(1)

    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")

    points = [(230, 6), (249, 3), (374, 7), (392, 4)]
    distance_threshold = 30

    center_points = find_cluster_centers(points, distance_threshold=distance_threshold)

    print(f"\n距离阈值: {distance_threshold}")
    print(f"输入点数量: {len(points)}")
    print(f"中心点数量: {len(center_points)}")
    print(f"中心点列表: {center_points}")

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (0, 128, 255)
    ]

    # points_arr = np.array(points)
    # for i, pt in enumerate(points_arr):
    #     color = colors[i % len(colors)]
    #     cv2.circle(img, tuple(pt), 2, color, -1)
    #     cv2.putText(img, str(i+1), (int(pt[0]) + 10, int(pt[1]) - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # for i in range(len(points) - 1):
    #     cv2.line(img, points[i], points[i+1], (200, 200, 200), 1)

    for i, center in enumerate(center_points):
        color = colors[i % len(colors)]
        cv2.circle(img, center, 2, (255, 255, 255), -1)
        cv2.circle(img, center, 2, color, 2)
        cv2.putText(img, f"M{i+1}", (int(center[0]) + 15, int(center[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"\n可视化结果已保存: {output_path}")