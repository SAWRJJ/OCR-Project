import numpy as np
import cv2


def calculate_distances(points):
    points_array = np.array(points)
    distances = np.sqrt(np.sum((points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]) ** 2, axis=2))
    return distances


def filter_distance_pairs(points, distances, min_dist=29, max_dist=43):
    filtered_pairs = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distances[i, j]
            if min_dist <= dist <= max_dist:
                filtered_pairs.append((tuple(points[i]), tuple(points[j]), dist))
    return filtered_pairs


def forms_triangle(filtered_pairs):
    unique_points = set()
    for p1, p2, _ in filtered_pairs:
        unique_points.add(p1)
        unique_points.add(p2)
    return len(unique_points) == 3 and len(filtered_pairs) == 3


def draw_distance_lines(points, filtered_pairs, offset_x=20, offset_y=10):
    max_x = max(p[0] for p in points) + offset_x + 50
    max_y = max(p[1] for p in points) + offset_y + 50
    canvas = np.ones((int(max_y), int(max_x), 3), dtype=np.uint8) * 255

    canvas_points = [(p[0] + offset_x, p[1] + offset_y) for p in points]
    point_dict = {tuple(p): idx for idx, p in enumerate(points)}

    for i, p in enumerate(canvas_points):
        cv2.circle(canvas, (int(p[0]), int(p[1])), 5, (0, 0, 0), -1)
        cv2.putText(canvas, f"P{i+1}", (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for p1, p2, dist in filtered_pairs:
        idx1 = point_dict[p1]
        idx2 = point_dict[p2]
        cv2.line(canvas,
                 (int(canvas_points[idx1][0]), int(canvas_points[idx1][1])),
                 (int(canvas_points[idx2][0]), int(canvas_points[idx2][1])),
                 (0, 0, 255), 2)

    return canvas

def if_triangle(points):
    distances = calculate_distances(points)

    print("距离在 29-31 之间的点对:")
    filtered_pairs = filter_distance_pairs(points, distances)
    for p1, p2, dist in filtered_pairs:
        print(f"  {p1} <-> {p2}: {dist:.2f}")

    is_triangle = forms_triangle(filtered_pairs)
    return is_triangle

if __name__ == '__main__':
    points = [(259, 72), (259, 43), (232, 58)]

    distances = calculate_distances(points)

    print("距离在 29-31 之间的点对:")
    filtered_pairs = filter_distance_pairs(points, distances)
    for p1, p2, dist in filtered_pairs:
        print(f"  {p1} <-> {p2}: {dist:.2f}")

    is_triangle = forms_triangle(filtered_pairs)
    print(f"\n是否构成三角形: {is_triangle}")

    canvas = draw_distance_lines(points, filtered_pairs)
    output_path = '/Users/saw/WorkSpace/work/OCR-Project/test/test19/distance_lines.jpg'
    cv2.imwrite(output_path, canvas)
    print(f"\n结果已保存到: {output_path}")