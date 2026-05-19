import cv2
import numpy as np

img_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test19/micro_0185_OO_0_SL20VI.jpg"
img = cv2.imread(img_path)

micro_poly = np.array([[297, 74], [490, 37], [493, 62], [300, 99]], dtype=np.int32)

color_centers = {
    'blue': [(435, 105)],
    'green': [(325, 82)],
    'red': [(341, 79)],
    'white': [(399, 69), (376, 74), (450, 103)],
    'yellow': [(356, 76), (310, 85)]
}


def find_farthest_point(poly, centers):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    left_x = min(poly[:, 0])
    right_x = max(poly[:, 0])

    filtered = {k: v for k, v in centers.items() if k not in ['blue', 'white']}

    all_points = []
    for color, pts in filtered.items():
        for p in pts:
            dist_left = p[0] - left_x
            dist_right = right_x - p[0]
            all_points.append((color, p[0], p[1], dist_left, dist_right))

    if not all_points:
        return None, ('none', left_x)

    farthest = max(all_points, key=lambda x: max(x[3], x[4]))
    color, x, y, dist_left, dist_right = farthest

    if dist_left >= dist_right:
        side_x = left_x
        side = 'left'
    else:
        side_x = right_x
        side = 'right'

    return (color, x, y), (side, side_x)


def get_rect(rect_x1, rect_y1, rect_x2, rect_y2):
    left = min(rect_x1, rect_x2)
    right = max(rect_x1, rect_x2)
    top = min(rect_y1, rect_y2)
    bottom = max(rect_y1, rect_y2)
    return left, top, right, bottom


def point_in_rect(px, py, rect):
    left, top, right, bottom = rect
    return left <= px <= right and top <= py <= bottom


def filter_white_points_in_rect(poly, centers, white_points0=None, is_linear=None):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    point_info, side_info = find_farthest_point(poly, centers)

    if point_info is None:
        filtered_centers = centers.copy()
        return filtered_centers, None, []

    color, farthest_x, farthest_y = point_info
    side, side_x = side_info

    top_y = min(poly[:, 1])
    bottom_y = max(poly[:, 1])

    rect_x1, rect_y1 = side_x, top_y
    rect_x2, rect_y2 = farthest_x, bottom_y
    rect = get_rect(rect_x1, rect_y1, rect_x2, rect_y2)

    if is_linear is False:
        left, top, right, bottom = rect
        rect = (left, top - 15, right, bottom + 15)

    if white_points0:
        white_points = white_points0
    else:
        white_points = centers.get('white', [])

    filtered_white = []
    removed_white = []
    for i, (wx, wy) in enumerate(white_points):
        inside = point_in_rect(wx, wy, rect)
        if inside:
            filtered_white.append((wx, wy))
        else:
            removed_white.append((i + 1, wx, wy))
            print(f"Removed White point {i+1} ({wx}, {wy}): OUTSIDE -> excluded")

    filtered_centers = centers.copy()
    filtered_centers['white'] = filtered_white

    return filtered_centers, rect, removed_white


def draw_rectangle(img, poly, centers):
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    point_info, side_info = find_farthest_point(poly, centers)
    color, x, y = point_info
    side, side_x = side_info

    top_y = min(poly[:, 1])
    bottom_y = max(poly[:, 1])

    cv2.rectangle(img, (side_x, top_y), (x, bottom_y), (0, 255, 255), 2)

    cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
    cv2.putText(img, f"{color}({x},{y})", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    print(f"Original side: {side} x={side_x}, Farthest point x={x}")
    print(f"Rectangle: ({side_x}, {top_y}) -> ({x}, {bottom_y})")

    return img


if __name__ == '__main__':
    filtered_centers, rect, removed = filter_white_points_in_rect(micro_poly, color_centers)
    print(f"\nFiltered white centers: {filtered_centers}")

    result_img = draw_rectangle(img.copy(), micro_poly, color_centers)

    cv2.polylines(result_img, [micro_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.rectangle(result_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)

    point_info, _ = find_farthest_point(micro_poly, color_centers)
    color, x, y = point_info
    cv2.circle(result_img, (x, y), 8, (0, 255, 255), -1)
    cv2.putText(result_img, f"{color}({x},{y})", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for i, (wx, wy) in enumerate(filtered_centers['white']):
        cv2.circle(result_img, (wx, wy), 6, (255, 255, 255), -1)
        cv2.putText(result_img, f"W{i + 1}({wx},{wy})", (wx + 6, wy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test19/micro_0185_OO_0_SL20VI_annotated.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"\nSaved to {output_path}")