import os

import cv2
import numpy as np
import json
from collections import deque
import sys
from ocr.find_boundary_dark import find_closed_dark_regions, visualize_all_dark_regions, find_drak_remove

def point_in_poly(point, poly):
    x, y = point
    poly_arr = np.array(poly, dtype=np.int32)
    return cv2.pointPolygonTest(poly_arr, (float(x), float(y)), False) >= 0


def center_to_border_distance(center, img_shape):
    h, w = img_shape[:2]
    x, y = center
    return min(x, y, w - x - 1, h - y - 1)


def find_all_white_regions(img, white_threshold=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    white_mask = (gray > white_threshold).astype(np.uint8)

    labeled = np.zeros_like(white_mask, dtype=np.int32)
    label = 0
    regions = []

    for row in range(h):
        for col in range(w):
            if white_mask[row, col] == 1 and labeled[row, col] == 0:
                label += 1
                region_pixels = []
                queue = deque([(row, col)])
                labeled[row, col] = label

                while queue:
                    r, c = queue.popleft()
                    region_pixels.append((r, c))

                    neighbors = [
                        (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
                        (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1),
                    ]

                    for nr, nc in neighbors:
                        if 0 <= nr < h and 0 <= nc < w:
                            if white_mask[nr, nc] == 1 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = label
                                queue.append((nr, nc))

                if region_pixels:
                    regions.append({
                        'label': label,
                        'pixels': region_pixels,
                        'min_col': min(p[1] for p in region_pixels),
                        'max_col': max(p[1] for p in region_pixels),
                        'min_row': min(p[0] for p in region_pixels),
                        'max_row': max(p[0] for p in region_pixels),
                    })

    return regions, white_mask


def is_ring_region(region, img_shape):
    bbox_width = region['max_col'] - region['min_col'] + 1
    bbox_height = region['max_row'] - region['min_row'] + 1
    bbox_area = bbox_width * bbox_height
    pixel_count = len(region['pixels'])

    fill_ratio = pixel_count / bbox_area if bbox_area > 0 else 0

    if fill_ratio < 0.5:
        return True
    return False


def fit_circle_contour_method(region, img_shape):
    h, w = img_shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    for pixel in region['pixels']:
        region_mask[pixel[0], pixel[1]] = 255

    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)

    if perimeter == 0:
        return None, None, 0.0

    circularity = 4 * np.pi * area / (perimeter ** 2)

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    return (int(x), int(y)), int(radius), float(circularity)


def detect_circular_white_regions(regions, img_shape, closed_circles=None, min_circularity=0.8):
    if closed_circles is None:
        closed_circles = []

    results = []
    too_closed_res = []
    j = 0
    for i, region in enumerate(regions):
        center, radius, circularity = fit_circle_contour_method(region, img_shape)
        if center is not None and circularity >= min_circularity and not is_ring_region(region, img_shape):
            too_close = False
            for cc in closed_circles:
                dist = np.sqrt((center[0] - cc['center'][0])**2 + (center[1] - cc['center'][1])**2)
                if dist < 2:
                    too_close = True
                    too_closed_res.append({
                        'region_id': j + 1,
                        'pixel_count': len(cc['pixels']),
                        'center': cc["center"],
                        'radius': cc['radius'],
                        'pixels': cc['pixels'],
                        'bbox': (cc['min_row'], cc['min_col'], cc['max_row'], cc['max_col'])
                    })
                    j+=1
                    break
            if not too_close:
                results.append({
                    'region_id': i + 1,
                    'pixel_count': len(region['pixels']),
                    'center': center,
                    'radius': radius,
                    'circularity': circularity,
                    'pixels': region['pixels'],
                    'bbox': (region['min_row'], region['min_col'], region['max_row'], region['max_col'])
                })


    return results,too_closed_res


def visualize_white_circular_regions(img, results, output_path):
    h, w = img.shape[:2]
    vis_img = img.copy()

    for result in results:
        center = result['center']
        radius = result['radius']

        for pixel in result['pixels']:
            vis_img[pixel[0], pixel[1]] = [0, 255, 0]

        cv2.circle(vis_img, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)
        cv2.circle(vis_img, (int(center[0]), int(center[1])), 3, (203, 192, 255), -1)

    # for i, result in enumerate(results):
    #     text = f"Region {result['region_id']}: center=({result['center'][0]:.1f}, {result['center'][1]:.1f}), " \
    #            f"radius={result['radius']:.1f}, circularity={result['circularity']:.3f}"
    #     cv2.putText(vis_img, text, (10, 30 + i * 25),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, vis_img)
    print(f"Saved: {output_path}")

    return results

def find_white_circles(img0, white_threshold=200,textbox_center=None,poly=None,output_path=None,min_circularity=0.8,target_char=None,dark_polygon = None,is_linear=False,ori_img=None,roi_offset=None):
    if dark_polygon is None:   
        h, w = img0.shape[:2]
        # img = img0[10:h-10, :] if h > 20 else img0
        mask = np.zeros(img0.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(dark_polygon)], 255)
        img = cv2.bitwise_and(img0, img0, mask=mask)
        regions, white_mask = find_all_white_regions(img, white_threshold=white_threshold)
        print(f"Found {len(regions)} white pixel regions")

        if output_path is not None:
            vis_white = img.copy()
            for region in regions:
                bbox = region.get('bbox', region.get('bounding_rect'))
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(vis_white, (x, y), (x + w, y + h), (255, 0, 0), 1)
            white_vis_path = output_path.replace('.png', '_white_regions.png')
            cv2.imwrite(white_vis_path, vis_white)
            print(f"White regions visualization saved to: {white_vis_path}")

            mask_vis_path = output_path.replace('.png', '_white_mask.png')
            cv2.imwrite(mask_vis_path, white_mask * 255)
            print(f"White mask visualization saved to: {mask_vis_path}")

        print("\nDetecting circular regions...")
        closed_regions = find_closed_dark_regions(img)
        print(f"Found {len(closed_regions)} closed dark circular regions")
        results, too_closed_res = detect_circular_white_regions(regions, img.shape, closed_circles=closed_regions, min_circularity=min_circularity)

        if not textbox_center:
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            textbox_center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
        print(f"文本框中心坐标: {textbox_center}")

        print("\nVisualizing...")
        if output_path is not None:
            visualize_white_circular_regions(img, results, output_path)


        print(f"\nDetected {len(results)} circular white regions:")
        filtered_results = []
        t = 0
        if "6" in target_char or "9" in target_char:
            t = 1
        elif "8" in target_char:
            t = 2
        boundary_dis = 20
        if len(results)-t >1:
            boundary_dis = 30
        for r in results:
            dist = np.sqrt((r['center'][0] - textbox_center[0])**2 + (r['center'][1] - textbox_center[1])**2)
            border_dist = center_to_border_distance(r['center'], img.shape)
            diameter = 2 * r['radius']
            print(f"  Region {r['region_id']}: center=({r['center'][0]:.1f}, {r['center'][1]:.1f}), "
                f"radius={r['radius']:.1f}, circularity={r['circularity']:.3f}, pixels={r['pixel_count']}, "
                f"距文本框距离={dist:.2f}, 距边界距离={border_dist:.2f}")
            if dist <= 160 and not point_in_poly(r['center'], poly) and border_dist > boundary_dis and r['pixel_count']>10:
                filtered_results.append(r)


        print(f"\nFiltered {len(results) - len(filtered_results)} regions with dist > 300")
        results = filtered_results
        for r in results:
            print(f"  Region {r['region_id']}")
        return results
    else:
        if not is_linear:
            mask = np.zeros(img0.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(dark_polygon)], 255)
            img = cv2.bitwise_and(img0, img0, mask=mask)
        else:
            img = img0.copy()
        regions, white_mask = find_all_white_regions(img, white_threshold=white_threshold)
        print(f"Found {len(regions)} white pixel regions")

        if output_path is not None:
            vis_white = img.copy()
            for region in regions:
                bbox = region.get('bbox', region.get('bounding_rect'))
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(vis_white, (x, y), (x + w, y + h), (255, 0, 0), 1)
            white_vis_path = output_path.replace('.png', '_white_regions.png')
            cv2.imwrite(white_vis_path, vis_white)
            print(f"White regions visualization saved to: {white_vis_path}")

            mask_vis_path = output_path.replace('.png', '_white_mask.png')
            cv2.imwrite(mask_vis_path, white_mask * 255)
            print(f"White mask visualization saved to: {mask_vis_path}")

        print("\nDetecting circular regions...")
        ox, oy = 0,0
        if roi_offset is not None:
            ox, oy = roi_offset
        if ori_img is not None:
            closed_regions = find_closed_dark_regions(ori_img,dark_threshold=85,path=output_path)
            vis_img = ori_img.copy()
        else:
            closed_regions = find_closed_dark_regions(img,dark_threshold=85,path=output_path)
            vis_img = img.copy()
        if len(closed_regions) > 0:
            filename = os.path.basename(output_path).replace('_white_circle.png', '_closed_circles_in_white_detection.png')
            vis_path = os.path.join('output', f'{filename}')
            visualize_all_dark_regions(vis_img, [], closed_regions, vis_path)
        for r in closed_regions:
            r['center'] = (r['center'][0] - ox, r['center'][1] - oy)
        print(f"Found {len(closed_regions)} closed dark circular regions")
        if is_linear:
            min_circularity=0.84
        results, too_closed_res = detect_circular_white_regions(regions, img.shape, closed_circles=closed_regions, min_circularity=min_circularity)
        filtered_results = []
        boundary_dis = 30


        for r in results:
            border_dist = abs(center_to_border_distance(r['center'], img.shape))
            r['center'] = (r['center'][0] + ox, r['center'][1] + oy)
            dist = np.sqrt((r['center'][0] - textbox_center[0])**2 + (r['center'][1] - textbox_center[1])**2)
            diameter = 2 * r['radius']
            if is_linear:
                boundary_dis =r['radius']+1
            print(f"  Region {r['region_id']}: center=({r['center'][0]:.1f}, {r['center'][1]:.1f}), "
                f"radius={r['radius']:.1f}, circularity={r['circularity']:.3f}, pixels={r['pixel_count']}, "
                f"距文本框距离={dist:.2f}, 距边界距离={border_dist:.2f}")
            if dist <= 160 and not point_in_poly(r['center'], poly) and r['pixel_count']>10 and r["radius"]>2:
                filtered_results.append(r)

        if output_path is not None:
            visualize_white_circular_regions(img, results, output_path)

        return filtered_results

def main():
    image_name = "micro_0028_S5.jpg"
    image_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test13/" + image_name
    json_path = image_path.replace(".jpg", ".json")
    output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test13/" + image_name.replace(".jpg", "_circles.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    poly = json_data.get('micro_poly', [])

    print("\nFinding white pixel regions...")
    regions, white_mask = find_all_white_regions(img, white_threshold=200)
    print(f"Found {len(regions)} white pixel regions")

    print("\nDetecting circular regions...")
    closed_regions = find_closed_dark_regions(img)
    print(f"Found {len(closed_regions)} closed dark circular regions")
    results, too_closed_res = detect_circular_white_regions(regions, img.shape, closed_circles=closed_regions, min_circularity=0.8)

    x_coords = [point[0] for point in poly]
    y_coords = [point[1] for point in poly]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    textbox_center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
    print(f"文本框中心坐标: {textbox_center}")

    print("\nVisualizing...")
    visualize_white_circular_regions(img, results, output_path)

    print(f"\nDetected {len(results)} circular white regions:")
    filtered_results = []
    for r in results:
        dist = np.sqrt((r['center'][0] - textbox_center[0])**2 + (r['center'][1] - textbox_center[1])**2)
        if dist <= 300:
            filtered_results.append(r)
            print(f"  Region {r['region_id']}: center=({r['center'][0]:.1f}, {r['center'][1]:.1f}), "
                  f"radius={r['radius']:.1f}, circularity={r['circularity']:.3f}, pixels={r['pixel_count']}, "
                  f"距文本框距离={dist:.2f}")

    print(f"\nFiltered {len(results) - len(filtered_results)} regions with dist > 300")
    results = filtered_results


if __name__ == "__main__":
    main()