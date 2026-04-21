import cv2
import os
import numpy as np
from scipy.spatial import KDTree


def get_contours(img_path):
    """
    Extracts contours from an image.
    Returns a list of contours, where each contour is a numpy array of points.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Canny edge detection for better contour extraction
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    return valid_contours


def center_and_scale_contour(contour, num_points=100):
    """
    Centers, scales, and resamples a contour to a fixed number of points.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)

    # 1. Resample to fixed number of points
    dists = np.sqrt(np.sum(np.diff(pts, axis=0, append=[pts[0]]) ** 2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]

    if total_dist == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    interp_dists = np.linspace(0, total_dist, num_points, endpoint=False)
    resampled_pts = np.zeros((num_points, 2), dtype=np.float32)
    resampled_pts[:, 0] = np.interp(interp_dists, cum_dists, np.append(pts[:, 0], pts[0, 0]))
    resampled_pts[:, 1] = np.interp(interp_dists, cum_dists, np.append(pts[:, 1], pts[0, 1]))

    # 2. Center
    centroid = np.mean(resampled_pts, axis=0)
    centered_pts = resampled_pts - centroid

    # 3. Scale
    max_dist = np.max(np.sqrt(np.sum(centered_pts ** 2, axis=1)))
    if max_dist > 0:
        scaled_pts = centered_pts / max_dist
    else:
        scaled_pts = centered_pts

    return scaled_pts


def match_with_kdtree(template_pts, target_pts, threshold=0.2):
    """
    Matches target points against template points using KD-Tree.
    """
    tree = KDTree(template_pts)
    distances, _ = tree.query(target_pts)
    avg_dist = np.mean(distances)

    tree_rev = KDTree(target_pts)
    distances_rev, _ = tree_rev.query(template_pts)
    avg_dist_rev = np.mean(distances_rev)

    combined_dist = (avg_dist + avg_dist_rev) / 2

    return combined_dist, combined_dist < threshold


def main(template_files=None):
    base_dir = r"d:\work\ocr+Transformer\template"
    demo_dir = os.path.join(base_dir, "demo_img")
    
    # Define output directories
    match_results_dir = os.path.join(base_dir, "match_results")
    template_vis_dir = os.path.join(base_dir, "template_vis")
    demo_contours_dir = os.path.join(base_dir, "demo_contours")
    
    for d in [match_results_dir, template_vis_dir, demo_contours_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    template_data = {}

    print("Extracting template contours...")
    for t_file in template_files:
        t_path = os.path.join(base_dir, t_file)
        contours = get_contours(t_path)
        if not contours:
            img = cv2.imread(t_path, 0)
            if img is not None:
                _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found in template: {t_file}")
            continue

        template_cnt = max(contours, key=cv2.contourArea)
        template_pts = center_and_scale_contour(template_cnt)
        template_data[t_file] = (template_pts, template_cnt)
        print(f"Template {t_file} processed.")

        # Visualize template contour
        t_img_vis = cv2.imread(t_path)
        if t_img_vis is not None:
            cv2.drawContours(t_img_vis, [template_cnt], -1, (0, 0, 255), 2)
            t_out_path = os.path.join(template_vis_dir, f"vis_{t_file}")
            cv2.imwrite(t_out_path, t_img_vis)
            print(f"  Template visualization saved to: template_vis/{os.path.basename(t_out_path)}")

    if not template_data:
        print("No templates processed successfully.")
        return

    demo_images = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for demo_file in demo_images:
        demo_path = os.path.join(demo_dir, demo_file)
        img_rgb = cv2.imread(demo_path)
        if img_rgb is None: continue
        
        # Create copies for different visualizations
        img_match = img_rgb.copy()
        img_all_contours = img_rgb.copy()
        
        print(f"\nProcessing demo image: {demo_file}")

        demo_contours = get_contours(demo_path)
        if not demo_contours:
            print("  No contours found.")
            continue
            
        # 1. Visualize ALL contours in demo image
        cv2.drawContours(img_all_contours, demo_contours, -1, (255, 0, 0), 1)
        cnt_out_path = os.path.join(demo_contours_dir, f"cnt_{demo_file}")
        cv2.imwrite(cnt_out_path, img_all_contours)
        print(f"  Demo contours saved to: demo_contours/{os.path.basename(cnt_out_path)}")

        # 2. Match and visualize results
        match_info = []
        for t_file, (t_pts, t_cnt_orig) in template_data.items():
            found_match = False
            best_dist = float('inf')
            best_cnt = None

            for i, d_cnt in enumerate(demo_contours):
                if cv2.contourArea(d_cnt) < 50:
                    continue
                d_pts = center_and_scale_contour(d_cnt)

                avg_dist, is_match = match_with_kdtree(t_pts, d_pts, threshold=0.07)
                
                if is_match:
                    found_match = True
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_cnt = d_cnt

            if found_match:
                print(f"  [MATCH] Template {t_file} found (Dist: {best_dist:.3f}, Area: {cv2.contourArea(best_cnt):.1f})")
                match_info.append((t_file, best_dist, best_cnt))
                cv2.drawContours(img_match, [best_cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(best_cnt)
                cv2.putText(img_match, f"{t_file} ({best_dist:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print(f"  [NO MATCH] Template {t_file} not found")

        if match_info:
            res_out_path = os.path.join(match_results_dir, f"res_{demo_file}")
            cv2.imwrite(res_out_path, img_match)
            print(f"  Match result saved to: match_results/{os.path.basename(res_out_path)}")


if __name__ == "__main__":
    templates = ["S.jpg", "2.jpg","er.jpg"]
    main(templates)
