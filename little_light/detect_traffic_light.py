import cv2
import numpy as np
import os

def detect_traffic_light_contour(img_path, output_path=None):
    """
    Detects traffic light contours (vertical rectangle with ~3 circles) in small images.
    Follows the pipeline:
    1. Preprocessing (Upscale -> Denoise -> Threshold/Canny -> Morphology)
    2. Contour Filtering (Rectangle shape, Aspect Ratio)
    3. Internal Structure Check (Circles/Blobs inside)
    """
    
    # 1. Read Image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error: Could not read image {img_path}")
        return

    h, w = original_img.shape[:2]
    print(f"Original Size: {w}x{h}")

    # 2. Preprocessing
    # Upscale if small (e.g., width < 100)
    scale_factor = 1
    processed_img = original_img.copy()
    if w < 100 or h < 100:
        scale_factor = 4  # Scale up 4x as suggested
        processed_img = cv2.resize(original_img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled by {scale_factor}x to: {processed_img.shape[1]}x{processed_img.shape[0]}")

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Edge Detection / Thresholding
    # Option A: Canny (Adaptive)
    # v = np.median(denoised)
    # sigma = 0.33
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges = cv2.Canny(denoised, lower, upper)
    
    # Option B: Otsu Thresholding (often better for solid shapes)
    # Use THRESH_BINARY_INV because traffic lights are usually darker than sky/background
    # Or adaptive if lighting varies
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology to close gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Close: connect broken parts
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Open: remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    debug_vis = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR) # For visualization

    # 3. Contour Detection
    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 * (scale_factor**2): # Filter very small noise
            continue
            
        # Approx Polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        x, y, w_rect, h_rect = cv2.boundingRect(approx)
        if w_rect == 0: continue
        aspect_ratio = float(h_rect) / w_rect
        
        # Check if it's roughly rectangular (4-6 vertices allowed for imperfect shapes)
        if len(approx) >= 4 and len(approx) <= 8:
            
            # Traffic lights are usually vertical rectangles (AR > 1.2 approx)
            if aspect_ratio > 1.2: 
                candidates.append((cnt, (x, y, w_rect, h_rect)))

    print(f"Found {len(candidates)} candidate rectangles.")
    
    final_matches = []
    
    # 4. Internal Structure Check (Circles inside)
    for i, (cnt, rect) in enumerate(candidates):
        x, y, w_rect, h_rect = rect
        roi = gray[y:y+h_rect, x:x+w_rect]
        
        # Look for circles in ROI
        # HoughCircles can be tricky on small ROIs. 
        # Alternative: Threshold and find small contours inside.
        
        # Let's try simple blob/contour detection inside ROI
        _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        roi_contours, _ = cv2.findContours(roi_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        circles_found = 0
        circle_centers = []
        valid_circle_contours = []
        
        for sub_cnt in roi_contours:
            sub_area = cv2.contourArea(sub_cnt)
            # Filter by area relative to ROI
            if sub_area < (w_rect * h_rect) * 0.05 or sub_area > (w_rect * h_rect) * 0.3:
                continue
                
            # Check circularity: 4*pi*Area / Perimeter^2
            sub_peri = cv2.arcLength(sub_cnt, True)
            if sub_peri == 0: continue
            circularity = 4 * np.pi * sub_area / (sub_peri * sub_peri)
            
            if circularity > 0.6: # Somewhat circular
                circles_found += 1
                valid_circle_contours.append(sub_cnt)
                M = cv2.moments(sub_cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    circle_centers.append((cX, cY))

        # Check for vertical alignment of circles
        is_traffic_light = False
        if circles_found >= 2: # At least 2 or 3 lights
            # Check vertical alignment (similar X coordinates)
            xs = [c[0] for c in circle_centers]
            if np.std(xs) < w_rect * 0.2: # Centers align vertically
                is_traffic_light = True
        
        # Heuristic: If we found a nice rectangle and circles/blobs inside
        if is_traffic_light or (aspect_ratio > 2.0 and circles_found >= 1):
             final_matches.append(cnt)
             cv2.drawContours(processed_img, [cnt], -1, (0, 255, 0), 2)
             
             # Draw internal circles
             for c_cnt in valid_circle_contours:
                 # Shift contour to global coordinates
                 c_cnt_shifted = c_cnt + np.array([[x, y]])
                 cv2.drawContours(processed_img, [c_cnt_shifted], -1, (0, 0, 255), 1)

             cv2.putText(processed_img, f"Signal {circles_found}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
             print(f"Match found at ({x}, {y}, {w_rect}, {h_rect}) with {circles_found} internal circles.")

    # 5. Output
    if output_path:
        cv2.imwrite(output_path, processed_img)
        print(f"Result saved to {output_path}")
        
        # Save debug image too
        debug_path = output_path.replace(".jpg", "_debug.jpg")
        cv2.imwrite(debug_path, debug_vis)

import sys

# Test on a file if it exists
if __name__ == "__main__":
    test_img = "micro_0004_S1.jpg" 
        
    if os.path.exists(test_img):
        detect_traffic_light_contour(test_img, "detected_traffic_light.jpg")
    else:
        print(f"Image {test_img} not found.")
