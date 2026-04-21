
import cv2
import numpy as np
import os

def get_contours(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None, None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu's binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small noise contours if necessary (e.g., area < 10)
    # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
    
    return img, contours

def match_contours(ref_path, target_path, output_path, match_threshold=0.2):
    print(f"Loading reference: {ref_path}")
    ref_img, ref_contours = get_contours(ref_path)
    if ref_contours is None: return

    print(f"Loading target: {target_path}")
    target_img, target_contours = get_contours(target_path)
    if target_contours is None: return

    print(f"Reference contours: {len(ref_contours)}")
    print(f"Target contours: {len(target_contours)}")

    matched_contours = []
    
    # Iterate through all target contours
    for i, target_cnt in enumerate(target_contours):
        best_score = float('inf')
        
        # Compare with all reference contours
        for ref_cnt in ref_contours:
            # Match shapes: Lower score means better match
            # Method 1 (I1) is usually fine.
            try:
                score = cv2.matchShapes(target_cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_score:
                    best_score = score
            except Exception as e:
                # Sometimes matchShapes fails if contours are too simple/small
                continue
        
        # If the best score is within threshold, consider it a match
        if best_score < match_threshold:
            print(f"Match found for target contour {i} with score: {best_score:.4f}")
            matched_contours.append(target_cnt)

    # Draw matches on the target image
    # Green for matches
    result_img = target_img.copy()
    if matched_contours:
        cv2.drawContours(result_img, matched_contours, -1, (0, 255, 0), 2)
        print(f"Total matched contours: {len(matched_contours)}")
    else:
        print("No matches found.")

    cv2.imwrite(output_path, result_img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    ref_image_path = r"d:\work\ocr+Transformer\test1\img.png"
    target_image_path = r"d:\work\ocr+Transformer\test1\img_1.png"
    output_image_path = r"d:\work\ocr+Transformer\test1\img_1_matched.png"
    
    # Threshold can be adjusted. 0.0 is perfect match. 
    # 0.1 is very close, 0.2-0.3 is usually good for similar shapes.
    match_contours(ref_image_path, target_image_path, output_image_path, match_threshold=0.02)
