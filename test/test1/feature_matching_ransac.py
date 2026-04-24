
import cv2
import numpy as np
import os

def feature_matching_ransac(img1_path, img2_path, output_path):
    # Check inputs
    if not os.path.exists(img1_path):
        print(f"Error: Reference image not found at {img1_path}")
        return
    if not os.path.exists(img2_path):
        print(f"Error: Target image not found at {img2_path}")
        return

    # Read images
    img1 = cv2.imread(img1_path) # Query Image (Template)
    img2 = cv2.imread(img2_path) # Train Image (Target)

    if img1 is None or img2 is None:
        print("Error: Could not read images.")
        return

    print(f"Image 1 size: {img1.shape}")
    print(f"Image 2 size: {img2.shape}")

    # Initialize Feature Detector
    # Try SIFT first, fallback to ORB if unavailable
    try:
        print("Attempting to use SIFT...")
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        print(f"SIFT: Found {len(kp1)} keypoints in reference, {len(kp2)} in target.")
        
    except Exception as e:
        print(f"SIFT initialization failed ({e}), switching to ORB...")
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # BFMatcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        print(f"ORB: Found {len(kp1)} keypoints in reference, {len(kp2)} in target.")

    if des1 is None or des2 is None or len(matches) == 0:
        print("Error: No descriptors or matches found.")
        return

    # Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Found {len(good_matches)} good matches after Lowe's ratio test.")

    if len(good_matches) > 4:
        # Extract location of good matches
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        # RANSAC to find Homography
        # cv2.RANSAC is the method used
        # 5.0 is the reprojection threshold
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            inliers_count = np.sum(matchesMask)
            print(f"RANSAC found {inliers_count} inliers out of {len(good_matches)} good matches.")
        else:
            matchesMask = None
            print("RANSAC failed to find a valid mask.")

        h, w = img1.shape[:2]
        
        # If a valid Homography matrix is found, draw the bounding box
        if M is not None:
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            try:
                dst = cv2.perspectiveTransform(pts, M)
                # Draw bounding box on a copy of img2 so we don't mess up the drawMatches later if we wanted clean img2
                # But drawMatches usually takes fresh images or handles it. 
                # Here we draw on img2 directly for the visualization part inside drawMatches is handled by the function.
                # Actually drawMatches creates a new image. 
                # So to see the box, we should probably draw it on the output of drawMatches or modify img2 before.
                # Let's modify img2 for the visualization, but keep a clean one for drawMatches?
                # No, drawMatches stacks images. Let's draw the box on the result of drawMatches?
                # Or easier: draw the box on img2, then call drawMatches.
                
                # Note: drawMatches uses the provided images to create the background. 
                # If we modify img2, the box will appear on the right side.
                img2_with_box = img2.copy()
                img2_with_box = cv2.polylines(img2_with_box, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                print("Bounding box drawn based on Homography.")
            except Exception as e:
                print(f"Could not apply perspective transform: {e}")
                img2_with_box = img2
        else:
            print("Homography matrix could not be computed.")
            img2_with_box = img2

        # Draw matches
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2) # cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

        result_img = cv2.drawMatches(img1, kp1, img2_with_box, kp2, good_matches, None, **draw_params)
        
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to {output_path}")

    else:
        print(f"Not enough matches are found - {len(good_matches)}/4 required for Homography")

if __name__ == "__main__":
    ref_image_path = r"img.png"
    target_image_path = r"img_1.png"
    output_image_path = r"/test/test1/img_1_feature_match.png"
    
    feature_matching_ransac(ref_image_path, target_image_path, output_image_path)
