
import cv2
import numpy as np
import os

def detect_and_visualize_contours(image_path, output_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (Otsu's binarization)
    # You might need to adjust this depending on the image content
    # Alternatively, use Canny edge detection: edges = cv2.Canny(gray, 100, 200)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    # cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours.
    # cv2.RETR_EXTERNAL would retrieve only the extreme outer contours.
    # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours.")

    # Draw all contours on the original image
    # -1 means draw all contours
    # (0, 255, 0) is the color (Green)
    # 2 is the thickness
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    input_image_path = r"/test/test1/img.png"
    output_image_path = r"d:\work\ocr+Transformer\test1\img_contours.png"
    
    detect_and_visualize_contours(input_image_path, output_image_path)
