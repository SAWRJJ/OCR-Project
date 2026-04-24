import os
from PIL import Image

def rotate_image(image_path, angle, output_path):
    """
    Rotates an image by a given angle and saves it.

    Args:
        image_path (str): Path to the input image.
        angle (float): The rotation angle in degrees.
        output_path (str): Path to save the rotated image.
    """
    try:
        with Image.open(image_path) as img:
            rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            rotated_img.save(output_path)
            print(f"Image successfully rotated by {angle} degrees and saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The script is in the same folder as the image
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Find the first jpg image in the folder
    image_name = "ll1_l.jpg"
    for file in os.listdir(current_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_name = file
            break

    if not image_name:
        print("No image found in the script's directory.")
    else:
        image_path = os.path.join(current_folder, image_name)
        
        angle = 37

        # Create output path
        file_name, file_ext = os.path.splitext(image_name)
        output_image_name = f"{file_name}_rotated_{angle}{file_ext}"
        output_path = os.path.join(current_folder, output_image_name)

        rotate_image(image_path, angle, output_path)
