from PIL import Image, ImageDraw

image_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test12/micro_0090_X.jpg"
output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test12/micro_0090_X_with_points.jpg"

img = Image.open(image_path)
draw = ImageDraw.Draw(img)

points = [[79, 40], [128, 53], [117, 99], [68, 87]]

for point in points:
    x, y = point
    draw.ellipse([x-3, y-3, x+3, y+3], fill='red', outline='red')

img.save(output_path)
print(f"Image saved to {output_path}")