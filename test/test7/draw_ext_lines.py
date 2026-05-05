from PIL import Image, ImageDraw, ImageFont
import numpy as np

image_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test7/t2.jpg"
img = Image.open(image_path)
print(f"图像尺寸: {img.size}")

top_ext = ([10227, 2202], [10576.374916698778, 2020.9985371319588])
bottom_ext = ([10259, 2263], [10608.374916698778, 2081.9985371319585])
vertical_expand = 40

poly = [[10227, 2202], [10310, 2159], [10342, 2220], [10259, 2263]]
dx = poly[1][0] - poly[0][0]  # 10310 - 10227 = 83
dy = poly[1][1] - poly[0][1]  # 2159 - 2202 = -43

angle_rad = math.atan2(dy, dx)
angle_deg = math.degrees(angle_rad)

print(f"dx={dx}, dy={dy}")
print(f"弧度: {angle_rad:.4f}")
print(f"角度: {angle_deg:.4f}°")

xs = []
xs.extend([top_ext[0][0], top_ext[1][0]])
xs.extend([bottom_ext[0][0], bottom_ext[1][0]])
for p in poly:
    xs.append(p[0])

min_x = min(xs)
max_x = max(xs)

top_dx = top_ext[1][0] - top_ext[0][0]
top_dy = top_ext[1][1] - top_ext[0][1]
top_length = np.sqrt(top_dx**2 + top_dy**2)
top_perp_x = -top_dy / top_length
top_perp_y = top_dx / top_length

bottom_dx = bottom_ext[1][0] - bottom_ext[0][0]
bottom_dy = bottom_ext[1][1] - bottom_ext[0][1]
bottom_length = np.sqrt(bottom_dx**2 + bottom_dy**2)
bottom_perp_x = -bottom_dy / bottom_length
bottom_perp_y = bottom_dx / bottom_length

print(f"top_ext 方向: ({top_dx:.2f}, {top_dy:.2f})")
print(f"top_ext 垂直方向: ({top_perp_x:.2f}, {top_perp_y:.2f})")
print(f"bottom_ext 方向: ({bottom_dx:.2f}, {bottom_dy:.2f})")
print(f"bottom_ext 垂直方向: ({bottom_perp_x:.2f}, {bottom_perp_y:.2f})")

poly_np = np.array(poly)
poly_min_y = np.min(poly_np[:, 1])
poly_max_y = np.max(poly_np[:, 1])
poly_min_x = np.min(poly_np[:, 0])
poly_max_x = np.max(poly_np[:, 0])

top_ext_start = np.array(top_ext[0])
top_ext_end = np.array(top_ext[1])
bottom_ext_start = np.array(bottom_ext[0])
bottom_ext_end = np.array(bottom_ext[1])

top_ext_start_perp = top_ext_start + np.array([top_perp_x, top_perp_y]) * vertical_expand
top_ext_end_perp = top_ext_end + np.array([top_perp_x, top_perp_y]) * vertical_expand
bottom_ext_start_perp = bottom_ext_start + np.array([bottom_perp_x, bottom_perp_y]) * vertical_expand
bottom_ext_end_perp = bottom_ext_end + np.array([bottom_perp_x, bottom_perp_y]) * vertical_expand

all_x = [poly_min_x, poly_max_x,
         top_ext_start_perp[0], top_ext_end_perp[0],
         bottom_ext_start_perp[0], bottom_ext_end_perp[0]]
all_y = [poly_min_y, poly_max_y,
         top_ext_start_perp[1], top_ext_end_perp[1],
         bottom_ext_start_perp[1], bottom_ext_end_perp[1]]

exp_min_x = min(all_x)
exp_max_x = max(all_x)
exp_min_y = min(all_y)
exp_max_y = max(all_y)

x0 = max(int(exp_min_x), 0)
x1 = min(int(exp_max_x), img.size[0])
y0 = max(int(exp_min_y), 0)
y1 = min(int(exp_max_y), img.size[1])

print(f"扩展后区域: min_x={exp_min_x:.2f}, max_x={exp_max_x:.2f}, min_y={exp_min_y:.2f}, max_y={exp_max_y:.2f}")
print(f"裁剪区域: x0={x0}, x1={x1}, y0={y0}, y1={y1}")

draw = ImageDraw.Draw(img)

top_start = (int(top_ext[0][0]), int(top_ext[0][1]))
top_end = (int(top_ext[1][0]), int(top_ext[1][1]))
bottom_start = (int(bottom_ext[0][0]), int(bottom_ext[0][1]))
bottom_end = (int(bottom_ext[1][0]), int(bottom_ext[1][1]))

draw.line([top_start, top_end], fill=(255, 0, 0), width=3)
draw.line([bottom_start, bottom_end], fill=(0, 255, 0), width=3)

draw.line([tuple(top_ext_start_perp.astype(int)), tuple(top_ext_end_perp.astype(int))], fill=(0, 255, 255), width=3)
draw.line([tuple(bottom_ext_start_perp.astype(int)), tuple(bottom_ext_end_perp.astype(int))], fill=(255, 0, 255), width=3)

draw.line([top_start, tuple(top_ext_start_perp.astype(int))], fill=(0, 255, 255), width=2)
draw.line([top_end, tuple(top_ext_end_perp.astype(int))], fill=(0, 255, 255), width=2)
draw.line([bottom_start, tuple(bottom_ext_start_perp.astype(int))], fill=(255, 0, 255), width=2)
draw.line([bottom_end, tuple(bottom_ext_end_perp.astype(int))], fill=(255, 0, 255), width=2)

draw.ellipse([top_start[0]-5, top_start[1]-5, top_start[0]+5, top_start[1]+5], fill=(0, 0, 255))
draw.ellipse([top_end[0]-5, top_end[1]-5, top_end[0]+5, top_end[1]+5], fill=(0, 0, 255))
draw.ellipse([bottom_start[0]-5, bottom_start[1]-5, bottom_start[0]+5, bottom_start[1]+5], fill=(255, 255, 0))
draw.ellipse([bottom_end[0]-5, bottom_end[1]-5, bottom_end[0]+5, bottom_end[1]+5], fill=(255, 255, 0))

draw.ellipse([int(top_ext_start_perp[0])-5, int(top_ext_start_perp[1])-5, int(top_ext_start_perp[0])+5, int(top_ext_start_perp[1])+5], fill=(0, 255, 255))
draw.ellipse([int(top_ext_end_perp[0])-5, int(top_ext_end_perp[1])-5, int(top_ext_end_perp[0])+5, int(top_ext_end_perp[1])+5], fill=(0, 255, 255))
draw.ellipse([int(bottom_ext_start_perp[0])-5, int(bottom_ext_start_perp[1])-5, int(bottom_ext_start_perp[0])+5, int(bottom_ext_start_perp[1])+5], fill=(255, 0, 255))
draw.ellipse([int(bottom_ext_end_perp[0])-5, int(bottom_ext_end_perp[1])-5, int(bottom_ext_end_perp[0])+5, int(bottom_ext_end_perp[1])+5], fill=(255, 0, 255))

draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)

font = ImageFont.load_default()
draw.text((x0 + 5, y0 + 5), f"x0={x0}", fill=(255, 255, 0))
draw.text((x0 + 5, y0 + 20), f"y0={y0}", fill=(255, 255, 0))
draw.text((x1 - 80, y1 - 20), f"x1={x1}", fill=(255, 255, 0))
draw.text((x1 - 80, y1 - 5), f"y1={y1}", fill=(255, 255, 0))
draw.text((x0 + 5, y0 + 40), f"垂直扩展={vertical_expand}", fill=(255, 255, 0))

output_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test7/t2_with_lines.jpg"
img.save(output_path)
print(f"结果已保存到: {output_path}")

img_array = np.array(img)
patch = img_array[y0:y1, x0:x1]
patch_img = Image.fromarray(patch)
patch_path = "/Users/saw/WorkSpace/work/OCR-Project/test/test7/t2_patch.jpg"
patch_img.save(patch_path)
print(f"Patch 已保存到: {patch_path}")
print(f"Patch 尺寸: {patch.shape[1]}x{patch.shape[0]}")