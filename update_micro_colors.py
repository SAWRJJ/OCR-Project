import json
import os

import cv2

from ocr.visualizer import detect_color_presence_bgr


def update_micro_dir(micro_dir: str):
    for name in os.listdir(micro_dir):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(micro_dir, name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_name = data.get("micro_image_name")
        if not img_name:
            continue
        img_path = os.path.join(micro_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        color_info = detect_color_presence_bgr(img)
        data["color_presence"] = color_info["presence"]
        data["color_stats"] = color_info["stats"]
        data["color_valid_pixels"] = color_info["valid_pixels"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main(output_root: str = "output"):
    if not os.path.exists(output_root):
        return
    for entry in os.listdir(output_root):
        micro_dir = os.path.join(output_root, entry, "micro_img")
        if os.path.isdir(micro_dir):
            update_micro_dir(micro_dir)


if __name__ == "__main__":
    main()

