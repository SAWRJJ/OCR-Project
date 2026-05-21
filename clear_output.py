import shutil
from pathlib import Path

def clear_folder(folder_path):
    folder_dir = Path(folder_path)
    if folder_dir.exists() and folder_dir.is_dir():
        for item in folder_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"已清空 {folder_path} 文件夹")
    else:
        print(f"{folder_path} 文件夹不存在")

def clear_output_folder(output_path="output"):
    clear_folder(output_path)
    clear_folder("splits")

if __name__ == "__main__":
    clear_folder("output")
    clear_folder("splits")
