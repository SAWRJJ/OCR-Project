import shutil
from pathlib import Path

def clear_output_folder(output_path="output"):
    output_dir = Path(output_path)
    if output_dir.exists() and output_dir.is_dir():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"已清空 output 文件夹")
    else:
        print(f"output 文件夹不存在")

if __name__ == "__main__":
    clear_output_folder()
