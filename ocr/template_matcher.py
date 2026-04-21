import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("ocr_system")

def rotate_image(image, angle):
    """
    Rotates an image (2D numpy array) by a given angle without cropping.
    The output image will be larger to fit the rotated content.
    Uses black as the border color.
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Get the center of the image
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new bounding box dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0,0,0))


def template_match_with_edge(target_img, template_img_path,
                             initial_angle=0,
                             scale_min=0.2,
                             scale_max=2.0,
                             scale_step=0.1,
                             threshold=0.6,
                             rotation_angles=None):
    """
    使用边缘检测 + 多尺度 + 多角度模板匹配方法进行匹配
    """
    # 检查模板图片是否存在
    if not os.path.exists(template_img_path):
        logger.error(f"模板图片不存在: {template_img_path}")
        return False, 0.0

    # 根据输入类型加载目标图片
    if isinstance(target_img, str):
        if not os.path.exists(target_img):
            logger.error(f"目标图片不存在: {target_img}")
            return False, 0.0
        big = cv2.imdecode(np.fromfile(target_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(target_img, np.ndarray):
        big = target_img
    else:
        logger.error(f"不支持的目标图片输入类型: {type(target_img)}")
        return False, 0.0

    small = cv2.imdecode(np.fromfile(template_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if big is None or small is None:
        logger.error(f"图片读取失败")
        return False, 0.0

    # 如果提供了初始角度，先旋转目标图片
    initial_angle = int(initial_angle)
    if abs(initial_angle) > 5:
        threshold = 0.4
        (h, w) = big.shape[:2]
        center = (w // 2, h // 2)
        # 注意：这里的角度需要取反，因为我们的角度是基于文本方向计算的
        M = cv2.getRotationMatrix2D(center, -initial_angle, 1.0)
        # 使用 BORDER_REPLICATE 以避免黑色边框影响边缘检测
        big = cv2.warpAffine(big, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 转换为灰度图
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 用边缘增强稳定性
    big_edge = cv2.Canny(big_gray, 50, 150)
    small_edge_original = cv2.Canny(small_gray, 50, 150)

    h_big, w_big = big_edge.shape

    best_score = 0
    best_box = None

    if rotation_angles is None:
        rotation_angles = [0] # Default to no rotation

    # 多角度扫描
    for angle in rotation_angles:
        # 旋转模板
        small_edge = rotate_image(small_edge_original, angle)
        
        # 多尺度扫描
        scales = np.arange(scale_min, scale_max, scale_step)
        if len(scales) == 0:
             scales = [1.0]

        for scale in scales:
            # Resize 模板
            resized = cv2.resize(small_edge, None, fx=scale, fy=scale)
            h_small, w_small = resized.shape

            # 模板必须小于大图
            if h_small > h_big or w_small > w_big:
                continue

            result = cv2.matchTemplate(
                big_edge,
                resized,
                cv2.TM_CCOEFF_NORMED
            )

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_box = (max_loc, w_small, h_small)

    target_log_name = os.path.basename(target_img) if isinstance(target_img, str) else 'memory_image'
    logger.info(f"模板匹配 ({os.path.basename(template_img_path)}) 最佳得分: {best_score:.3f} on {target_log_name}")
    print(
        f"模板匹配 ({os.path.basename(template_img_path)}) 最佳得分: {best_score:.3f} on {target_log_name}")
    if best_score > threshold:
        return True, best_score
    else:
        return False, best_score


if __name__ == '__main__':
    # 使用这个 main 函数来测试模板匹配功能
    # 1. 准备一张大图和一张模板图片
    # 2. 将图片路径替换下面的 placeholder
    # 注意: 请使用绝对路径或者保证图片在当前工作目录下
    target_image = r"D:\work\ocr+Transformer\test3\micro_0111_S15.jpg"
    template_image = r"D:\work\ocr+Transformer\resource\ll1_l.png"

    # 3. (可选) 调整参数
    scales = (0.5, 1.5, 0.1)
    angles = [0, 90, 180, 270]
    score_threshold = 0.4

    # 检查图片是否存在
    if not os.path.exists(target_image) or not os.path.exists(template_image):
        print("="*60)
        print("ERROR: 请确保 target_image 和 template_image 的路径正确")
        print(f"  - target_image: {target_image}")
        print(f"  - template_image: {template_image}")
        print("  请替换上面的 'path/to/your/...' 为真实的图片路径.")
        print("="*60)
    else:
        # 配置日志记录器
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 执行模板匹配
        found, score = template_match_with_edge(
            target_img=target_image,
            template_img_path=template_image,
            threshold=score_threshold,
            initial_angle=-37,
        )

        if found:
            print(f"模板匹配成功! 得分: {score:.3f}")
        else:
            print(f"模板匹配失败. 最高得分: {score:.3f}")
