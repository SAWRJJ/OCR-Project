import cv2
import numpy as np


def industrial_match(big_path, small_path,
                     scale_min=0.2,
                     scale_max=2.0,
                     scale_step=0.1,
                     threshold=0.6):

    big = cv2.imread(big_path)
    small = cv2.imread(small_path)

    if big is None or small is None:
        print("图片读取失败")
        return False

    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 用边缘增强稳定性
    big_edge = cv2.Canny(big_gray, 50, 150)
    small_edge = cv2.Canny(small_gray, 50, 150)

    h_big, w_big = big_edge.shape

    best_score = 0
    best_box = None

    # 多尺度扫描
    for scale in np.arange(scale_min, scale_max, scale_step):

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

    print(f"最佳匹配得分: {best_score:.3f}")

    if best_score > threshold and best_box is not None:
        top_left, w, h = best_box
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(big, top_left, bottom_right, (0, 0, 255), 2)
        print("检测到目标")
        found = True
    else:
        print("未检测到")
        found = False

    cv2.imshow("result", big)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return found


if __name__ == "__main__":
    industrial_match("img_1.png", "img.png",
                     scale_min=0.3,
                     scale_max=1.5,
                     scale_step=0.05,
                     threshold=0.3)
