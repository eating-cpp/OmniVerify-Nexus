import cv2
import numpy as np
import os
import sys
import random

# 读取图像并保持彩色
def read_image_with_chinese_path(path):
    with open(path, 'rb') as f:
        image_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return image

# 保存图像
def save_image_with_chinese_path(image, path):
    result, image_bytes = cv2.imencode('.jpg', image)
    if result:
        with open(path, 'wb') as f:
            f.write(image_bytes.tobytes())
    else:
        raise ValueError(f"无法保存图像到 {path}")

if __name__ == "__main__":
    # 检查命令行参数是否足够
    if len(sys.argv) != 3:
        print("用法: python sift.py 路径1 路径2")
        sys.exit(1)

    # 获取命令行输入的图像路径
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    # 读取图像
    image1 = read_image_with_chinese_path(img1_path)
    image2 = read_image_with_chinese_path(img2_path)

    # 检查图像是否成功读取
    if image1 is None:
        raise ValueError(f"无法读取图像 {img1_path}")
    if image2 is None:
        raise ValueError(f"无法读取图像 {img2_path}")

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 将图像转换为灰度图进行特征检测
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 检测关键点并计算描述符
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试来筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 创建空白彩色图像用于绘制结果
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    # 大小：长是两幅图像长之和，宽是两幅图像宽之和
    result = np.zeros((h1 + h2, w1 + w2, 3), dtype=np.uint8)

    # 将image1放置在左上角
    result[0:h1, 0:w1] = image1

    # 将image2放置在右下角
    result[h1:h1 + h2, w1:w1 + w2] = image2

    # 绘制匹配的关键点
    for match in good_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        x2 += w1
        y2 += h1

        # 生成随机颜色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 使用随机颜色线条绘制匹配点
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # 定义输出文件路径
    output_folder = './result/'
    output_file_path = os.path.join(output_folder, 'SIFToutpu.jpg')

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"> \t创建文件夹: {output_folder}")

    # 打印输出文件路径，确保路径正确
    print(f"> \t输出文件路径: {output_file_path}")

    # 保存结果图像
    save_image_with_chinese_path(result, output_file_path)
    print(f"> \t已将可视化结果保存到 {output_file_path}")