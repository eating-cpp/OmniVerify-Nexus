import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 读取图像并保持彩色
def read_image_with_chinese_path(path):
    with open(path, 'rb') as f:
        image_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    return image

# 保存图像
def save_image_with_chinese_path(image, path):
    result, image_bytes = cv.imencode('.png', image)
    if result:
        with open(path, 'wb') as f:
            f.write(image_bytes.tobytes())
    else:
        raise ValueError(f"无法保存图像到 {path}")

if __name__ == "__main__":
    # 检查命令行参数是否足够
    if len(sys.argv) != 3:
        print("用法: python ORB.py 路径1 路径2")
        sys.exit(1)

    # 获取命令行输入的图像路径
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]

    # 读取图像
    img1 = read_image_with_chinese_path(img1_path)
    img2 = read_image_with_chinese_path(img2_path)

    # 检查图像是否成功读取
    if img1 is None:
        raise ValueError(f"无法读取图像 {img1_path}")
    if img2 is None:
        raise ValueError(f"无法读取图像 {img2_path}")

    # 创建 ORB 检测器
    orb = cv.ORB_create()

    # 检测关键点和计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 创建 BFMatcher 对象
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 按距离排序匹配结果
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配的关键点对
    match_kp1 = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 2)
    match_kp2 = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 2)

    # 获取 image1 和 image2 的高度和宽度
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 创建一个空白图像，用于存储最终的可视化结果
    total_height = height1 + height2
    total_width = width1 + width2
    viz = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # 将 image1 放置在输出图像的左上角
    viz[:height1, :width1] = img1
    # 将 image2 放置在输出图像的右下角
    viz[height1:, width1:] = img2

    # 绘制匹配线
    for (pt1, pt2) in zip(match_kp1, match_kp2):
        # 将 image2 中的关键点坐标调整到输出图像的正确位置
        pt2_adjusted = (pt2[0] + width1, pt2[1] + height1)
        # 生成随机颜色
        color = tuple(np.random.randint(0, 256, 3).tolist())
        # 绘制匹配线
        cv.line(viz, tuple(map(int, pt1)), tuple(map(int, pt2_adjusted)), color, 2)
        # 绘制关键点圆圈
        cv.circle(viz, tuple(map(int, pt1)), 4, (0, 0, 255), 2)
        cv.circle(viz, tuple(map(int, pt2_adjusted)), 4, (0, 0, 255), 2)

    # 定义输出文件路径
    output_folder = './result/'
    output_file_path = os.path.join(output_folder, 'ORBoutput.png')

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"> \t创建文件夹: {output_folder}")

    # 打印输出文件路径，确保路径正确
    print(f"> \t输出文件路径: {output_file_path}")

    # 保存图像
    save_image_with_chinese_path(viz, output_file_path)
    print(f"> \t已将可视化结果保存到 {output_file_path}")

    # # 显示结果
    # plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
    # plt.axis("off")  # 关闭坐标轴
    # plt.imshow(cv.cvtColor(viz, cv.COLOR_BGR2RGB))  # 将 BGR 格式转换为 RGB 格式
    # plt.show()