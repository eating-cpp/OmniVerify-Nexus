#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
import os
import tempfile

# -*- coding: utf-8 -*-

def take_photo(base_filename, max_photos):
    # 打开摄像头，0 表示默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    photo_count = 0

    while photo_count < max_photos:
        # 读取摄像头的一帧图像
        ret, frame = cap.read()

        if not ret:
            print("无法获取图像")
            break

        # 显示当前帧图像
        cv2.imshow('Camera', frame)

        # 等待按键事件，按下 's' 键拍照，按下 'q' 键退出
        key = cv2.waitKey(1)
        if key == ord('s'):
            # 生成保存的文件名
            filename = os.path.join(base_filename,f"{photo_count + 1}.jpg")
            # 保存图像
            cv2.imwrite(filename, frame)
            print(f"照片已保存为 {filename}")
            photo_count += 1
        elif key == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

    if photo_count == max_photos:
        return combine_photos(base_filename, max_photos)


def combine_photos(base_filename, max_photos):
    # 确保 res 文件夹存在
    # res_folder = "res"
    # if not os.path.exists(res_folder):
    #     os.makedirs(res_folder)

    # 读取十张照片
    photos = []
    for i in range(1, max_photos + 1):
        temp_file = os.path.join(base_filename,f"{i}.jpg")
        photo = cv2.imread(temp_file)
        if photo is None:
            print(f"\t无法读取 {base_filename}/{i}.jpg")
            return None, None
        else:
            print(f">\t读取 {base_filename}/{i}.jpg")
            photos.append(photo)

    # 获取照片的尺寸
    base_h, base_w, _ = photos[0].shape
    for i in range(len(photos)):
        if photos[i].shape != (base_h, base_w, 3):
            photos[i] = cv2.resize(photos[i], (base_w, base_h))

        # 创建组合画布时使用统一后的尺寸
    combined = np.zeros((3 * base_h, 4 * base_w, 3), dtype=np.uint8)

    # 24张照片分别放置在指定位置
    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3)
        # (3, 0), (3, 1), (3, 2), (3, 3),
    ]

    for i, (row, col) in enumerate(positions):
        start_y = row * base_h
        start_x = col * base_w
        end_y = start_y + base_h
        end_x = start_x + base_w
        combined[start_y:end_y, start_x:end_x] = photos[i]


    # 直接进行过曝处理
    processed_image = process_overexposed_from_image(combined)

    # 保存处理后的图像到临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        cv2.imwrite(temp_file.name, processed_image)
        file_size = os.path.getsize(temp_file.name) / 1024  # 转换为 KB

        # 检测文件大小，如果大于 700KB 则进行缩放
        scale = 1
        while file_size > 1024*3:
            scale *= 0.99  
            processed_image = cv2.resize(processed_image, (int(processed_image.shape[1] * scale), int(processed_image.shape[0] * scale)))
            cv2.imwrite(temp_file.name, processed_image)
            file_size = os.path.getsize(temp_file.name) / 1024  # 重新检测文件大小

    finally:
        # 确保临时文件对象被关闭
        temp_file.close()

    # 删除临时文件
    os.remove(temp_file.name)

    # 调用缩放和保存函数
    scaled_image, final_file_name = scale_and_save(processed_image, base_filename)
    
    # # 展示图片
    # cv2.imshow("Combined Image", scaled_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    return scaled_image, final_file_name

def check_local_contrast(image_path, blocksnum, threshold):
    try:
        with open(image_path, 'rb') as f:
            img_np = np.frombuffer(f.read(), np.uint8)
            color_image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            image = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error: Unable to load image: {e}")
        return 0, []

    h, w = image.shape
    # 计算行数和列数
    rows = int(np.sqrt(blocksnum))
    cols = int(np.ceil(blocksnum / rows))

    # 计算均匀的块大小
    block_h = int(np.ceil(h / rows))
    block_w = int(np.ceil(w / cols))

    overexposed_blocks = 0
    total_blocks = rows * cols
    overexposed_positions = []  # 记录过曝小块的位置

    for i in range(rows):
        for j in range(cols):
            start_y = i * block_h
            start_x = j * block_w
            end_y = min(start_y + block_h, h)
            end_x = min(start_x + block_w, w)
            block = image[start_y:end_y, start_x:end_x]
            if block.size > 0:
                mean_luminance = np.mean(block)
                if mean_luminance > threshold:
                    overexposed_blocks += 1
                    overexposed_positions.append((start_y, start_x, end_y, end_x))
                    # 绘制红色边框表示过曝区域
                    cv2.rectangle(color_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                else:
                    # 绘制蓝色边框表示未过曝区域
                    cv2.rectangle(color_image, (start_x, start_y), (end_x, end_y), (41, 41, 41), 2)

    over_percentage = (overexposed_blocks / total_blocks) * 100
     # 显示标注后的图像
    cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
    # 显示标注后的图像
    cv2.imshow('Annotated Image', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return over_percentage, overexposed_positions


def apply_clahe(image):
    """
    对输入的图像应用对比度受限的自适应直方图均衡化（CLAHE），以增强图像的对比度。

    参数:
    image (numpy.ndarray): 输入的图像，数据类型为 uint8，颜色空间为 BGR。

    返回:
    numpy.ndarray: 应用 CLAHE 后的图像，数据类型为 uint8，颜色空间为 BGR。
    """
    # 将输入的 BGR 图像转换为 Lab 颜色空间
    # Lab 颜色空间包含亮度通道（L）和两个色度通道（A 和 B）
    # 这样可以在不影响颜色信息的情况下单独处理亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 将 Lab 图像分离为三个通道：亮度通道 L、色度通道 A 和色度通道 B
    L, A, B = cv2.split(lab)
    # 创建一个 CLAHE 对象，用于进行对比度受限的自适应直方图均衡化
    # clipLimit=3.0 表示对比度限制参数，控制每个小块直方图的对比度增强程度
    # tileGridSize=(16, 16) 表示将图像分割成 16x16 的小块，每个小块独立进行直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    # 对亮度通道 L 应用 CLAHE 处理，增强其对比度
    L = clahe.apply(L)
    # 将处理后的亮度通道 L 与未处理的色度通道 A 和 B 合并回 Lab 颜色空间
    adjusted_lab = cv2.merge([L, A, B])
    # 将处理后的 Lab 图像转换回 BGR 颜色空间，以便后续使用
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_Lab2BGR)



def process_overexposed_from_image(image, blocksnum=1800 * 1800, threshold=220):
    if image is None:
        print("无法读取图像，请检查输入。")
        return

    # 保存原图副本
    original_image = image.copy()

    # 这里需要将图像保存为临时文件以复用 check_local_contrast 函数
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, image)
        image_path = temp_file.name

    over_percentage, overexposed_positions = check_local_contrast(image_path, blocksnum, threshold)
    print(f"局部过曝区域比例: {over_percentage:.2f}%")

    # 删除临时文件
    os.remove(image_path)

    h, w = image.shape[:2]
    # 计算行数和列数
    rows = int(np.sqrt(blocksnum))
    cols = int(np.ceil(blocksnum / rows))

    # 计算均匀的块大小
    block_h = int(np.ceil(h / rows))
    block_w = int(np.ceil(w / cols))

    all_blocks = []
    for i in range(rows):
        for j in range(cols):
            start_y = i * block_h
            start_x = j * block_w
            end_y = min(start_y + block_h, h)
            end_x = min(start_x + block_w, w)
            all_blocks.append((start_y, start_x, end_y, end_x))

    # 创建一个空白图像用于存储移除过曝区域后的图像
    removed_overexposed_image = image.copy()
    for start_y, start_x, end_y, end_x in overexposed_positions:
        removed_overexposed_image[start_y:end_y, start_x:end_x] = 0

    # 对移除过曝区域后的图像应用 CLAHE 处理
    enhanced_removed_overexposed_image=apply_clahe(image)
    # enhanced_removed_overexposed_image = apply_clahe(removed_overexposed_image)

    return enhanced_removed_overexposed_image
    # return removed_overexposed_image


def scale_and_save(image, base_filename):
    res_folder = "res"
    # 统一缩放比例
    # scale = 1
    # scaled_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    # # 保存最终的图像

    # image_name = input("> 请输入您想要保存的与处理图片名：")
    # 获取basefilename所在的文件夹名称
    image_name = os.path.basename(base_filename)
    final_filename = os.path.join(base_filename,f"{image_name}.jpg")
    cv2.imwrite(final_filename, image)
    print(f">\t最终处理后的照片已保存为 {final_filename}")
    # # 展示照片
    # cv2.imshow('Processed Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image, final_filename


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用摄像头拍照并组合照片')
    # 添加基础文件名参数，默认值为 'photo'
    parser.add_argument('--base_filename', type=str, default='',
                        help='保存照片的基础文件名，默认为 ''')
    # 添加最多可拍摄照片数量参数，默认值为 10
    parser.add_argument('--max_photos', type=int, default=24,
                        help='最多可拍摄的照片数量，默认为 24')

    # 解析命令行参数
    args = parser.parse_args()
    base_filename = args.base_filename
    max_photos = args.max_photos

    if max_photos < 16:
        print("需要至少 16 张照片才能进行拼接。")
    else:
        # 询问用户是使用已有照片还是调用摄像头
        choice = input(">\t是否使用已有照片进行拼接？(y/n，默认 n): ")
