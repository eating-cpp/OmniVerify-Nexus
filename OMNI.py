#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import src.omniglue as omniglue
from src.omniglue import utils
from PIL import Image
import cv2
from src.omniglue.pre_process import combine_photos, take_photo

plt.rcParams['font.family'] = 'SimHei'  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# import tensorflow as tf
import tensorflow as tf
 # 导入 ConfidenceAnalysis 类
from src.omniglue.confidence_analysis import ConfidenceAnalysis


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 设置日志级别为 2，忽略警告信息
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
    主函数，负责处理命令行参数并调用其他函数完成图像匹配任务。

    参数:
        argv (list): 命令行参数列表，包括脚本名称和两个图像文件路径。
        order (int): 处理方式的指示，1 表示从数据库加载特征进行匹配，2 表示从零开始提取特征并匹配,3 表示录入图片数据
"""

print("> 加载 OmniGlue（及其子模块：SuperPoint 和 DINOv2）...")
start = time.time()
og = omniglue.OmniGlue(
    og_export="./models/og_export",  # OmniGlue 模型导出路径
    sp_export="./models/sp_v6",  # SuperPoint 模型导出路径
            dino_export="./models/dinov2_vitb14_pretrain.pth",  # DINOv2 模型预训练权重路径
)
print(f"> \t耗时 {time.time() - start} 秒。")

print()
print("> 欢迎使用OMNI.py!")
print()
print()
print(f"处理方式：1 从数据库加载特征进行匹配; 2 录入图片数据; 3 OMNIGLUE基本演示; Exit退出")

while True:
    user_input=None
    print()
    print()
    user_input=(input(">:请输入处理方式:"))
    print()

        # order=int(argv[1])
    if user_input=='Exit':
        break
    elif user_input=='1' or user_input=='2' or user_input=='3':
        order=int(user_input)
    else:
        print("> 输入格式错误，请按照 '1' 或 '2' 或 '3' 或 'Exit' 的格式输入。")
        continue
        
    if order==1:

        print()
        temp_order=int(input("> 您是否有已经预处理过的图片？  1. 我有原图但未处理 2.我有预处理过的图片 3.我没有图片，需要拍照"))

        # print("> 从数据库中加载特征以进行比对...")

        max_matches=0
        max_match_kp0=None
        max_match_kp1=None
        max_match_confidences=None
        
        if temp_order==2:
            print("> 请输入需要验证的图像路径：   ")
            temp_input=input()
            temp_input=temp_input.split()
            
            
            if len(temp_input) != 1:
                print("> 输入格式错误，请按照 '<image_path>' 的格式输入。")
                continue
            
            if not os.path.exists(temp_input[0]) or not os.path.isfile(temp_input[0]):
                print(f"> 图像文件路径 '{temp_input[0]}' 不存在或不是文件。")
                continue
            
            # confidences=float(temp_input[0])
            
            image1_fp=temp_input[0]    
            image1 = Image.open(image1_fp).convert("RGB")
        elif temp_order==1:
            max_photos=12
            input_folder=input("> 请输入您想要预处理的图片所在文件夹：")
            # res_folder = "res"
            all_files_exist = True
            folder_name=os.path.join('./res',input_folder)

            if not os.path.exists(folder_name):
                print(f"> 文件夹 {folder_name} 不存在。")
                continue
            existing_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name)]  
            for i in range(1, max_photos + 1):
                target_filename = os.path.join(folder_name,f"{i}.jpg")
                if target_filename not in existing_files:
                    all_files_exist = False
                    print(f"缺少照片文件 {target_filename}，将调用摄像头拍照。")
                    break


            if all_files_exist:
                image1,final_file_name=combine_photos(folder_name, max_photos)

        elif temp_order==3:
            print()
            # from src.omniglue.pre_process import take_photo

            # 调用 take_photo 函数
            base_filename=input("> 请输入您想要将图片保存的文件夹名称：")
            base_filename=os.path.join('./res/',base_filename)
            if not os.path.exists(base_filename):
                os.makedirs(base_filename, exist_ok=True)
            image1,final_file_name=take_photo(base_filename,12)
            
        else:
            print("> 输入格式错误，请按照 '1' 或 '2' 或 '3' 的格式输入。")
            continue
        
        print()
        print("> 加载图像...")
        image1 = np.array(image1)  # 将第二张图像转换为 RGB 格式并加载为 NumPy 数组

        print(f"> 开始提取特征...")
        height1,width1,sp_features1,dino_descriptors1=og.FindFeatures_single_image(image1,order)
        print(f"> 提取完成")

       
        
        
        """
        # TODO:
        1.遍历数据库，统计总共个有多少个文件夹
        2.从第一个文件开始遍历，传参进入FindMatches_single_image
        3.得到match_kp0, match_kp1, match_confidences，进行推理
        4.每次推理完后会保存当前加载的图像和用户需要验证的图像的匹配点数量，存入一个计数器中，该计数器只保留最多的匹配点：将match_kp0, match_kp1, match_confidences保存到一个临时变量中；一个新的临时变量用来存储当前所在文件夹名称
        5.遍历完以后，根据最大匹配点数量对应的match_kp0, match_kp1, match_confidences，和保存好的文件名称，进行可视化
        
        """
        
        # 统计多少个文件夹
        features_data_base_path = './features_data_base/'
        if os.path.exists(features_data_base_path):
            subfolders = [f for f in os.listdir(features_data_base_path) if os.path.isdir(os.path.join(features_data_base_path, f))]
            folder_count = len(subfolders)
            print(f">\t在 {features_data_base_path} 下有 {folder_count} 个子文件夹。")
        else:
            print(f"> {features_data_base_path} 路径不存在。")
            continue
            
        

        total_slices=1000
        confidences_levels = np.linspace(0, 1, total_slices)
        max_match_file_name_list = [None] * total_slices
        max_num_matches_list = [0] * total_slices
        mean_confidences_list = [0] * folder_count

        # 用于存储每个文件的置信度分布
        all_confidence_distributions = []
        all_subfolder_names = []  # 用于存储所有子文件夹的名称

        # ... existing code ...
        for i in range(folder_count):
            print(f" ")
            subfolder_name = subfolders[i]
            subfolder_path = os.path.join(features_data_base_path, subfolder_name)
            print(f"> 当前正在处理子文件夹: {subfolder_path}")

            current_file_name = subfolder_name

            match_kp0, match_kp1, match_confidences = og.FindMatches_single_image(height1, width1, sp_features1, dino_descriptors1, current_file_name)  # 调用 OmniGlue 的 FindMatches 方法进行匹配
            num_matches = match_kp0.shape[0]  # 获取匹配点的数量

            # 计算当前子文件夹的置信度分布
            distribution = []
            for level in confidences_levels:
                num_filtered_matches = np.sum(match_confidences > level)
                distribution.append(num_filtered_matches)

            # 收集当前子文件夹的置信度分布数据
            all_confidence_distributions.append(distribution)
            all_subfolder_names.append(subfolder_name)
            # mean_confidences_list[i]=np.mean(confidences_levels)
            

            # 绘制所有子文件夹的置信度分布在一张折线图上
            plt.figure(figsize=(10, 6))
            for i, distribution in enumerate(all_confidence_distributions):
                # 修改这里，直接使用置信度水平作为 x 轴
                plt.plot(confidences_levels, distribution, label=all_subfolder_names[i])
            plt.xlabel('置信度')
            plt.ylabel('匹配点数量')
            plt.title('所有子文件夹的置信度分布')
            plt.legend()
            plt.grid(True)
            output_path = f"./result/confidence_distribution_all.png"
            plt.savefig(output_path)
            plt.close()

            # print(f"> 已将所有子文件夹的置信度分布折线图保存到 {output_path}")

            '''
            TODO:
                1.保存的每个文件的置信度分布
                2.将每个文件的置信度送入一个机器学习模型，挑出最显著的高于其他文件的文件
                3.输出他显著的高于其他文件的程度（类似于相关系数？方差分析？）
                4.统计大量的文件，设定合理的阈值。
                5.以后找到显著度最高的文件，计算显著的程度，如果高于了某个阈值，那么就认定为匹配成功，否则为匹配失败。
            '''

           

        # 创建 ConfidenceAnalysis 类的实例
        analysis = ConfidenceAnalysis(all_confidence_distributions, all_subfolder_names)

        # 调用 integrate_and_sort 方法进行积分、排序和归一化操作
        analysis.integrate_and_sort()
            
            
            # for j, level in enumerate(confidences_levels):
            #     num_filtered_matches = np.sum(match_confidences > level)
            #     if num_filtered_matches > max_num_matches_list[j]:
            #         max_num_matches_list[j] = num_filtered_matches
            #         max_match_file_name_list[j] = current_file_name
            #     print(f"> \t当前文件在置信度 {level} 下匹配点数量为：{num_filtered_matches}/{num_matches}")
                            
            # print(max_match_file_name_list)
                            # print(f"> \t{num_filtered_matches}/{num_matches} 个匹配点的置信度高于阈值 {match_threshold}")

                    # 统计max_match_file_name_list中每个元素出现的次数，计算每个文件名出现的概率作为并输出，并挑选概率最大的文件做后续工作
            # for j in range(folder_count):
            #     print(f">\t文件{all_subfolder_names[j]}的平均值：{mean_confidences_list[j]}")   
            
            # print(mean_confidences_list)
        # # from collections import Counter

        # # # 过滤掉列表中的 None 值
        # # non_none_files = [file for file in max_match_file_name_list if file is not None]

        # # # 统计每个文件名出现的次数
        # # file_count = Counter(non_none_files)

        # # # 计算非 None 文件的总次数
        # # total_count = len(non_none_files)

        # # # 找到出现次数最多的文件及其出现次数
        # # if file_count:
        # #     most_common_file, most_common_count = file_count.most_common(1)[0]

        # #     # 计算出现次数最多的文件出现的次数在非 None 文件总次数中的占比
        # #     ratio = most_common_count / total_count
        # #     # ratio=most_common_count/100

        # #     # 输出结果
        # #     print(f"> 出现次数最多的文件是 {most_common_file}，出现次数为 {most_common_count}")
        # #     print(f"> 该文件出现次数在非 None 文件总次数中的占比为: {ratio * 100:.2f}%")
        # # else:
        # #     print("> 列表中没有非 None 的文件。")

        
        
        
        
        # total_slices = 100
        # confidences_levels = np.linspace(0, 1, total_slices)
        # max_match_file_name_list = [None] * total_slices
        # max_num_matches_list = [0] * total_slices
        # main_confidences=[0]*folder_count
        # temp_folder_name=[None]*folder_count

        # # 用于存储每个文件的置信度分布
        # all_confidence_distributions = []
        # all_subfolder_names = []  # 用于存储所有子文件夹的名称

        # for i in range(folder_count):
        #     print(f" ")
        #     subfolder_name = subfolders[i]
        #     subfolder_path = os.path.join(features_data_base_path, subfolder_name)
        #     print(f"> 当前正在处理子文件夹: {subfolder_path}")

        #     current_file_name = subfolder_name

        #     match_kp0, match_kp1, match_confidences = og.FindMatches_single_image(height1, width1, sp_features1, dino_descriptors1, current_file_name)  # 调用 OmniGlue 的 FindMatches 方法进行匹配
        #     num_matches = match_kp0.shape[0]  # 获取匹配点的数量
        
        #     # 输出均值
        #     mean_confidence = np.mean(match_confidences)
        #     main_confidences[i]=mean_confidence
        #     temp_folder_name[i]=(subfolder_name)
        #     # print(f"> 匹配点的置信度均值为: {mean_confidence}")
        

        #     # 计算当前子文件夹的置信度分布
        #     distribution = []
        #     for i in range(len(confidences_levels) - 1):
        #         lower_bound = confidences_levels[i]
        #         upper_bound = confidences_levels[i + 1]
        #         # 计算置信度在区间 [lower_bound, upper_bound) 内的特征点数量
        #         num_filtered_matches = np.sum((match_confidences >= lower_bound) & (match_confidences < upper_bound))
        #         distribution.append(num_filtered_matches)

        #     # 收集当前子文件夹的置信度分布数据
        #     all_confidence_distributions.append(distribution)
        #     all_subfolder_names.append(subfolder_name)

        #     # for j in range(len(confidences_levels) - 1):
        #     #     lower_bound = confidences_levels[j]
        #     #     upper_bound = confidences_levels[j + 1]
        #     #     num_filtered_matches = distribution[j]
        #     #     if num_filtered_matches > max_num_matches_list[j]:
        #     #         max_num_matches_list[j] = num_filtered_matches
        #     #         max_match_file_name_list[j] = current_file_name
        #     #     # print(f"> \t当前文件在置信度区间 [{lower_bound:.2f}, {upper_bound:.2f}) 下匹配点数量为：{num_filtered_matches}/{num_matches}")

        #     # # print(max_match_file_name_list)
            
        # for i in range(folder_count):
        #     print(f"> {temp_folder_name[i]}的置信度均值为：{main_confidences[i]}")
            
            
            
        
            
            
        

        # # 计算每个区间的中点
        # confidence_midpoints = (confidences_levels[:-1] + confidences_levels[1:]) / 2

        # # 绘制所有子文件夹的置信度分布在一张折线图上
        # plt.figure(figsize=(10, 6))
        # for i, distribution in enumerate(all_confidence_distributions):
        #     # 使用置信度区间中点作为 x 轴
        #     plt.plot(confidence_midpoints, distribution, label=all_subfolder_names[i])
        # plt.xlabel('置信度')
        # plt.ylabel('匹配点数量')
        # plt.title('所有子文件夹的置信度分布')
        # plt.legend()
        # plt.grid(True)
        # output_path = f"./result/confidence_distribution_all.png"
        # plt.savefig(output_path)
        # plt.close()
        # print(f"> 已将所有子文件夹的置信度分布折线图保存到 {output_path}")

        # # from collections import Counter

        # # # 过滤掉列表中的 None 值
        # # non_none_files = [file for file in max_match_file_name_list if file is not None]

        # # # 统计每个文件名出现的次数
        # # file_count = Counter(non_none_files)

        # # # 计算非 None 文件的总次数
        # # total_count = len(non_none_files)

        # # # 找到出现次数最多的文件及其出现次数
        # # if file_count:
        # #     most_common_file, most_common_count = file_count.most_common(1)[0]

        # #     # 计算出现次数最多的文件出现的次数在非 None 文件总次数中的占比
        # #     ratio = most_common_count / total_count

        # #     # 输出结果
        # #     print(f"> 出现次数最多的文件是 {most_common_file}，出现次数为 {most_common_count}")
        # #     print(f"> 该文件出现次数在非 None 文件总次数中的占比为: {ratio * 100:.2f}%")
        # # else:
        # #     print("> 列表中没有非 None 的文件。")







        # specific_confidences = [0.5, 0.3, 0.2, 0.08, 0.05, 0.02]
        # for confidences in specific_confidences:
        #     # 找到对应置信度的索引
        #     if confidences in confidences_levels:
        #         j = confidences_levels.tolist().index(confidences)
        #     else:
        #         print(f"> 未找到置信度 {confidences} 的数据，跳过。")
        #         continue

        #     current_file_name = max_match_file_name_list[j]

        #     if current_file_name != max_probability_file:
        #         continue

        #     image_folder = os.path.join('./features_data_base', current_file_name, 'image')
        #     # 查找该文件夹下的 JPG 图像
        #     jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
        #     if jpg_files:
        #         # 假设只有一个 JPG 图像，取第一个
        #         image0_fp = os.path.join(image_folder, jpg_files[0])
        #         image0 = np.array(Image.open(image0_fp).convert("RGB"))

        #         # 分别绘制六个置信度条件下的图片
        #         max_matches = max_matches_list[j]
        #         max_match_kp0 = max_match_kp0_list[j]
        #         max_match_kp1 = max_match_kp1_list[j]

        #         print(f"> {max_matches}/{max_num_matches} 个匹配点的置信度高于阈值 {confidences}")

        #         viz = utils.visualize_matches(
        #             image0,  # image0: (H, W, 3) 数组，包含 image0 的内容。
        #             image1,  # image1: (H, W, 3) 数组，包含 image1 的内容。
        #             max_match_kp0,  # kp0: (N, 2) 数组，其中每一行代表 image0 中关键点的 (x, y) 坐标。
        #             max_match_kp1,  # kp1: (M, 2) 数组，其中每一行代表 image1 中关键点的 (x, y) 坐标。
        #             np.eye(max_matches),  # 创建一个单位矩阵作为匹配关系矩阵
        #             match_labels=None,
        #             show_keypoints=True,  # 显示关键点
        #             highlight_unmatched=True,  # 高亮未匹配的关键点
        #             title=f"{max_matches} matches at confidence {confidences}",  # 图像标题
        #             line_width=2,  # 匹配线宽
        #         )
        #         output_path = f"./result/OMNIoutput_confidence_{confidences}.png"
        #         plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")  # 创建一个新的图形窗口
        #         plt.axis("on")  # 关闭坐标轴
        #         plt.imshow(viz)  # 显示可视化结果
        #         plt.imsave(output_path, viz)  # 保存可视化结果到文件
        #         print(f"> 已将置信度 {confidences} 的可视化结果保存到 {output_path}")
        #     # else:
        #         # print(f">\t在 {image_folder} 中未找到 JPG 图像。")

        print()
            
            
    elif order==2:
        print()
        # print("> 录入图片特征数据到数据库...")
        print(f"录入方式：1 给定路径录入已经经过处理的图片; 2 调用摄像头录入; 3 从指定路径获取原图处理后录入")
        print()
        temp_order=int(input("> 请输入录入方式："))
        
        
        if temp_order==1:

            im_fp=input("> 请输入需要加载的图像路径：   ")
            # im_fp = input()
            file_name = input("> 请输入您想要保存的文件名：   ")

            if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
                print(f"图像文件路径 '{im_fp}' 不存在或不是文件。")
                continue

            image0_fp = im_fp
            image0 = np.array(Image.open(image0_fp).convert("RGB"))

           
            # 执行推理
            print(f"")
            print("> 查找匹配点...")
            start = time.time()
            og.SaveFeatures(image0, order, file_name)

            # 创建保存图像的目录
            image_save_dir = os.path.join('./features_data_base', file_name, 'image')
            os.makedirs(image_save_dir, exist_ok=True)

            # 保存图像
            image_save_path = os.path.join(image_save_dir, os.path.basename(image0_fp))
            Image.fromarray(image0).save(image_save_path)

            print(f"> \t耗时 {time.time() - start} 秒。")
            print(f"> 已将特征数据保存到数据库中:./features_data_base/{file_name}")
            print(f"> 已将图像保存到: {image_save_path}")
        elif temp_order==2:
            
            print()
            # from src.omniglue.pre_process import take_photo

            # 调用 take_photo 函数
            base_filename=input("> 请输入您想要将图片保存的文件夹名称：")
            base_filename=os.path.join('./res/',base_filename)
            if not os.path.exists(base_filename):
                os.makedirs(base_filename, exist_ok=True)
            scaled_image,final_file_name=take_photo(base_filename,24)
            # if scaled_image==-1 or final_file_name==-1:
                # print("> 拍照失败，请检查摄像头是否连接正常。")
                # continue

            file_name = input("> 请输入您想要保存的文件名：   ")
            image0=np.array(scaled_image)

             # 执行推理
            print(f"")
            print("> 查找匹配点...")
            start = time.time()
            og.SaveFeatures(image0, order, file_name)

            # 创建保存图像的目录
            image_save_dir = os.path.join('./features_data_base', file_name, 'image')
            os.makedirs(image_save_dir, exist_ok=True)

            # 保存图像
            image0_fp = final_file_name
            image_save_path = os.path.join(image_save_dir, os.path.basename(image0_fp))
            Image.fromarray(image0).save(image_save_path)

            print(f"> \t耗时 {time.time() - start} 秒。")
            print(f"> 已将特征数据保存到数据库中:./features_data_base/{file_name}")
            print(f"> 已将图像保存到: {image_save_path}")

        elif temp_order==3:
            
            max_photos=12
            input_folder=input("> 请输入您想要预处理的图片所在文件夹：")
            # res_folder = "res"
            all_files_exist = True
            folder_name=os.path.join('./res',input_folder)

            if not os.path.exists(folder_name):
                print(f"> 文件夹 {folder_name} 不存在。")
                continue
            existing_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name)]  
            for i in range(1, max_photos + 1):
                target_filename = os.path.join(folder_name,f"{i}.jpg")
                if target_filename not in existing_files:
                    all_files_exist = False
                    print(f"缺少照片文件 {target_filename}，将调用摄像头拍照。")
                    break


            if all_files_exist:
                tempimage,final_file_name=combine_photos(folder_name, max_photos)
                
                # # 展示scaled_image
                # cv2.imshow("Combined Image", tempimage)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(f"> 已将照片组合到 {final_file_name}")
                
               
                # # 询问用户是否继续
                # user_input = input("> 请输入 'y' 继续，'n' 退出：")
                # if user_input.lower() != 'y':
                #     print("> 已退出。")
                #     continue
                
                
                image0_fp = final_file_name

                file_name = input("> 请输入您想要保存的文件名：   ")
                image0=np.array(tempimage)

                 # 执行推理
                print(f"")
                print("> 查找匹配点...")
                start = time.time()
                og.SaveFeatures(image0, order, file_name)

                # 创建保存图像的目录
                image_save_dir = os.path.join('./features_data_base', file_name, 'image')
                os.makedirs(image_save_dir, exist_ok=True)

                # 保存图像
                image0_fp = final_file_name
                image_save_path = os.path.join(image_save_dir, os.path.basename(image0_fp))
               
                # 保存tempimage
                cv2.imwrite(image_save_path, tempimage)
                
                

                print(f"> \t耗时 {time.time() - start} 秒。")
                print(f"> 已将特征数据保存到数据库中:./features_data_base/{file_name}")
                print(f"> 已将图像保存到: {image_save_path}")
        else:
            print("> 输入格式错误，请按照 '1' 或 '2' 或 '3' 的格式输入。")
            continue  



    elif order==3:
        file_name=None
            # # argv=sys.argv
            # print(argv)
            # print(f">:从零开始提取特征并匹配:")
        temp_input = input("> 请输入图像路径和置信度:   ")
            # 格式：image0_path image1_path confidence
            # 按空格分割输入字符串
        inputs = temp_input.split()
        if len(inputs) != 3:
            print("输入格式错误，请按照 'image0_path image1_path confidence' 的格式输入。")
            continue
            
            # 获取图像文件路径
        image0_fp = inputs[0]
        image1_fp = inputs[1]
        confidence = float(inputs[2])

            # 验证图像文件路径是否存在且为文件
        for im_fp in [image0_fp, image1_fp]:
            if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
                print(f"图像文件路径 '{im_fp}' 不存在或不是文件。")
                continue

            # 加载图像
        print("> 加载图像...")
        image0 = np.array(Image.open(image0_fp).convert("RGB"))  # 将第一张图像转换为 RGB 格式并加载为 NumPy 数组
        image1 = np.array(Image.open(image1_fp).convert("RGB"))  # 将第二张图像转换为 RGB 格式并加载为 NumPy 数组

           

            # 执行推理
        print("> 查找匹配点...")
        start = time.time()
        match_kp0, match_kp1, match_confidences = og.FindMatches_two_images(image0, image1,order,file_name)  # 调用 OmniGlue 的 FindMatches 方法进行匹配
        num_matches = match_kp0.shape[0]  # 获取匹配点的数量
        print(f"> \t找到 {num_matches} 个匹配点。")
        print(f"> \t耗时 {time.time() - start} 秒。")

            # 根据置信度过滤匹配点
        print("> 过滤匹配点...")
        match_threshold = confidence  # 置信度阈值，范围为 [0.0, 1.0)
        keep_idx = []  # 保存置信度高于阈值的匹配点索引
        for i in range(match_kp0.shape[0]):
            if match_confidences[i] > match_threshold:
                keep_idx.append(i)
        num_filtered_matches = len(keep_idx)  # 过滤后的匹配点数量
        match_kp0 = match_kp0[keep_idx]  # 只保留置信度高于阈值的匹配点
        match_kp1 = match_kp1[keep_idx]
        match_confidences = match_confidences[keep_idx]

            # print(f"> 过滤后的匹配点及其置信度:")
            # for i in range(num_filtered_matches):
            #     print(f"> match_kp0: {match_kp0[i]}, match_kp1: {match_kp1[i]}, confidence: {match_confidences[i]}")

        print(f"> \t{num_filtered_matches}/{num_matches} 个匹配点的置信度高于阈值 {match_threshold}")
        print("> 可视化匹配点...")
        viz = utils.visualize_matches(
            image0,#image0: (H, W, 3) 数组，包含 image0 的内容。
            image1,#image1: (H, W, 3) 数组，包含 image1 的内容。
            match_kp0,#kp0: (N, 2) 数组，其中每一行代表 image0 中关键点的 (x, y) 坐标。
            match_kp1,#kp1: (M, 2) 数组，其中每一行代表 image1 中关键点的 (x, y) 坐标。
            np.eye(num_filtered_matches),  # 创建一个单位矩阵作为匹配关系矩阵
            match_labels=None,
            show_keypoints=True,  # 显示关键点
            highlight_unmatched=True,  # 高亮未匹配的关键点
            title=f"{num_filtered_matches} matches",  # 图像标题
            # dino_descriptor_map0=not None,
            # dino_descriptor_map1=not None,
            line_width=2,  # 匹配线宽
        )
        # print("匹配矩阵:\n", np.eye(num_filtered_matches))  # 打印匹配矩阵
        # plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")  # 创建一个
        output_path = f"./result/OMNIoutput_confidence_{confidence}.png"
        plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")  # 创建一个新的图形窗口
        plt.axis("on")  # 关闭坐标轴
        plt.imshow(viz)  # 显示可视化结果
        plt.imsave(output_path, viz)  # 保存可视化结果到文件
        print(f"> 已将置信度 {confidence} 的可视化结果保存到 {output_path}")