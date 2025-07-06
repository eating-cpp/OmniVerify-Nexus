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

"""Wrapper for performing OmniGlue inference, plus (optionally) SP/DINO."""

from typing import Optional

import numpy as np
from src.omniglue import dino_extract
from src.omniglue import superpoint_extract
from src.omniglue import utils
import tensorflow as tf
import os
from PIL import Image

import tensorflow as tf

# 设置日志级别为 2，忽略警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DINO_FEATURE_DIM = 768
MATCH_THRESHOLD = 1e-3


class OmniGlue:
    # TODO(omniglue): class docstring
    def __init__(
        self,
        og_export: str,
        sp_export: Optional[str] = None,
        dino_export: Optional[str] = None,
    ) -> None:
        """
        初始化 OmniGlue 类的实例。

        参数:
        og_export (str): OmniGlue 模型的导出路径，用于加载匹配器模型。
        sp_export (Optional[str]): SuperPoint 模型的导出路径，默认为 None。
                                   如果提供了路径，则会初始化 SuperPoint 特征提取器。
        dino_export (Optional[str]): DINO 模型的导出路径，默认为 None。
                                     如果提供了路径，则会初始化 DINO 特征提取器。
        """
       
        
        # 加载 OmniGlue 匹配器模型
        self.matcher = tf.saved_model.load(og_export)
        # 如果提供了 SuperPoint 模型的导出路径，则初始化 SuperPoint 特征提取器
        if sp_export is not None:
            self.sp_extract = superpoint_extract.SuperPointExtract(sp_export)
        # 如果提供了 DINO 模型的导出路径，则初始化 DINO 特征提取器
        if dino_export is not None:
            self.dino_extract = dino_extract.DINOExtract(dino_export, feature_layer=1)


    def FindFeatures_single_image(self,image1,order):
        height1, width1 = image1.shape[:2]
        
        sp_features1 = self.sp_extract(image1,order,None)
        
        dino_features1 = self.dino_extract(image1,order,None)
        
        
        # print(f"> 正在提取用户输入图像的关键点...")
        dino_descriptors1 = dino_extract.get_dino_descriptors(
                dino_features1,
                tf.convert_to_tensor(sp_features1[0], dtype=tf.float32),
                tf.convert_to_tensor(height1, dtype=tf.int32),
                tf.convert_to_tensor(width1, dtype=tf.int32),
                DINO_FEATURE_DIM,
            )
        
        return height1,width1,sp_features1,dino_descriptors1
        
    def FindMatches_single_image(self,height1,width1,sp_features1,dino_descriptors1,current_file_name):
        
        # height1, width1 = image1.shape[:2]
        
        # sp_features1 = self.sp_extract(image1,order,None)
        
        # dino_features1 = self.dino_extract(image1,order,None)
        
        """

            # TODO:
            1.接受参数current_file_name
            2.根据参数current_file_name找到特定文件夹加载sp_descriptors,sp_keypoints,sp_scores,dino_features
            3.构造dino_descriptors
            4.构造inputs字典，进行推理
            5.返回match_kp0s, match_kp1s, match_confidences
            
        """
        image_folder = os.path.join('./features_data_base', current_file_name, 'image')
        
        
        # 查找该文件夹下的 JPG 图像
        jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
        if jpg_files:
            # 假设只有一个 JPG 图像，取第一个
            image0_fp = os.path.join(image_folder, jpg_files[0])
            image0 = np.array(Image.open(image0_fp).convert("RGB"))
            
        height0, width0 = image0.shape[:2]    
        
        sp_features0=self.sp_LoadFromDatabse(os.path.join('./features_data_base', current_file_name,'sp'))
        dino_features0=self.dino_LoadFromDatabse(current_file_name)
        
        
        print(f"> \t提取数据库图像的关键点...")
        dino_descriptors0=dino_extract.get_dino_descriptors(
            dino_features0,
            tf.convert_to_tensor(sp_features0[0], dtype=tf.float32),
            tf.convert_to_tensor(height0, dtype=tf.int32),
            tf.convert_to_tensor(width0, dtype=tf.int32),
            DINO_FEATURE_DIM,
        )
        print(f"> \t数据库加载完成")
       
       
        inputs = self._construct_inputs(
            width0,  # 图像0宽度
            height0,  # 图像0高度
            width1,  # 图像1宽度
            height1,  # 图像1高度
            sp_features0,  # 图像0的SuperPoint特征
            sp_features1,  # 图像1的SuperPoint特征
            dino_descriptors0,  # 图像0的DINO描述符
            dino_descriptors1,  # 图像1的DINO描述符
        )

        # 调用匹配器模型的默认签名进行推理
        og_outputs = self.matcher.signatures["serving_default"](**inputs)
        # 获取soft assignment矩阵，并去除最后一行和最后一列（通常为填充值）
        soft_assignment = og_outputs["soft_assignment"][:, :-1, :-1]

        # 将soft assignment转换为二进制匹配矩阵
        match_matrix = (
            utils.soft_assignment_to_match_matrix(soft_assignment, MATCH_THRESHOLD)
            .numpy()  # 转换为numpy数组
            .squeeze()  # 去除单维度
        )

        # 过滤掉置信度为0的匹配点
        match_indices = np.argwhere(match_matrix)  # 获取所有匹配的索引
        keep = []  # 用于存储有效的匹配索引
        for i in range(match_indices.shape[0]):
            match = match_indices[i, :]  # 当前匹配的索引对
            # 检查两个关键点的置信度是否都大于0
            if (sp_features0[2][match[0]] > 0.0) and (sp_features1[2][match[1]] > 0.0):
                keep.append(i)  # 如果有效则保留
        match_indices = match_indices[keep]  # 更新为过滤后的匹配索引

        # 将匹配结果格式化为关键点坐标和置信度
        match_kp0s = []  # 图像0的匹配关键点
        match_kp1s = []  # 图像1的匹配关键点
        match_confidences = []  # 匹配置信度
        for match in match_indices:
            # 添加图像0的匹配关键点坐标
            match_kp0s.append(sp_features0[0][match[0], :])
            # 添加图像1的匹配关键点坐标
            match_kp1s.append(sp_features1[0][match[1], :])
            # 添加当前匹配的置信度
            match_confidences.append(soft_assignment[0, match[0], match[1]])

            # # 输出当前匹配的图像1和图像2的坐标
            # print(f"Image1 coordinate: {sp_features0[0][match[0], :]}")
            # print(f"Image2 coordinate: {sp_features1[0][match[1], :]}")

        # 将列表转换为numpy数组
        match_kp0s = np.array(match_kp0s)
        match_kp1s = np.array(match_kp1s)
        match_confidences = np.array(match_confidences)
        # 输出 match_kp0s 和 match_kp1s
        # print("match_kp0s:", match_kp0s)
        # print("match_kp1s:", match_kp1s)

        # 返回匹配结果
        return match_kp0s, match_kp1s, match_confidences
        


    def sp_LoadFromDatabse(self,current_file_name):
        """
        从数据库中加载特定图像的 SuperPoint 特征。
        参数:
        current_file_name (str): 当前图像的文件名。
        返回:
        tuple: 包含三个 numpy 数组的元组，分别表示：
            - sp_keypoints: 加载的 SuperPoint 关键点坐标，形状为 (
                
            )
        """
        
         # 获取文件所在的目录
        directory = os.path.dirname(current_file_name)
        
        # 加载关键点
        keypoints = np.loadtxt(os.path.join(directory,'sp', 'sp_keypoints.txt'))
        
        # 加载描述符
        descriptors = np.loadtxt(os.path.join(directory,'sp', 'sp_descriptors.txt'))
        
        # 加载分数
        scores = np.loadtxt(os.path.join(directory,'sp', 'sp_scores.txt'))
        
        return (keypoints, descriptors, scores)
    
    
    def dino_LoadFromDatabse(self,current_file_name):
        
    
        """
        # TODO:
        1.接受参数current_file_name
        2.根据参数current_file_name找到特定文件夹加载dino_features
        3.返回dino_features
        """
        
        features = np.loadtxt(os.path.join('./features_data_base', current_file_name,'dino','dino_features.txt'))
        # print(features)
        return features
        

    def SaveFeatures(self, image0: np.ndarray, order, file_name):
        
        # 计算图像的高度和宽度
        height0, width0 = image0.shape[:2]
        # 提取 SuperPoint 特征
        self.sp_extract(image0, order, file_name)
        print(f">\t正在提取用户输入图像的sp关键点...")
        # 提取 DINO 特征
        self.dino_extract(image0, order, file_name)
        print(f">\t正在提取用户输入图像的dino特征...")

        # 创建保存图像信息的目录
        image_save_dir = f"./features_data_base/{file_name}/image"
        os.makedirs(image_save_dir, exist_ok=True)

        # 保存高度和宽度到 txt 文件
        image_info_path = os.path.join(image_save_dir, "image_info.txt")
        with open(image_info_path, 'w') as f:
            f.write(f"height: {height0}\n")
            f.write(f"width: {width0}\n")
        
    
    def FindMatches_two_images(self, image0: np.ndarray, image1: np.ndarray,order,file_name):
        """在两幅图像之间寻找匹配点。

        该方法通过以下步骤在两幅图像之间寻找匹配点：
        1. 提取两幅图像的特征和描述符
        2. 构建模型输入
        3. 运行匹配器模型
        4. 过滤和格式化匹配结果

        参数:
            image0 (np.ndarray): 第一幅输入图像，应为numpy数组格式
            image1 (np.ndarray): 第二幅输入图像，应为numpy数组格式

        返回:
            tuple: 包含三个numpy数组的元组，分别表示：
                - match_kp0s: 第一幅图像中的匹配关键点坐标，形状为(N, 2)
                - match_kp1s: 第二幅图像中的匹配关键点坐标，形状为(N, 2)
                - match_confidences: 每个匹配点的置信度，形状为(N,)
        """
        # 获取图像的高度和宽度，用于后续特征提取和描述符计算
        height0, width0 = image0.shape[:2]
        height1, width1 = image1.shape[:2]
        # else: height1, width1 = None, None

        # 使用SuperPoint提取器获取图像0的特征，包括关键点、描述符和置信度
        sp_features0 = self.sp_extract(image0,order,file_name)
        # 考虑输出关键点
        # print(f">:superpoint of image0:{sp_features0[0]}")

        # 使用SuperPoint提取器获取图像1的特征
        if order!=2: 
            sp_features1 = self.sp_extract(image1,order,file_name)
        # 考虑输出关键点
        # print(f"image2_superpoint:{sp_features1[0]}")

        # 使用DINO提取器获取图像0的深度特征
        dino_features0 = self.dino_extract(image0,order,file_name)# 对应dino_extract中的def extract（）
        # print(f"image0_dino:{dino_features0}")
        # save to txt
        # np.savetxt('./result/dino_features0.txt', dino_features0)

        # 使用DINO提取器获取图像1的深度特征
        if order!=2: dino_features1 = self.dino_extract(image1,order,file_name)
        # dino_features1 = np.loadtxt(os.path.join('./features_data_base', 'F1_demo1','dino','dino_features.txt'))
        # print(f"image1_dino:{dino_features1}")
        # save to txt
        # np.savetxt('./result/dino_features1.txt', dino_features1)

        # 根据DINO特征和SuperPoint关键点计算图像0的DINO描述符
        dino_descriptors0 = dino_extract.get_dino_descriptors(
            dino_features0,  # DINO特征
            tf.convert_to_tensor(sp_features0[0], dtype=tf.float32),  # 关键点坐标
            tf.convert_to_tensor(height0, dtype=tf.int32),  # 图像高度
            tf.convert_to_tensor(width0, dtype=tf.int32),  # 图像宽度
            DINO_FEATURE_DIM,  # DINO特征维度 等于768
        )
        # 计算图像1的DINO描述符
        if order!=2: 
            dino_descriptors1 = dino_extract.get_dino_descriptors(
                dino_features1,
                tf.convert_to_tensor(sp_features1[0], dtype=tf.float32),
                tf.convert_to_tensor(height1, dtype=tf.int32),
                tf.convert_to_tensor(width1, dtype=tf.int32),
                DINO_FEATURE_DIM,
            )
        

        # 构建匹配器模型所需的输入字典
        inputs = self._construct_inputs(
            width0,  # 图像0宽度
            height0,  # 图像0高度
            width1,  # 图像1宽度
            height1,  # 图像1高度
            sp_features0,  # 图像0的SuperPoint特征
            sp_features1,  # 图像1的SuperPoint特征
            dino_descriptors0,  # 图像0的DINO描述符
            dino_descriptors1,  # 图像1的DINO描述符
        )
        # print(f"输入字典：")
        # print(f"- 图像0宽度:")
        # print(f"{width0}")
        # print(f"- 图像0高度: {height0}")
        # print(f"{height0}")
        # print(f"- 图像1宽度: ")
        # print(f"{width1}")
        # print(f"- 图像1高度: ")
        # print(f"{height1}")
        # print(f"- 图像0的SuperPoint特征: ")
        # print(f"{sp_features0}")
        # print(f"- 图像1的SuperPoint特征: ")
        # print(f"{sp_features1}")
        # print(f"- 图像0的DINO描述符: ")
        # print(f"{dino_descriptors0}")
        # print(f"- 图像1的DINO描述符: ")
        # print(f"{dino_descriptors1}")

        # 调用匹配器模型的默认签名进行推理
        og_outputs = self.matcher.signatures["serving_default"](**inputs)
        # 获取soft assignment矩阵，并去除最后一行和最后一列（通常为填充值）
        soft_assignment = og_outputs["soft_assignment"][:, :-1, :-1]

        # 将soft assignment转换为二进制匹配矩阵
        match_matrix = (
            utils.soft_assignment_to_match_matrix(soft_assignment, MATCH_THRESHOLD)
            .numpy()  # 转换为numpy数组
            .squeeze()  # 去除单维度
        )

        # 过滤掉置信度为0的匹配点
        match_indices = np.argwhere(match_matrix)  # 获取所有匹配的索引
        keep = []  # 用于存储有效的匹配索引
        for i in range(match_indices.shape[0]):
            match = match_indices[i, :]  # 当前匹配的索引对
            # 检查两个关键点的置信度是否都大于0
            if (sp_features0[2][match[0]] > 0.0) and (sp_features1[2][match[1]] > 0.0):
                keep.append(i)  # 如果有效则保留
        match_indices = match_indices[keep]  # 更新为过滤后的匹配索引

        # 将匹配结果格式化为关键点坐标和置信度
        match_kp0s = []  # 图像0的匹配关键点
        match_kp1s = []  # 图像1的匹配关键点
        match_confidences = []  # 匹配置信度
        for match in match_indices:
            # 添加图像0的匹配关键点坐标
            match_kp0s.append(sp_features0[0][match[0], :])
            # 添加图像1的匹配关键点坐标
            match_kp1s.append(sp_features1[0][match[1], :])
            # 添加当前匹配的置信度
            match_confidences.append(soft_assignment[0, match[0], match[1]])

            # # 输出当前匹配的图像1和图像2的坐标
            # print(f"Image1 coordinate: {sp_features0[0][match[0], :]}")
            # print(f"Image2 coordinate: {sp_features1[0][match[1], :]}")

        # 将列表转换为numpy数组
        match_kp0s = np.array(match_kp0s)
        match_kp1s = np.array(match_kp1s)
        match_confidences = np.array(match_confidences)
        # 输出 match_kp0s 和 match_kp1s
        # print("match_kp0s:", match_kp0s)
        # print("match_kp1s:", match_kp1s)

        # 返回匹配结果
        return match_kp0s, match_kp1s, match_confidences

    ### Private methods ###

    def _construct_inputs(
        self,
        width0,
        height0,
        width1,
        height1,
        sp_features0,
        sp_features1,
        dino_descriptors0,
        dino_descriptors1,
        # order
    ):
        """构建OmniGlue模型所需的输入张量字典。

        参数说明：
        width0, height0: 图像0的宽度和高度（整数）
        width1, height1: 图像1的宽度和高度（整数）
        sp_features0: 图像0的SuperPoint特征元组，包含：
            [0] -> 关键点坐标数组（形状：[N, 2]）
            [1] -> 描述符数组（形状：[N, 256]）
            [2] -> 关键点置信度分数数组（形状：[N]）
        sp_features1: 图像1的SuperPoint特征（结构同上）
        dino_descriptors0: 图像0的DINO描述符（形状：[N, 768]）
        dino_descriptors1: 图像1的DINO描述符（形状：[M, 768]）

        返回值：
        包含以下键的字典：
            keypoints0/keypoints1: 关键点坐标（形状：[1, N, 2]）
            descriptors0/descriptors1: SuperPoint描述符（形状：[1, N, 256]）
            scores0/scores1: 关键点置信度（形状：[1, 1, N, 1]）
            descriptors0_dino/descriptors1_dino: DINO描述符（形状：[1, N, 768]）
            width0/width1: 图像宽度（形状：[1]）
            height0/height1: 图像高度（形状：[1]）
        """
        inputs = {
            # SuperPoint关键点坐标（添加批次维度）
            "keypoints0": tf.convert_to_tensor(
                np.expand_dims(sp_features0[0], axis=0),  # 从[N,2]变为[1,N,2]
                dtype=tf.float32,
            ),
            "keypoints1": tf.convert_to_tensor(
                np.expand_dims(sp_features1[0], axis=0), dtype=tf.float32
            ),
            # SuperPoint描述符（添加批次维度）
            "descriptors0": tf.convert_to_tensor(
                np.expand_dims(sp_features0[1], axis=0),  # 从[N,256]变为[1,N,256]
                dtype=tf.float32,
            ),
            "descriptors1": tf.convert_to_tensor(
                np.expand_dims(sp_features1[1], axis=0), dtype=tf.float32
            ),
            # SuperPoint置信度分数（添加批次和通道维度）
            "scores0": tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features0[2], axis=0), axis=-1),  # 从[N]变为[1,1,N,1]
                dtype=tf.float32,
            ),
            "scores1": tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features1[2], axis=0), axis=-1),
                dtype=tf.float32,
            ),
            # DINO描述符（添加批次维度）
            "descriptors0_dino": tf.expand_dims(dino_descriptors0, axis=0),  # 从[N,768]变为[1,N,768]
            "descriptors1_dino": tf.expand_dims(dino_descriptors1, axis=0),
            # 图像尺寸参数（添加批次维度）
            "width0": tf.convert_to_tensor(
                np.expand_dims(width0, axis=0),  # 标量变为[1]
                dtype=tf.int32,
            ),
            "width1": tf.convert_to_tensor(
                np.expand_dims(width1, axis=0), dtype=tf.int32
            ),
            "height0": tf.convert_to_tensor(
                np.expand_dims(height0, axis=0), dtype=tf.int32
            ),
            "height1": tf.convert_to_tensor(
                np.expand_dims(height1, axis=0), dtype=tf.int32
            ),
        }
        return inputs