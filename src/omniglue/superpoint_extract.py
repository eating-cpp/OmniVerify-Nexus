# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 若不遵守本许可，不得使用此文件。
# 您可在以下网址获取许可副本：
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则在许可下分发的软件
# 按“原样”分发，
# 无任何形式的明示或暗示保证和条件。
# 请参阅许可以了解管理权限和
# 限制的特定语言。

"""用于执行 SuperPoint 推理的包装器。"""

# 导入 math 模块，但代码中未使用，可考虑移除
import math
from typing import Optional, Tuple

import os
import cv2
import numpy as np
from src.omniglue import utils

import tensorflow as tf



# 导入 TensorFlow 1.x 兼容模块
import tensorflow.compat.v1 as tf1 # type: ignore


# 设置日志级别为 2，忽略警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SuperPointExtract:
    """初始化 SuperPoint 模型并从图像中提取特征的类。

    为了与 SuperPoint 训练和评估配置保持一致，
    请将图像调整为 (320x240) 或 (640x480)。

    属性:
        model_path: 字符串，保存的 SuperPoint TF1 模型权重的文件路径。
    """
    count=0

    def __init__(self, model_path: str):
        # 保存模型路径
        self.model_path = model_path
        # 创建一个新的 TensorFlow 图
        self._graph = tf1.Graph()
        # 创建一个 TensorFlow 会话
        self._sess = tf1.Session(graph=self._graph)
        # 加载保存的模型
        tf1.saved_model.loader.load(
            self._sess, [tf1.saved_model.tag_constants.SERVING], model_path
        )

    def __call__(
        self,
        image,
        order,
        file_name,
        segmentation_mask=None,
        num_features=1024,
        pad_random_features=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 调用 compute 方法进行特征计算
        return self.compute(
            image,
            order,
            file_name,
            segmentation_mask=segmentation_mask,
            num_features=num_features,
            pad_random_features=pad_random_features,
        )

    def compute(
        self,
        image: np.ndarray,
        order,
        file_name,
        segmentation_mask: Optional[np.ndarray] = None,
        num_features: int = 1024,
        pad_random_features: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将图像输入 SuperPoint 模型以提取关键点和特征。

        参数:
            image: (H, W, 3) 的 numpy 数组，解码后的图像字节。
            order: 1代表从数据库加载进行特征匹配； 2表示从零开始特征匹配；
            segmentation_mask: (H, W) 的二进制 numpy 数组或 None。
                如果不为 None，提取的关键点将限制在掩码内。
            num_features: 要提取的最大特征数（或 0 表示保留所有提取的特征）。
            pad_random_features: 如果为 True，则向输出中添加随机采样的关键点，
                使得输出中恰好有 'num_features' 个关键点。
                这些采样关键点的描述符取自网络的描述符图输出，分数设置为 0。
                如果 num_features = 0，则不执行任何操作。

        返回:
            keypoints: (N, 2) 的 numpy 数组，关键点的坐标，以浮点数表示。
            descriptors: (N, 256) 的 numpy 数组，关键点的描述符，以浮点数表示。
            scores: (N, 1) 的 numpy 数组，关键点的置信度值，以浮点数表示。
        """
        if order==1:
            image, keypoint_scale_factors = self._resize_input_image(image)
            if segmentation_mask is not None:
                # 调整分割掩码的大小
                segmentation_mask, _ = self._resize_input_image(
                    segmentation_mask, interpolation=cv2.INTER_NEAREST
                )
            # 确保图像和分割掩码的尺寸匹配
            assert (
                segmentation_mask is None
                or image.shape[:2] == segmentation_mask.shape[:2]
            )

            # 预处理图像并进行前向传播
            image_preprocessed = self._preprocess_image(image)
            # 获取输入图像的张量
            input_image_tensor = self._graph.get_tensor_by_name('superpoint/image:0')
            # 获取非极大值抑制后的概率张量
            output_prob_nms_tensor = self._graph.get_tensor_by_name(
                'superpoint/prob_nms:0'
            )
            # 获取描述符张量
            output_desc_tensors = self._graph.get_tensor_by_name(
                'superpoint/descriptors:0'
            )
            # 运行会话，得到输出
            out = self._sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_image_tensor: np.expand_dims(image_preprocessed, 0)},
            )

            # 处理网络输出
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            if segmentation_mask is not None:
                # 如果有分割掩码，将掩码外的关键点概率置为 0
                keypoint_map = np.where(segmentation_mask, keypoint_map, 0.0)
            # 提取 SuperPoint 输出
            keypoints, descriptors, scores = self._extract_superpoint_output(
                keypoint_map, descriptor_map, num_features, pad_random_features
            )
            

            # 将关键点位置缩放到原始输入图像大小并返回
            keypoints = keypoints / keypoint_scale_factors
            
            # 保存信息到文件
            # if self.count==1: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base//')
            # elif self.count==2: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base')
            # self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base/')
            # 输出提取的关键点、描述符和分数
            # print(f"Extracted {len(keypoints)} keypoints")
            # print(f"Keypoints: {keypoints}")
            # print(f"Descriptors: {descriptors}")
            # print(f"Scores: {scores}")
            return (keypoints, descriptors, scores)
            
        
        elif order==2: # 录入数据库
            image, keypoint_scale_factors = self._resize_input_image(image)
            if segmentation_mask is not None:
                # 调整分割掩码的大小
                segmentation_mask, _ = self._resize_input_image(
                    segmentation_mask, interpolation=cv2.INTER_NEAREST
                )
            # 确保图像和分割掩码的尺寸匹配
            assert (
                segmentation_mask is None
                or image.shape[:2] == segmentation_mask.shape[:2]
            )

            # 预处理图像并进行前向传播
            image_preprocessed = self._preprocess_image(image)
            # 获取输入图像的张量
            input_image_tensor = self._graph.get_tensor_by_name('superpoint/image:0')
            # 获取非极大值抑制后的概率张量
            output_prob_nms_tensor = self._graph.get_tensor_by_name(
                'superpoint/prob_nms:0'
            )
            # 获取描述符张量
            output_desc_tensors = self._graph.get_tensor_by_name(
                'superpoint/descriptors:0'
            )
            # 运行会话，得到输出
            out = self._sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_image_tensor: np.expand_dims(image_preprocessed, 0)},
            )

            # 处理网络输出
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            if segmentation_mask is not None:
                # 如果有分割掩码，将掩码外的关键点概率置为 0
                keypoint_map = np.where(segmentation_mask, keypoint_map, 0.0)
            # 提取 SuperPoint 输出
            keypoints, descriptors, scores = self._extract_superpoint_output(
                keypoint_map, descriptor_map, num_features, pad_random_features
            )
            

            # 将关键点位置缩放到原始输入图像大小并返回
            keypoints = keypoints / keypoint_scale_factors
            
            # 保存信息到文件
            # if self.count==1: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base//')
            # elif self.count==2: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base')
            self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base/'+file_name+'/sp/')
        
            # return (keypoints, descriptors, scores)
        
        elif order==3: # 简单演示功能
            # 调整图像大小，使两个维度都能被 8 整除
            image, keypoint_scale_factors = self._resize_input_image(image)
            if segmentation_mask is not None:
                # 调整分割掩码的大小
                segmentation_mask, _ = self._resize_input_image(
                    segmentation_mask, interpolation=cv2.INTER_NEAREST
                )
            # 确保图像和分割掩码的尺寸匹配
            assert (
                segmentation_mask is None
                or image.shape[:2] == segmentation_mask.shape[:2]
            )

            # 预处理图像并进行前向传播
            image_preprocessed = self._preprocess_image(image)
            # 获取输入图像的张量
            input_image_tensor = self._graph.get_tensor_by_name('superpoint/image:0')
            # 获取非极大值抑制后的概率张量
            output_prob_nms_tensor = self._graph.get_tensor_by_name(
                'superpoint/prob_nms:0'
            )
            # 获取描述符张量
            output_desc_tensors = self._graph.get_tensor_by_name(
                'superpoint/descriptors:0'
            )
            # 运行会话，得到输出
            out = self._sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_image_tensor: np.expand_dims(image_preprocessed, 0)},
            )

            # 处理网络输出
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            if segmentation_mask is not None:
                # 如果有分割掩码，将掩码外的关键点概率置为 0
                keypoint_map = np.where(segmentation_mask, keypoint_map, 0.0)
            # 提取 SuperPoint 输出
            keypoints, descriptors, scores = self._extract_superpoint_output(
                keypoint_map, descriptor_map, num_features, pad_random_features
            )
            

            # 将关键点位置缩放到原始输入图像大小并返回
            keypoints = keypoints / keypoint_scale_factors
            
            # 保存信息到文件
            # if self.count==1: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base//')
            # elif self.count==2: self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base')
            # self.save_superpoint_output(keypoints, descriptors, scores, './features_data_base/')
            # 输出提取的关键点、描述符和分数
            # print(f"Extracted {len(keypoints)} keypoints")
            # print(f"Keypoints: {keypoints}")
            # print(f"Descriptors: {descriptors}")
            # print(f"Scores: {scores}")
            return (keypoints, descriptors, scores)
    # keypoints是一个形状为（N，2）的numpy数组y' array，其中每个元素表示一个关键点的位置（x，y）
    # descriptors是一个形状为（N，256）的numpy数组，其中每个元素表示一个关键点的描述符
    # scores是一个形状为（N，1）的numpy数组，其中每个元素表示一个关键点的置信度值
    
    def save_superpoint_output(self, keypoints, descriptors, scores, filename):
        """
        将 SuperPoint 提取的关键点、描述符和分数保存到 txt 文件中。
        :param keypoints: (N, 2) 的 numpy 数组，关键点的坐标
        :param descriptors: (N, 256) 的 numpy 数组，关键点的描述符
        :param scores: (N, 1) 的 numpy 数组，关键点的置信度值
        :param filename: 保存的文件名
        """
        # self.count+=1
        # 获取文件所在的目录
        directory = os.path.dirname(filename)
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存关键点
        # if self.count==1:
        np.savetxt(os.path.join(directory, './sp_keypoints.txt'), keypoints)
            # print(f">   sp1关键点保存到 './features/image1/sp/sp_keypoints1.txt'")
            
        np.savetxt(os.path.join(directory, './sp_descriptors.txt'), descriptors)
            # print(f">   sp1描述符保存到 {os.path.join(directory, './image1/sp/sp_descriptors1.txt')}")
            
        np.savetxt(os.path.join(directory, './sp_scores.txt'), scores)
            # print(f">   sp1置信度保存到 {os.path.join(directory, './image1/sp/sp_scores1.txt')}")
        # elif self.count==2:
        #     np.savetxt(os.path.join(directory, './image2/sp/sp_keypoints2.txt'), keypoints)
        #     # print(f">   sp2关键点保存到 {os.path.join(directory, './image2/sp/sp_keypoints2.txt')}")
            
        #     np.savetxt(os.path.join(directory, './image2/sp/sp_descriptors2.txt'), descriptors)
        #     # print(f">   sp2描述符保存到 {os.path.join(directory, './image2/sp/sp_descriptors2.txt')}")
            
        #     np.savetxt(os.path.join(directory,'./image2/sp/sp_scores2.txt'), scores)
        #     # print(f">   sp2置信度保存到 {os.path.join(directory, './image2/sp/sp_scores2.txt')}")

    def _resize_input_image(self, image, interpolation=cv2.INTER_LINEAR):
        """调整图像大小，使两个维度都能被 8 整除。"""

        # 计算新的图像维度和每个维度的缩放因子
        new_dim = [-1, -1]
        keypoint_scale_factors = [1.0, 1.0]
        for i in range(2):
            dim_size = image.shape[i]
            mod_eight = dim_size % 8
            if mod_eight < 4:
                # 向下取整到最近的 8 的倍数
                new_dim[i] = dim_size - mod_eight
            elif mod_eight >= 4:
                # 向上取整到最近的 8 的倍数
                new_dim[i] = dim_size + (8 - mod_eight)
            keypoint_scale_factors[i] = (new_dim[i] - 1) / (dim_size - 1)

        # 调整维度顺序，从 (行, 列) 转换为 (x, y)
        new_dim = new_dim[::-1]
        keypoint_scale_factors = keypoint_scale_factors[::-1]
        # 调整图像大小
        image = cv2.resize(image, tuple(new_dim), interpolation=interpolation)
        return image, keypoint_scale_factors

    def _preprocess_image(self, image):
        """将图像转换为灰度图并归一化，以作为模型输入。"""
        # 将图像转换为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 添加一个维度
        image = np.expand_dims(image, 2)
        # 转换为浮点型
        image = image.astype(np.float32)
        # 归一化到 [0, 1]
        image = image / 255.0
        return image

    def _extract_superpoint_output(
        self,
        keypoint_map,
        descriptor_map,
        keep_k_points=512,
        pad_random_features=False,
    ):
        """将原始的 SuperPoint 输出（特征图）转换为 numpy 数组。

        如果 keep_k_points 为 0，则保留所有检测到的关键点。
        否则，按置信度排序并仅保留前 k 个置信度最高的关键点。

        参数:
            keypoint_map: (H, W, 1) 的 numpy 数组，SuperPoint 模型的原始输出置信度值。
            descriptor_map: (H, W, 256) 的 numpy 数组，SuperPoint 模型的原始输出描述符。
            keep_k_points: 要保留的关键点数量（或 0 表示保留所有检测到的关键点）。
            pad_random_features: 如果为 True，则向输出中添加随机采样的关键点，
                使得输出中恰好有 'num_features' 个关键点。
                这些采样关键点的描述符取自网络的描述符图输出，分数设置为 0。
                如果 keep_k_points = 0，则不执行任何操作。

        返回:
            keypoints: (N, 2) 的 numpy 数组，关键点的图像坐标 (x, y)，以浮点数表示。
            descriptors: (N, 256) 的 numpy 数组，关键点的描述符，以浮点数表示。
            scores: (N, 1) 的 numpy 数组，关键点的置信度值，以浮点数表示。
        """

        def _select_k_best(points, k):
            """选择置信度最高的 k 个关键点。"""
            # 按置信度排序
            sorted_prob = points[points[:, 2].argsort(), :]
            start = min(k, points.shape[0])
            # 返回前 k 个关键点的坐标和置信度
            return sorted_prob[-start:, :2], sorted_prob[-start:, 2]

        # 找到置信度大于 0 的关键点
        keypoints = np.where(keypoint_map > 0)
        # 获取这些关键点的置信度
        prob = keypoint_map[keypoints[0], keypoints[1]]
        # 将关键点的坐标和置信度堆叠在一起
        keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

        # 如果 keep_k_points 为 0，则保留所有关键点
        if keep_k_points == 0:
            keep_k_points = keypoints.shape[0]
        # 选择置信度最高的 k 个关键点
        keypoints, scores = _select_k_best(keypoints, keep_k_points)

        # 可选：用随机特征填充（置信度分数为 0）
        image_shape = np.array(keypoint_map.shape[:2])
        if pad_random_features and (keep_k_points > keypoints.shape[0]):
            # 计算需要填充的关键点数量
            num_pad = keep_k_points - keypoints.shape[0]
            # 随机生成填充的关键点坐标
            keypoints_pad = (image_shape - 1) * np.random.uniform(size=(num_pad, 2))
            # 拼接关键点
            keypoints = np.concatenate((keypoints, keypoints_pad))
            # 生成填充关键点的置信度为 0
            scores_pad = np.zeros((num_pad))
            # 拼接置信度
            scores = np.concatenate((scores, scores_pad))

        # 通过双线性插值查找描述符
        # TODO: 使用双线性插值批量查找描述符
        # 交换坐标顺序，从 (行, 列) 转换为 (x, y)
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        descriptors = []
        for kp in keypoints:
            # 通过双线性插值查找描述符
            descriptors.append(utils.lookup_descriptor_bilinear(kp, descriptor_map))
        descriptors = np.array(descriptors)
        return keypoints, descriptors, scores