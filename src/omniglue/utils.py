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

"""Shared utility functions for OmniGlue."""

import math
from typing import Optional
import cv2
import numpy as np
import tensorflow as tf
import tensorflow as tf


# 定义 DINO 特征维度
DINO_FEATURE_DIM = 768
# 定义匹配阈值
MATCH_THRESHOLD = 1e-3


def lookup_descriptor_bilinear(
    keypoint: np.ndarray, descriptor_map: np.ndarray
) -> np.ndarray:
    """从密集描述符图中查找关键点的描述符值。

    使用双线性插值在非整数位置查找描述符值。

    参数:
        keypoint: 2维 numpy 数组，包含关键点的 (x, y) 图像坐标。
        descriptor_map: (H, W, D) 形状的 numpy 数组，表示密集描述符图。

    返回:
        输入 'keypoint' 位置的 D 维描述符值。

    异常:
        ValueError: 如果关键点位置超出描述符图的边界。
    """
    # 获取描述符图的高度和宽度
    height, width = np.shape(descriptor_map)[:2]
    # 检查关键点是否在描述符图的边界内
    if (
        keypoint[0] < 0
        or keypoint[0] > width
        or keypoint[1] < 0
        or keypoint[1] > height
    ):
        raise ValueError(
            '关键点位置 (%f, %f) 超出描述符图边界 (%i w x %i h).' % (keypoint[0], keypoint[1], width, height)
        )

    # 计算关键点 x 坐标的整数部分和可能的下一个整数部分
    x_range = [math.floor(keypoint[0])]
    if not keypoint[0].is_integer() and keypoint[0] < width - 1:
        x_range.append(x_range[0] + 1)
    # 计算关键点 y 坐标的整数部分和可能的下一个整数部分
    y_range = [math.floor(keypoint[1])]
    if not keypoint[1].is_integer() and keypoint[1] < height - 1:
        y_range.append(y_range[0] + 1)

    # 初始化双线性插值描述符
    bilinear_descriptor = np.zeros(np.shape(descriptor_map)[2])
    # 遍历 x 和 y 的整数范围进行双线性插值
    for curr_x in x_range:
        for curr_y in y_range:
            # 获取当前整数位置的描述符
            curr_descriptor = descriptor_map[curr_y, curr_x, :]
            # 计算双线性插值权重
            bilinear_scalar = (1.0 - abs(keypoint[0] - curr_x)) * (
                1.0 - abs(keypoint[1] - curr_y)
            )
            # 累加加权描述符
            bilinear_descriptor += bilinear_scalar * curr_descriptor
    return bilinear_descriptor


def soft_assignment_to_match_matrix(
    soft_assignment: tf.Tensor, match_threshold: float
) -> tf.Tensor:
    """将软分配矩阵转换为二进制匹配矩阵。

    在 soft_assignment 中搜索行和列的最大值，这些值表示两个唯一关键点集之间的互最近邻匹配。
    同时，确保匹配的分数值高于指定的阈值。

    参数:
        soft_assignment: (B, N, M) 形状的张量，包含不同集合特征之间的匹配可能性值。
                         N 是 image0 中的特征数量，M 是 image1 中的特征数量。
                         较高的值表示更可能匹配。
        match_threshold: 浮点数，用于判断匹配是否有效的阈值。

    返回:
        (B, N, M) 形状的二进制张量。在索引 (x, y) 处的值为 1 表示 image0 中索引 'x'（共 N 个）与 image1 中索引 'y'（共 M 个）匹配。
    """

    def _range_like(x, dim):
        """返回包含 (0, 1, 2, ..., N) 的张量，对应输入 x 的指定维度。"""
        return tf.range(tf.shape(x)[dim], dtype=x.dtype)

    # TODO(omniglue): 批量循环和 SparseTensor 较慢。使用 tf 操作优化。
    # 初始化 TensorArray 用于存储匹配矩阵
    matches = tf.TensorArray(tf.float32, size=tf.shape(soft_assignment)[0])
    # 遍历批次中的每个示例
    for i in range(tf.shape(soft_assignment)[0]):
        # 扩展维度以匹配输入形状
        scores = tf.expand_dims(soft_assignment[i, :], 0)  # 形状: (1, N, M)。

        # 找出行的最大值
        max0 = tf.math.reduce_max(scores, axis=2)  # 形状: (1, N)。
        # 找出行最大值的索引
        indices0 = tf.math.argmax(scores, axis=2)  # 形状: (1, N)。
        # 找出列的最大值
        indices1 = tf.math.argmax(scores, axis=1)  # 形状: (1, M)。

        # 找出互最近邻匹配
        mutual = tf.expand_dims(_range_like(indices0, 1), 0) == tf.gather(
            indices1, indices0, axis=1
        )

        # 创建匹配矩阵
        kp_ind_pairs = tf.stack(
            [_range_like(indices0, 1), tf.squeeze(indices0)], axis=1
        )
        mutual_max0 = tf.squeeze(tf.squeeze(tf.where(mutual, max0, 0), 0))
        sparse = tf.sparse.SparseTensor(
            kp_ind_pairs, mutual_max0, tf.shape(scores, out_type=tf.int64)[1:]
        )
        match_matrix = tf.sparse.to_dense(sparse)
        matches = matches.write(i, match_matrix)

    # 根据阈值进行阈值处理，并将值转换为二进制 (0, 1)
    match_matrix = matches.stack()
    match_matrix = match_matrix > match_threshold
    return match_matrix


def visualize_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    match_matrix: np.ndarray,
    match_labels: Optional[np.ndarray] = None,
    show_keypoints: bool = True,
    highlight_unmatched: bool = True,
    title: Optional[str] =None,
    line_width: int = 1,
    circle_radius: int = 1,
    circle_thickness: int = 2,
    rng: Optional['np.random.Generator'] =None,
):
    """生成两个图像的关键点和匹配的可视化结果。

    将 image0 放置在左上角，image1 放置在右下角，输出图像的右上角和左下角为空白矩形。
    如果两个图像的高度不同，不会对 image1 进行缩放。
    注意：关键点必须是 (x, y) 格式，而不是 (row, col)。如果 match_matrix 包含未匹配的尘桶（dustbins），则在可视化匹配之前会移除这些尘桶。

    参数:
        image0: (H, W, 3) 数组，包含 image0 的内容。
        image1: (H, W, 3) 数组，包含 image1 的内容。
        kp0: (N, 2) 数组，其中每一行代表 image0 中关键点的 (x, y) 坐标。
        kp1: (M, 2) 数组，其中每一行代表 image1 中关键点的 (x, y) 坐标。
        match_matrix: (N, M) 二进制数组，其中非零值表示关键点索引组成匹配。
        match_labels: (N, M) 二进制数组，其中非零值表示关键点索引组成的真实匹配。
                      当为 None 时，match_matrix 中的匹配将被随机着色。否则，match_matrix 中的匹配将根据准确性（与标签比较）着色。
        show_keypoints: 如果为 True，则可视化 image0 和 image1 中的所有关键点（包括未匹配的关键点）。
        highlight_unmatched: 如果为 True，则用蓝色高亮未匹配的关键点。
        title: 如果不为 None，则在可视化结果的左上角添加标题文本。
        line_width: 匹配线的宽度，以像素为单位。
        circle_radius: 关键点圆的半径，如果可视化关键点。
        circle_thickness: 关键点圆的厚度，如果可视化关键点。
        rng: np 随机数生成器，用于生成线条颜色。

    返回:
        包含 image0 和 image1 拼接在一起的 numpy 数组，并根据 match_matrix 添加匹配线。
        如果 show_keypoints 为 True，则还可视化两个图像中的关键点。
    """
    # 初始化随机数生成器
    if rng is None:
        rng = np.random.default_rng()

    # 复制输入参数，以防在函数中被修改
    kp0 = np.copy(kp0)
    kp1 = np.copy(kp1)

    # 检测未匹配的尘桶
    has_unmatched_dustbins = (match_matrix.shape[0] == kp0.shape[0] + 1) and (
        match_matrix.shape[1] == kp1.shape[0] + 1
    )

    # 获取 image0 的高度和宽度
    height0, width0 = image0.shape[:2]
    # 获取 image1 的高度和宽度
    height1, width1 = image1.shape[:2]

    # 计算输出图像的总高度和总宽度
    total_height = height0 + height1
    total_width = width0 + width1

    # 创建一个全零的空白图像，用于存储最终的可视化结果
    viz = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # 将 image0 放置在输出图像的左上角
    viz[:height0, :width0] = image0
    # 将 image1 放置在输出图像的右下角
    viz[height0:, width0:] = image1

    # 找出匹配的关键点对，如果存在未匹配的尘桶，则去除尘桶部分
    matches = np.argwhere(
        match_matrix[:-1, :-1] if has_unmatched_dustbins else match_matrix
    )

    # 遍历所有匹配的关键点对
    for match in matches:
        # 获取 image0 中关键点的坐标
        pt0 = (int(kp0[match[0], 0]), int(kp0[match[0], 1]))
        # 获取 image1 中关键点的坐标，并将其调整到输出图像的正确位置
        pt1 = (int(kp1[match[1], 0] + width0), int(kp1[match[1], 1] + height0))

        # 确保匹配线的颜色始终是随机的
        color = tuple(rng.integers(0, 255, size=3).tolist())

        # 在输出图像上绘制匹配线
        cv2.line(viz, pt0, pt1, color, line_width)

    # 可选地，在输出图像中添加圆圈以表示每个关键点
    if show_keypoints:
        # 遍历 image0 中的所有关键点
        for i in range(kp0.shape[0]):
            kp = kp0[i, :]
            # 如果需要高亮未匹配的关键点，且该关键点未匹配，则用蓝色绘制圆圈
            if highlight_unmatched and has_unmatched_dustbins and match_matrix[i, -1]:
                color = (255, 0, 0)  # 蓝色表示未匹配
            else:
                color = (0, 0, 255)  # 红色表示匹配

            # 绘制关键点圆圈
            cv2.circle(
                viz,
                tuple(kp.astype(np.int32).tolist()),
                circle_radius,
                color,
                circle_thickness,
            )

            # 在关键点旁边添加坐标文本
            text = f"({kp[0]:.2f}, {kp[1]:.2f})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x_text = int(kp[0]) + 5
            y_text = int(kp[1]) + 5
            if x_text + text_size[0] > width0:
                x_text = int(kp[0]) - text_size[0] - 5
            if y_text + text_size[1] > height0:
                y_text = int(kp[1]) - text_size[1] - 5
            cv2.putText(
                viz,
                text,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # 遍历 image1 中的所有关键点
        for j in range(kp1.shape[0]):
            kp = kp1[j, :]
            # 调整 image1 中关键点的坐标到输出图像的正确位置
            kp[0] += width0
            kp[1] += height0
            # 如果需要高亮未匹配的关键点，且该关键点未匹配，则用蓝色绘制圆圈
            if highlight_unmatched and has_unmatched_dustbins and match_matrix[-1, j]:
                color = (255, 0, 0)  # 蓝色表示未匹配
            else:
                color = (0, 0, 255)  # 红色表示匹配

            # 绘制关键点圆圈
            cv2.circle(
                viz,
                tuple(kp.astype(np.int32).tolist()),
                circle_radius,
                color,
                circle_thickness,
            )

            # 在关键点旁边添加坐标文本
            text = f"({kp[0]:.2f}, {kp[1]:.2f})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x_text = int(kp[0]) + 5
            y_text = int(kp[1]) + 5
            if x_text + text_size[0] > total_width:
                x_text = int(kp[0]) - text_size[0] - 5
            if y_text + text_size[1] > total_height:
                y_text = int(kp[1]) - text_size[1] - 5
            cv2.putText(
                viz,
                text,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    # 定义 image1 的四个角点坐标
    image1_corners = [(0, 0), (width1 - 1, 0), (0, height1 - 1), (width1 - 1, height1 - 1)]

    # 移除 image1 的左上角坐标信息
    image1_corners = image1_corners[1:]

    # 在 image1 中绘制剩余的角点
    for point in image1_corners:
        x, y = point
        x += width0  # 调整到合并图像中的正确位置
        y += height0  # 调整到合并图像中的正确位置
        if 0 <= x < total_width and 0 <= y < total_height:
            # 绘制角点圆圈
            cv2.circle(
                viz,
                (x, y),
                circle_radius * 2,  # 使用更大的圆圈来标记角点
                (0, 255, 255),  # 青色表示角点
                circle_thickness * 2,  # 使用更粗的线条
            )
            # 在角点旁边添加文本
            text = f"({x}, {y})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x_text = x + 5
            y_text = y + 5
            if x_text + text_size[0] > total_width:
                x_text = x - text_size[0] - 5
            if y_text + text_size[1] > total_height:
                y_text = y - text_size[1] - 5
            cv2.putText(
                viz,
                text,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # 青色文本
                1,
                cv2.LINE_AA,
            )

    # 定义 image0 的四个角点坐标
    image0_corners = [(0, 0), (width0 - 1, 0), (0, height0 - 1), (width0 - 1, height0 - 1)]

    # 在 image0 中绘制四个角点
    for point in image0_corners:
        x, y = point
        if 0 <= x < width0 and 0 <= y < height0:
            # 绘制角点圆圈
            cv2.circle(
                viz,
                (x, y),
                circle_radius * 2,  # 使用更大的圆圈来标记角点
                (0, 255, 255),  # 青色表示角点
                circle_thickness * 2,  # 使用更粗的线条
            )
            # 在角点旁边添加文本
            text = f"({x}, {y})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x_text = x + 5
            y_text = y + 5
            if x_text + text_size[0] > width0:
                x_text = x - text_size[0] - 5
            if y_text + text_size[1] > height0:
                y_text = y - text_size[1] - 5
            cv2.putText(
                viz,
                text,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # 青色文本
                1,
                cv2.LINE_AA,
            )

    # 如果提供了标题，则在可视化结果的左上角添加标题文本
    if title is not None:
        viz = cv2.putText(
            viz,
            title,
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),  # 红色标题
            2,
            cv2.LINE_AA,
        )

    return viz



