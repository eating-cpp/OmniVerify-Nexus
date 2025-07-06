# 版权所有 2024 Google LLC
#
# 根据 Apache 许可证，版本 2.0（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则在许可证下分发的软件
# 按“原样”基础分发，
# 没有任何形式的保证和条件，无论是明示的还是暗示的。
# 请参阅许可证以了解管理权限和
# 限制的特定语言。

"""用于执行 DINOv2 推理的包装器。"""

import cv2
import numpy as np
from third_party.dinov2 import dino
from src.omniglue import utils
import tensorflow as tf
import torch
import os

import tensorflow as tf


# 设置日志级别为 2，忽略警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DINOExtract:
    """用于初始化 DINO 模型并从图像中提取特征的类。"""
    count_dinotimes=0
    def __init__(self, cpt_path: str, feature_layer: int = 1):
        # 检查是否有可用的 CUDA 设备，如果有则使用 GPU，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 如果gpu设备可用输出可用的gpu设备数量
        # if torch.cuda.is_available():
        #     print(f"GPU 数量: {torch.cuda.device_count()}")
        #     print(f"当前 GPU 设备: {torch.cuda.current_device()}")
        #     print(f"当前 GPU 设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        # else:
        #     print(">:没有可用的 GPU 设备，正在使用cpu")
            
        # 指定要提取特征的层
        self.feature_layer = feature_layer
        # 初始化 DINO 的基础视觉 Transformer 模型
        self.model = dino.vit_base()
        # 从指定路径加载模型的预训练权重
        state_dict_raw = torch.load(cpt_path, map_location='cpu')

        # state_dict = {}
        # for k, v in state_dict_raw.items():
        #   state_dict[k.replace('blocks', 'blocks.0')] = v

        # 将加载的权重应用到模型上
        self.model.load_state_dict(state_dict_raw)
        # 将模型移动到指定的设备（GPU 或 CPU）上
        self.model = self.model.to(self.device)
        # 将模型设置为评估模式
        self.model.eval()

        # 图像最大边长的阈值
        self.image_size_max = 630

        # 高度和宽度的下采样率
        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

    def __call__(self, image,order,file_name) -> np.ndarray:
        # 调用 forward 方法进行特征提取
        
        # if order==1:# 从库中加载特征
        #     self.count_dinotimes+=1
        #     # features = self.forward(image)
        #     # # 保存特征到 txt 文件
        #     # if self.count_dinotimes==1:
        #     #     self.save_features_to_txt(features, './features_data_base/image1/dino/dino_features1.txt')
        #     # elif self.count_dinotimes==2:
        #     #     self.save_features_to_txt(features, './features_data_base/image2/dino/dino_features2.txt')
            
        #     # 加载特征
        #     if self.count_dinotimes==1:
        #         features = self.load_features_from_txt('./features_data_base/image1/dino/dino_features1.txt')
        #         print(f">   从database中加载dino1特征形状：{features.shape}")
            
        #     elif self.count_dinotimes==2:
        #         features = self.load_features_from_txt('./features_data_base/image2/dino/dino_features2.txt')
        #         print(f">   从database中加载dino2特征形状：{features.shape}")
        features = self.forward(image)
        
        if order==2:# 录入特征
            # self.count_dinotimes+=1
            # features = self.forward(image)
            
            directory = os.path.dirname('./features_data_base/'+file_name+'/dino/')
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # 保存特征到 txt 文件
            self.save_features_to_txt(features, './features_data_base/'+file_name+'/dino/dino_features.txt')
            
            # return features
            
        else:
            self.count_dinotimes+=1
            # features = self.forward(image)
            
            # 保存特征到 txt 文件
            # if self.count_dinotimes==1:
            #     self.save_features_to_txt(features, './features_data_base/image1/dino/dino_features1.txt')
            #     print(f">:将dino1特征保存到 ./features_data_base/image1/dino/dino_features1.txt")
            # elif self.count_dinotimes==2:
            #     self.save_features_to_txt(features, './features_data_base/image2/dino/dino_features2.txt')
            #     print(f">:将dino2特征保存到 ./features_data_base/image2/dino/dino_features2.txt")
            # # print(f">   从新图片中提取dino特征形状：{features.shape}")
            return features
    
    def save_features_to_txt(self, features, filename):
        """
        将特征保存到 txt 文件中。
        :param features: 要保存的特征，numpy 数组
        :param filename: 保存的文件名
        """
        # 如果特征是多维数组，将其转换为二维数组
        if len(features.shape) > 2:
            features = features.reshape(-1, features.shape[-1])
        np.savetxt(filename, features)
    
    def load_features_from_txt(self,filename):
        """
        从 txt 文件中加载特征。
        :param filename: 保存特征的文件名
        :return: 加载的特征，numpy 数组
        """
        features = np.loadtxt(filename)
        return features

    def forward(self, image: np.ndarray) -> np.ndarray:
        """将图像输入 DINO ViT 模型以提取特征。

        参数:
          image: (H, W, 3) 的 numpy 数组，解码后的图像字节，值范围 [0, 255]。

        返回:
          features: (H // 14, W // 14, C) 的 numpy 数组，表示图像特征。
        """
        # 调整输入图像的大小
        image = self._resize_input_image(image)
        # 对图像进行预处理
        image_processed = self._process_image(image)
        # 添加批次维度，转换为浮点型并移动到指定设备
        image_processed = image_processed.unsqueeze(0).float().to(self.device)
        # 提取图像特征
        features = self.extract_feature(image_processed)
        # 去除批次维度，调整维度顺序并转换为 numpy 数组
        features = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return features

    def _resize_input_image(
        self, image: np.ndarray, interpolation=cv2.INTER_LINEAR
    ):
        """调整图像大小，使两个维度都能被下采样率整除。"""
        # 获取图像的高度和宽度
        h_image, w_image = image.shape[:2]
        # 判断高度是否大于宽度
        h_larger_flag = h_image > w_image
        # 获取图像的最大边长
        large_side_image = max(h_image, w_image)

        # 如果最大边长超过阈值，则调整图像大小以加速 ViT 骨干网络的推理
        if large_side_image > self.image_size_max:
            if h_larger_flag:
                # 如果高度较大，调整高度为阈值，宽度按比例调整
                h_image_target = self.image_size_max
                w_image_target = int(self.image_size_max * w_image / h_image)
            else:
                # 如果宽度较大，调整宽度为阈值，高度按比例调整
                w_image_target = self.image_size_max
                h_image_target = int(self.image_size_max * h_image / w_image)
        else:
            # 如果最大边长未超过阈值，保持原始大小
            h_image_target = h_image
            w_image_target = w_image

        # 计算调整后的高度和宽度，使其能被下采样率整除
        h, w = (
            h_image_target // self.h_down_rate,
            w_image_target // self.w_down_rate,
        )
        h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
        # 使用 OpenCV 调整图像大小
        image = cv2.resize(image, (w_resize, h_resize), interpolation=interpolation)
        return image

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """将图像转换为 PyTorch 张量并进行归一化处理。"""
        # ImageNet 数据集的均值
        mean = np.array([0.485, 0.456, 0.406])
        # ImageNet 数据集的标准差
        std = np.array([0.229, 0.224, 0.225])

        # 将图像像素值从 [0, 255] 转换为 [0, 1]
        image_processed = image / 255.0
        # 对图像进行归一化处理
        image_processed = (image_processed - mean) / std
        # 将 numpy 数组转换为 PyTorch 张量，并调整维度顺序
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
        return image_processed

    def extract_feature(self, image):
        """从图像中提取特征。

        参数:
          image: (B, 3, H, W) 的 PyTorch 张量，使用 ImageNet 的均值和标准差进行归一化。

        返回:
          features: (B, C, H//14, W//14) 的 PyTorch 张量，表示图像特征。
        """
        # 获取图像的批次大小、通道数、原始高度和宽度
        b, _, h_origin, w_origin = image.shape
        # 获取指定层的中间特征
        out = self.model.get_intermediate_layers(image, n=self.feature_layer)[0] #提取特征
        # 计算调整后的高度和宽度
        h = int(h_origin / self.h_down_rate)
        w = int(w_origin / self.w_down_rate)
        # 获取特征的维度
        dim = out.shape[-1]
        # 调整特征的形状并分离梯度
        out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()
        
        #输出out
        # print(f"out:{out}")
        return out

def _preprocess_shape(
    h_image, w_image, image_size_max=630, h_down_rate=14, w_down_rate=14
):
    # 去除张量的单维度
    h_image = tf.squeeze(h_image)
    w_image = tf.squeeze(w_image)
    # logging.info(h_image, w_image)

    # 判断高度是否大于宽度
    h_larger_flag = tf.greater(h_image, w_image)
    # 获取图像的最大边长
    large_side_image = tf.maximum(h_image, w_image)

    # 定义当高度较大时计算新维度的函数
    def resize_h_larger():
        h_image_target = image_size_max
        w_image_target = tf.cast(image_size_max * w_image / h_image, tf.int32)
        return h_image_target, w_image_target

    # 定义当宽度较大或相等时计算新维度的函数
    def resize_w_larger_or_equal():
        w_image_target = image_size_max
        h_image_target = tf.cast(image_size_max * h_image / w_image, tf.int32)
        return h_image_target, w_image_target

    # 定义保持原始维度的函数
    def keep_original():
        return h_image, w_image

    # 根据最大边长是否超过阈值，选择不同的维度调整策略
    h_image_target, w_image_target = tf.cond(
        tf.greater(large_side_image, image_size_max),
        lambda: tf.cond(h_larger_flag, resize_h_larger, resize_w_larger_or_equal),
        keep_original,
    )

    # 调整维度使其能被 patch 大小整除
    h = h_image_target // h_down_rate
    w = w_image_target // w_down_rate
    h_resize = h * h_down_rate
    w_resize = w * w_down_rate

    # 添加单维度
    h_resize = tf.expand_dims(h_resize, 0)
    w_resize = tf.expand_dims(w_resize, 0)

    return h_resize, w_resize

def get_dino_descriptors(dino_features, keypoints, height, width, feature_dim):
    """使用 Superpoint 关键点获取 DINO 描述符。

    参数:
      dino_features: 一维的 DINO 特征。
      keypoints: Superpoint 关键点位置，格式为 (x, y)，单位为像素，形状为 (N, 2)。
      height: 图像高度，类型为 tf.Tensor.int32。
      width: 图像宽度，类型为 tf.Tensor.int32。
      feature_dim: DINO 特征通道大小，类型为 tf.Tensor.int32。

    返回:
      插值后的 DINO 描述符。
    """
    # TODO(omniglue): 修复硬编码的 DINO patch 大小 (14)。
    # 将高度和宽度调整为一维张量
    height_1d = tf.reshape(height, [1])
    width_1d = tf.reshape(width, [1])

    # 对图像的高度和宽度进行预处理
    height_1d_resized, width_1d_resized = _preprocess_shape(
        height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14
    )

    # 计算特征图的高度和宽度
    height_feat = height_1d_resized // 14
    width_feat = width_1d_resized // 14
    # 将特征维度调整为一维张量
    feature_dim_1d = tf.reshape(feature_dim, [1])

    # 拼接特征图的高度、宽度和特征维度
    size_feature = tf.concat([height_feat, width_feat, feature_dim_1d], axis=0)
    
     # 添加调试信息
    # print(f"输入张量 dino_features 的元素数量: {tf.size(dino_features)}")
    # print(f"请求的形状 size_feature: {size_feature}")
    # print(f"请求的形状所需元素数量: {tf.reduce_prod(size_feature)}")
    
    
    # 调整 DINO 特征的形状
    dino_features = tf.reshape(dino_features, size_feature)

    # 将图像的宽度和高度转换为浮点型张量
    img_size = tf.cast(tf.concat([width_1d, height_1d], axis=0), tf.float32)
    # 将特征图的宽度和高度转换为浮点型张量
    feature_size = tf.cast(
        tf.concat([width_feat, height_feat], axis=0), tf.float32
    )

    # 计算关键点在特征图上的位置
    keypoints_feature = (
        keypoints
        / tf.expand_dims(img_size, axis=0)
        * tf.expand_dims(feature_size, axis=0)
    )

        # 初始化 DINO 描述符列表
    dino_descriptors = []
    # 遍历每个关键点
    count=0
    for kp in keypoints_feature:
        # 使用双线性插值方法查找关键点对应的 DINO 描述符
        descriptor = utils.lookup_descriptor_bilinear(kp.numpy(), dino_features.numpy())
        dino_descriptors.append(descriptor)
        # 输出当前关键点对应的 DINO 描述符
        # print(f"关键点{count}({kp.numpy()}) 描述符: {descriptor}")
        count+=1
    
    # 将 DINO 描述符列表转换为 TensorFlow 张量
    dino_descriptors = tf.convert_to_tensor(
        np.array(dino_descriptors), dtype=tf.float32
    )
    return dino_descriptors