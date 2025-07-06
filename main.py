# 声明全局变量 og
og = None

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# utf8 格式
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import Ui_Main
from Ui_Main import Ui_Dialog
import Ui_Output_cmd_graph
import Ui_takephotoUI
import Ui_inputdialog
import Ui_startUI
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import src.omniglue as omniglue
from src.omniglue import utils
from PIL import Image
import cv2
from src.omniglue.pre_process import combine_photos, take_photo
import shutil
from Ui_UI_omni1_visualization import Ui_Form

plt.rcParams['font.family'] = 'SimHei'  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入 ConfidenceAnalysis 类
from src.omniglue.confidence_analysis import ConfidenceAnalysis

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置日志级别为 2，忽略警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QCameraViewfinder,QGraphicsVideoItem
from PyQt5.QtWidgets import *


class GraphVisualizationDialog(Ui_Form,):
    def __init__(self, subDialog,max_match_confidence,image0_fp,image1_path,distribution_graph_path,match_kp0,match_kp1,name):
        super().setupUi(subDialog)  # 调用父类的 setupUI 函数
        self.max_match_confidence = max_match_confidence
        self.image0_fp = image0_fp
        self.image1_path = image1_path
        self.distribution_graph_path = distribution_graph_path
        self.match_kp0 = match_kp0
        self.match_kp1 = match_kp1
        self.name=name
        self.confidence=None
        self.image0=None
        print(self.image0_fp)
        print(f"获取的置信度图{self.distribution_graph_path}")
        self.label_4.setText(f"系统认为的正确的对象：{self.name}")
        self.image1=np.array(Image.open(self.image1_path).convert('RGB'))
        
        self.Order1_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.Order2_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.Order4_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.pushButton.clicked.connect(self.detect_and_visualize)
        self.horizontalSlider.valueChanged.connect(self.get_current_value)
        self.pushButton.setText("进行可视化匹配")
        
        
        # 将相对路径转换为绝对路径
        self.distribution_graph_path = os.path.abspath(self.distribution_graph_path)
        print(f"转换后的绝对路径: {self.distribution_graph_path}")
        temp_path=self.distribution_graph_path

       
        # 在 label6 中显示图像
        pixmap = QtGui.QPixmap(temp_path)
        if not pixmap.isNull():
            # 让图片自适应 label 大小
            self.label_6.setScaledContents(True)
            self.label_6.setPixmap(pixmap)
            

    def detect_and_visualize(self):
        
        self.pushButton.setText("正在可视化")
        # 强制刷新上面的“正在可视化”
        QtWidgets.QApplication.processEvents()
        
        
        time.sleep(0.5)
        self.pushButton.setEnabled(False)
        self.image0=np.array(Image.open(self.image0_fp).convert('RGB'))
        match_threshold = self.confidence
        keep_idx = []
        for i in range(self.match_kp0.shape[0]):
            if self.max_match_confidence[i] > match_threshold:
                keep_idx.append(i)
        num_filtered_matches = len(keep_idx)
        match_kp0 = self.match_kp0[keep_idx]
        match_kp1 =self.match_kp1[keep_idx]
        # match_confidences = self.match_confidences[keep_idx]
        
        viz = utils.visualize_matches(
            self.image0,
            self.image1,
            match_kp0,
            match_kp1,
            np.eye(num_filtered_matches),
            match_labels=None,
            show_keypoints=True,
            highlight_unmatched=True,
            title=f"{num_filtered_matches} matches",
            line_width=2,
        )
        temp_output_path = f"./result/OMNIoutput_confidence_{self.confidence}.png"
        plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
        plt.axis("on")
        plt.imshow(viz)
        plt.imsave(temp_output_path, viz)
        plt.close()
        # 全部改成使用opencv
        # cv2.imwrite(temp_output_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        
        
        
        image = QImage(temp_output_path)
        if not image.isNull():
            # 转换为 RGB 格式
            # image = image.convertToFormat(QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            # 在 label7 中显示图像
            if not pixmap.isNull():
                # 让图片自适应 label 大小
                self.label_7.setScaledContents(True)
                self.label_7.setPixmap(pixmap)
        self.pushButton.setText("完成")
    
    def append_text(self, text):
        """
        向 QTextEdit 控件 cmdshow 追加文本
        :param text: 要追加的文本内容
        """
        self.textEdit.append(text)
        # 强制刷新显示
        QtWidgets.QApplication.processEvents()
    def clear_cmdshow(self):
        self.textEdit.clear()
        
    def get_current_value(self):
            """
            获取当前滑块的值并显示在 label_8 中。
            """
            self.pushButton.setEnabled(True)
            current_value = (self.horizontalSlider.value())/1000
            self.label.setText('置信度选择：'+str(current_value))
            self.confidence = current_value
            self.pushButton.setText("开始可视化")
        
Camera_selected_folder=None
class CameraDialog(Ui_takephotoUI.Ui_takephoto):
    def __init__(self, subDialog):
        super().setupUi(subDialog)  # 假设 Ui_takephoto 有 setupUi 方法
        
        # 定义信号，用于传递选择的文件夹路径
        self.folder_selected = pyqtSignal(str)
        self.flag = 0
        self.photo_count = 0
        self.camera = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.pushButton.clicked.connect(self.takephoto)
        self.pushButton_2.clicked.connect(self.combine_photos)
        self.pushButton_3.clicked.connect(self.choosefolder)
        self.pushButton_4.clicked.connect(self.reset)
        self.pushButton_4.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.start_camera()
        # self.callback_selected_folder = callback_selected_folder

    def choosefolder(self):
        global Camera_selected_folder
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        # 弹出文件选择对话框
        if self.file_dialog.exec_():
            # 获取用户选择的文件夹路径
            Camera_selected_folder = self.file_dialog.selectedFiles()[0]
            print(f"用户选择的文件夹路径是: {Camera_selected_folder}")
            self.pushButton_2.setText(Camera_selected_folder)
            self.pushButton.setEnabled(True)

    def reset(self):
        global Camera_selected_folder
        if Camera_selected_folder and os.path.isdir(Camera_selected_folder):
            try:
                # 清空文件夹中的所有文件和子文件夹
                for filename in os.listdir(Camera_selected_folder):
                    file_path = os.path.join(Camera_selected_folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"已清空文件夹 {Camera_selected_folder} 中的所有内容。")
            except Exception as e:
                print(f"清空文件夹 {Camera_selected_folder} 时出错: {e}")
        else:
            print("未指定有效文件夹路径或文件夹不存在。")
            
        # 清除 QGraphicsView 中的图像
        for i in range(1, 13):
            graphics_view_name = f"photo{i}"
            graphics_view = getattr(self, graphics_view_name, None)
            if graphics_view:
                scene = graphics_view.scene()
                if scene:
                    scene.clear()  # 清除场景中的所有项
                graphics_view.setScene(None)  # 移除场景
        
        # 重置 photo_count 和按钮状态
        self.photo_count = 0
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_4.setEnabled(False)

    def takephoto(self):
        global Camera_selected_folder
        ret, frame = self.camera.read()
        if ret:
            base_filename = Camera_selected_folder
            print(base_filename)
            if base_filename:
                self.filename = os.path.join(base_filename, f"{self.photo_count + 1}.jpg")
                cv2.imwrite(self.filename, frame)  # 将图像保存到指定路径
                self.photo_count += 1

                if self.photo_count == 0:
                    self.pushButton_4.setEnabled(False)
                else:
                    self.pushButton_4.setEnabled(True)
                if self.photo_count == 12:
                    self.pushButton.setEnabled(False)
                    self.pushButton_2.setEnabled(True)
                    self.pushButton_2.setText("开始拼接和过曝处理")

                # 将拍摄的照片显示在对应的 QGraphicsView 中
                if self.photo_count <= 12:
                    # 获取对应的 QGraphicsView 对象
                    graphics_view_name = f"photo{self.photo_count}"
                    graphics_view = getattr(self, graphics_view_name, None)
                    if graphics_view:
                        # 将 OpenCV 图像转换为 QPixmap
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(image)

                        # 创建 QGraphicsScene 并添加图像项
                        scene = QtWidgets.QGraphicsScene()
                        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                        scene.addItem(pixmap_item)

                        # 将场景设置给 QGraphicsView 并调整视图
                        graphics_view.setScene(scene)
                        graphics_view.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)
            else:
                print("未选择有效文件夹路径，无法保存照片。")

    def combine_photos(self):
        global Camera_selected_folder
        self.SubWindow_cmd = QtWidgets.QDialog()
        self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)
        self.subui_cmd.cmdButton.setText("正在加载...")
        self.SubWindow_cmd.show()
        self.subui_cmd.append_text("> 正在拼接图像和过曝处理...")
        
        if Camera_selected_folder:
            combine_photos(Camera_selected_folder, 12)
        
        self.SubWindow_cmd.close()

    def start_camera(self):
        self.timer.start(30)
        # self.pushButton.setEnabled(True)
        # self.pushButton_2.setEnabled(False)
        
    def pause_camera(self):
        if self.timer.isActive():
            self.timer.stop()  # 停止定时器
    
    def update_frame(self):
        ret, frame = self.camera.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            # 修改部分
            self.label.setScaledContents(True)
            self.label.setPixmap(pixmap)



class GifThread(QThread):
    def __init__(self, movie):
        super().__init__()
        self.movie = movie
        
    def run(self):
        self.movie.start()

class ModelLoadingThread(QThread):
    def run(self):
        global og
        # 加载模型
        og = omniglue.OmniGlue(
            og_export="./models/og_export",
            sp_export="./models/sp_v6",
            dino_export="./models/dinov2_vitb14_pretrain.pth",
        )


class StartUI_Dialog(Ui_startUI.Ui_Dialog):
    def __init__(self, subDialog):
        super().setupUi(subDialog)  # 调用父类的 setupUI 函数
        gif_path = "./icon/loading.gif"
        # Create a QMovie object and load the GIF file
        movie = QtGui.QMovie(gif_path)
        
        # 设置label4为正方形尺寸
        square_size = min(self.label_4.width(), self.label_4.height())
        self.label_4.setFixedSize(square_size, square_size)
        
        # 设置GIF缩放尺寸为正方形
        movie.setScaledSize(QtCore.QSize(square_size, square_size))
        
        # Set the QMovie object to the label_4 widget
        self.label_4.setMovie(movie)
        movie.start()
        
        # # 启动 GIF 线程
        # self.gif_thread = GifThread(movie)
        # self.gif_thread.start()
        
    def append_text(self, text):
        """
        向 QTextEdit 控件 cmdshow 追加文本
        :param text: 要追加的文本内容
        """
        self.textEdit.clear()
        self.textEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.textEdit.append(text)
        # 强制刷新显示
        QtWidgets.QApplication.processEvents()


class Output_cmd_graph_Dialog(Ui_Output_cmd_graph.Ui_Dialog):
    def __init__(self, subDialog):
        super().setupUi(subDialog)  # 调用父类的 setupUI 函数
        
        self.cmdButton.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        # self.graphButton.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))

    def append_text(self, text):
        """
        向 QTextEdit 控件 cmdshow 追加文本
        :param text: 要追加的文本内容
        """
        self.CmdShow.append(text)
        # 强制刷新显示
        QtWidgets.QApplication.processEvents()
    def clear_cmdshow(self):
        self.CmdShow.clear()


class MainDialog(Ui_Main.Ui_Dialog):
    def __init__(self, Dialog):
        global og
        super().setupUi(Dialog)
        self.og = og  # 保存og对象为实例变量
        # Dialog.setStyleSheet("background-color: #FFF7FA;")
        
        # self.SubWindow_cmd = QtWidgets.QMainWindow()
        # self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
        
        # 连接按钮点击信号到切换页面的方法
        self.Order1_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.Order2_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.Order3_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.Order4_bar.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))

        """
        原omni.py中orde==1功能实现
        """
        self.pushButton_4.clicked.connect(lambda:self.OMNI1(1))
        self.pushButton_5.clicked.connect(lambda:self.OMNI1(2))
        self.pushButton_6.clicked.connect(lambda:self.OMNI1(3))
        self.pushButton_10.clicked.connect(lambda:self.OMNI1(4))

        """
        原omni.py中orde==2功能实现
        """
        self.pushButton_7.clicked.connect(lambda:self.OMNI2(1))
        self.pushButton_8.clicked.connect(lambda:self.OMNI2(2))
        self.pushButton_9.clicked.connect(lambda:self.OMNI2(3))
        self.pushButton_11.clicked.connect(lambda:self.OMNI2(4))

        """
        原omni.py中orde==3功能实现
        """
        self.pushButton.clicked.connect(lambda:self.OMNI3(1))
        self.pushButton_2.clicked.connect(lambda:self.OMNI3(2))
        self.pushButton_3.clicked.connect(lambda:self.OMNI3(3))
        # self.pushButton_3.clicked.connect(self.open)
        self.horizontalSlider.valueChanged.connect(lambda:self.OMNI3(4))
    
        self.final_file_name_image1 = None
        
        self.textEdit.append("\n")
        self.textEdit.append("当前版本:2.1.9")
        self.textEdit.append("\n")
        self.textEdit.append("联系方式:liyujie2305@outlook.com")
    
    def OMNI1(self,orderindex):
        def load_from_raw(self):
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
           
            self.subui_cmd.cmdButton.setText("正在加载...")
            self.subui_cmd.append_text("> 等待用户输入图像...")
            self.file_dialog = QtWidgets.QFileDialog()
            self.file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)  # 设置为选择文件夹模式
            # self.image1 = None


            self.subui_cmd.cmdButton.setText("正在加载...")
            max_photos = 12

            if self.file_dialog.exec_():
                self.SubWindow_cmd.show()
                self.subui_cmd.append_text("> 正在处理图像...")
                folder_name = self.file_dialog.selectedFiles()[0]
                existing_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name)]
                all_files_exist = True
                for i in range(1, max_photos + 1):
                    target_filename = os.path.join(folder_name, f"{i}.jpg")
                    if target_filename not in existing_files:
                        all_files_exist = False
                        print(f"缺少照片文件 {target_filename}，将调用摄像头拍照。")
                        break

                if all_files_exist:
                    self.image1, self.final_file_name_image1 = combine_photos(folder_name, max_photos)
                    self.subui_cmd.append_text("> 图像处理完成")
                    self.subui_cmd.cmdButton.setText("完成")
                    
                    show_preprocessed_image(self, self.final_file_name_image1)
            else:
                print("用户取消了文件夹选择。")
                self.subui_cmd.append_text("> 已取消")
                # 可以在这里添加更多处理逻辑，比如重置相关变量等
                self.image1 = None
                self.final_file_name_image1 = None
            
            pass
        def load_from_preprocessed(self):
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Images (*.jpg *.png)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    self.image_path = file_paths[0]
                    # 调用 show_preprocessed_image 函数显示图片
                    show_preprocessed_image(self, self.image_path)
                    self.image1=(Image.open(self.image_path).convert("RGB"))
                    show_preprocessed_image(self,self.image_path)
                    self.final_file_name_image1 = self.image_path
            else:
                print("用户取消了文件选择。")
                self.image_path = None  # 确保 image_path 为空
                self.image1 = None  # 确保 image1 为空
                self.final_file_name_image1 = None
            
            pass
        def load_from_camera(self):
            global Camera_selected_folder
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)
            self.SubWindow_cmd.show()
            self.subui_cmd.cmdButton.setText("正在加载...")
            self.subui_cmd.append_text("> 正在打开摄像头并处理图像...")

            self.TakePhoto_ui = QtWidgets.QDialog()
            self.takephoto_ui_ui = CameraDialog(self.TakePhoto_ui)
            
           # 连接信号到槽函数
            # self.takephoto_ui_ui.folder_selected.connect(self.handle_selected_folder)
            
            self.TakePhoto_ui.show()
            self.SubWindow_cmd.close()
            foldername=Camera_selected_folder
            print(foldername)
            
            # 等待 TakePhoto_ui 关闭
            self.TakePhoto_ui.exec_()
            
            foldername = Camera_selected_folder
            print(foldername)
            if foldername:
                # ... existing code ...
                lastfolder = os.path.basename(foldername)
                # 修正路径拼接，原代码存在多余的逗号
                self.image_path = os.path.join(foldername, f"{lastfolder}.jpg")
                try:
                    self.image1 = Image.open(self.image_path).convert("RGB")
                    show_preprocessed_image(self, self.image_path)
                    self.final_file_name_image1 = self.image_path
                except FileNotFoundError:
                    print(f"未找到文件: {self.image_path}")
            else:
                print("未选择有效文件夹路径，无法处理图像。")
                self.image1 = None
                self.final_file_name_image1 = None
            
            # foldername=selected_folder
            # print(foldername)
            # last_folder_name=os.path.basename(foldername)
            # self.image_path=os.path.join(foldername,last_folder_name,'.jpg')
            # self.image1=(Image.open(self.image_path).convert("RGB"))
            # show_preprocessed_image(self,self.image_path)

        # def handle_selected_folder(self, folder_path):
        #     print(folder_path)
        #     last_folder_name = os.path.basename(folder_path)
        #     # 修正路径拼接逻辑，推测你可能想拼接文件名
        #     self.image_path = os.path.join(folder_path, f"{last_folder_name}.jpg")
        #     try:
        #         self.image1 = Image.open(self.image_path).convert("RGB")
        #         show_preprocessed_image(self, self.image_path)
        #     except FileNotFoundError:
        #         print(f"未找到文件: {self.image_path}")
        
        def detect_and_match(self,image1):
            
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
            
            
            self.subui_cmd.cmdButton.setText("正在加载...")
            self.SubWindow_cmd.show()
            
            
            scaled_image=image1
            image1=np.array(scaled_image)
            
            
            print(f"> 开始提取特征...")
            # self.subui_cmd.cmdButton.setText("> 正在加载...")
            self.subui_cmd.append_text("> 开始提取特征...")
            height1,width1,sp_features1,dino_descriptors1=self.og.FindFeatures_single_image(image1,1)
            print(f"> 提取完成")
            self.subui_cmd.append_text("> 提取完成...")
            
            features_data_base_path = './features_data_base/'
            if os.path.exists(features_data_base_path):
                subfolders = [f for f in os.listdir(features_data_base_path) if os.path.isdir(os.path.join(features_data_base_path, f))]
                folder_count = len(subfolders)
                print(f">\t在 {features_data_base_path} 下有 {folder_count} 个子文件夹。")
                self.subui_cmd.append_text(f"> 在 {features_data_base_path} 下有 {folder_count} 个子文件夹。")
            else:
                print(f"> {features_data_base_path} 路径不存在。")
                self.subui_cmd.append_text(f"> {features_data_base_path} 路径不存在。")
                
            
            total_slices=1000
            confidences_levels = np.linspace(0, 1, total_slices)
            max_match_file_name_list = [None] * total_slices
            max_num_matches_list = [0] * total_slices
            mean_confidences_list = [0] * folder_count

            # 用于存储每个文件的置信度分布
            all_confidence_distributions = []
            all_subfolder_names = []  # 用于存储所有子文件夹的名称
            self.all_match_confidence=[]
            self.all_match_kp0=[]
            self.all_match_kp1=[]

            # ... existing code ...
            
            output_path=None
            for i in range(folder_count):
                print(f" ")
                subfolder_name = subfolders[i]
                subfolder_path = os.path.join(features_data_base_path, subfolder_name)
                print(f"> 当前正在处理子文件夹: {subfolder_path}")
                self.subui_cmd.append_text(f"> 当前正在处理子文件夹: {subfolder_path}")

                current_file_name = subfolder_name

                match_kp0, match_kp1, match_confidences = self.og.FindMatches_single_image(height1, width1, sp_features1, dino_descriptors1, current_file_name)  # 调用 OmniGlue 的 FindMatches 方法进行匹配
                num_matches = match_kp0.shape[0]  # 获取匹配点的数量

                # 计算当前子文件夹的置信度分布
                distribution = []
                for level in confidences_levels:
                    num_filtered_matches = np.sum(match_confidences > level)
                    distribution.append(num_filtered_matches)

                # 收集当前子文件夹的置信度分布数据
                all_confidence_distributions.append(distribution)
                all_subfolder_names.append(subfolder_name)
                self.all_match_confidence.append(match_confidences)
                self.all_match_kp0.append(match_kp0)
                self.all_match_kp1.append(match_kp1)
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
                
                self.subui_cmd.append_text(f"> 完成")
                self.subui_cmd.append_text(f"\n")
            
            self.subui_cmd.append_text(f"> 已将所有子文件夹的置信度分布折线图保存到 {output_path}")
            
            
            self.subui_cmd.cmdButton.setText("正在分析...")
            # # 创建一个dialog，用于显示置信度折线图
            # dialog = QtWidgets.QDialog()
            # dialog.setWindowTitle("置信度折线图")
            # layout = QtWidgets.QVBoxLayout()

            # 创建 QGraphicsView 和 QGraphicsScene 来显示图像
            # scene = QtWidgets.QGraphicsScene()
            # view = QtWidgets.QGraphicsView(scene)
            # pixmap = QtGui.QPixmap(output_path)
            # if not pixmap.isNull():
            #     pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
            #     scene.addItem(pixmap_item)
            #     view.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)

            # layout.addWidget(view)
            # dialog.setLayout(layout)
            # dialog.show()
            
            analysis = ConfidenceAnalysis(all_confidence_distributions, all_subfolder_names,self.all_match_confidence,self.all_match_kp0,self.all_match_kp1)

            # 调用 integrate_and_sort 方法进行积分、排序和归一化操作
            mean, std_dev, iqr, lower_bound, upper_bound, outliers, outliers_index,sorted_distribution_list, sorted_subfolder_name_list,sorted_integrals,sotred_all_match_confidence,sorted_match_kp0,sorted_match_kp1=analysis.integrate_and_sort()
            self.subui_cmd.append_text("> 分析完成")   
            self.subui_cmd.cmdButton.setText("完成")
            self.subui_cmd.cmdButton.setEnabled(False)
            
            self.subui_cmd.clear_cmdshow()     
            self.subui_cmd.append_text("> 输出结果:")
            self.subui_cmd.append_text(f"> \t平均值: {mean}")
            self.subui_cmd.append_text(f"> \t标准差: {std_dev}")
            self.subui_cmd.append_text(f"> \t均值加两倍标准差: {mean+2*std_dev}")
            self.subui_cmd.append_text(f"> \t四分位距: {iqr}")
            self.subui_cmd.append_text(f"> \t四分位距下界: {lower_bound}")
            self.subui_cmd.append_text(f"> \t四分位距上界: {upper_bound}")
            self.subui_cmd.append_text(f"> \t离群值: {outliers}")
            for i in range(len(outliers_index)):
                print(f">\t离群值索引: {outliers_index[i]}, 对应的对象文件名: {sorted_subfolder_name_list[outliers_index[i]]}")
                self.subui_cmd.append_text(f"> \t离群值索引: {outliers_index[i]}, 对应的对象文件名: {sorted_subfolder_name_list[outliers_index[i]]}")

            max_match_confidence=None
            max_outlier_name=None
            max_match_kp0=None
            max_match_kp1=None
            # 记录离群值中最大和第二大的值，求出差值，然后将差值除以最大值，如果大于50%,则输出信息：可信的对象名称
            if len(outliers) >= 2:
                max_outlier = max(outliers)
                second_max_outlier = outliers[len(outliers)-2]
                difference = max_outlier - second_max_outlier
                if difference / max_outlier > 0.5:
                    max_outlier_index = np.where(sorted_integrals == max_outlier)[0][0]
                    # second_max_outlier_index = outliers.index(second_max_outlier)
                    max_outlier_name = sorted_subfolder_name_list[max_outlier_index]
                    # second_max_outlier_name = sorted_subfolder_name_list[second_max_outlier_index]
                    self.subui_cmd.append_text(f"> \t可信的对象名称: {max_outlier_name}")
                    max_match_confidence=sotred_all_match_confidence[max_outlier_index]
                    basename=os.path.basename(max_outlier_name)
                    image0_fp=os.path.join('./features_data_base/',basename,'image',f'{basename}.jpg')
                    max_match_kp0=sorted_match_kp0[max_outlier_index]
                    max_match_kp1=sorted_match_kp1[max_outlier_index]
                    
            else:
                # 直接输出离群对象名称
                self.subui_cmd.append_text(f"> \t可信的对象名称: {sorted_subfolder_name_list[0]}")
                max_match_confidence=sotred_all_match_confidence[len(sorted_subfolder_name_list)-1]
                max_outlier_name=sorted_subfolder_name_list[len(sorted_subfolder_name_list)-1]
                basename=os.path.basename(max_outlier_name)
                image0_fp=os.path.join('./features_data_base/',basename,'image',f'{basename}.jpg')
                max_match_kp0=sorted_match_kp0[len(sorted_subfolder_name_list)-1]
                max_match_kp1=sorted_match_kp1[len(sorted_subfolder_name_list)-1]
                print(image0_fp)
                
            
            
            if max_match_confidence is not None:
                self.VisualizeDialog=QtWidgets.QDialog()
                self.visualize_dialog = GraphVisualizationDialog(self.VisualizeDialog,max_match_confidence,image0_fp,self.final_file_name_image1,output_path,max_match_kp0,max_match_kp1,max_outlier_name)
                # self.subui_cmd.append_text(f"> \t平均值: {mean}")
                # self.subui_cmd.append_text(f"> \t标准差: {std_dev}")
                # self.subui_cmd.append_text(f"> \t均值加两倍标准差: {mean+2*std_dev}")
                # self.subui_cmd.append_text(f"> \t四分位距: {iqr}")
                # self.subui_cmd.append_text(f"> \t四分位距下界: {lower_bound}")
                # self.subui_cmd.append_text(f"> \t四分位距上界: {upper_bound}")
                # self.subui_cmd.append_text(f"> \t离群值: {outliers}")
                # for i in range(len(outliers_index)):
                #     print(f">\t离群值索引: {outliers_index[i]}, 对应的文件名: {sorted_subfolder_name_list[outliers_index[i]]}")
                #     self.subui_cmd.append_text(f"> \t离群值索引: {outliers_index[i]}, 对应的文件名: {sorted_subfolder_name_list[outliers_index[i]]}")
                self.visualize_dialog.append_text(f"平均值: {mean}")
                self.visualize_dialog.append_text(f"标准差: {std_dev}")
                self.visualize_dialog.append_text(f"均值加两倍标准差: {mean+2*std_dev}")
                self.visualize_dialog.append_text(f"四分位距: {iqr}")
                self.visualize_dialog.append_text(f"四分位距下界: {lower_bound}")
                self.visualize_dialog.append_text(f"四分位距上界: {upper_bound}")
                self.visualize_dialog.append_text(f"离群值: {outliers}")
                for i in range(len(outliers_index)):
                    self.visualize_dialog.append_text(f"离群值索引: {outliers_index[i]}, 对应的对象文件名: {sorted_subfolder_name_list[outliers_index[i]]}")
                self.visualize_dialog.append_text("\n\n\n")
                self.visualize_dialog.append_text(f"可信的对象名称: {max_outlier_name}")
                print(output_path)
                self.VisualizeDialog.show()
                self.VisualizeDialog.exec()
            else:
                self.subui_cmd.append_text(f"> \t可信的对象名称: 无")
            

        def show_preprocessed_image(self, temp_path):
            self.image_path=temp_path
            pixmap = QtGui.QPixmap(temp_path)
            if not pixmap.isNull():
                scene = QtWidgets.QGraphicsScene()
                pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                scene.addItem(pixmap_item)
                self.graphicsView_3.setScene(scene)
                self.graphicsView_3.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)
                
                
        if orderindex==1:
            load_from_raw(self)
        elif orderindex==2:
            load_from_preprocessed(self) 
        elif orderindex==3:
            load_from_camera(self)
        elif orderindex==4:
            detect_and_match(self,self.image1)
        
        
    def OMNI2(self,orderindex):
        # self.image_path=None
        
        def find_preprocessed_image(self):
            # self.image0=None
            
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setNameFilter("Images (*.jpg *.png)")
            file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    self.image_path = file_paths[0]
                    # 调用 show_preprocessed_image 函数显示图片
                    show_preprocessed_image(self, self.image_path)
                    # self.image0=Image.open(self.image_path).convert("RGB")
                    # 改用opencv打开
                    self.image0 = cv2.imread(self.image_path)
            else:
                print("用户取消了文件选择。")
                self.image_path = None  # 确保 image_path 为空
                self.image0 = None  # 确保 image0 为空
            # show_preprocessed_image(self,self.image_path)
            
        def use_camera(self):
            
            # self.SubWindow_cmd = QtWidgets.QDialog()
            # self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
            
            
            #  # 初始化 QInputDialog
            # input_dialog = QInputDialog()
            # input_dialog.setWindowTitle('输入文件名')
            # input_dialog.setLabelText('请输入要保存的文件名:')
            # input_dialog.setTextValue('')

            # # 显示对话框并获取用户输入
            # if input_dialog.exec_() == QDialog.Accepted:
            #     file_name = input_dialog.textValue()
            #     if not file_name:
            #         print("未输入有效文件名，操作取消。")
            #         return
            # else:
            #     print("用户取消了操作。")
            #     return
            # base_filename=file_name
            # base_filename=os.path.join('./res/',base_filename)
            # if not os.path.exists(base_filename):
            #     os.makedirs(base_filename, exist_ok=True)
            
            # self.SubWindow_cmd.show()    
            # self.subui_cmd.cmdButton.setText("正在分析...")
            # self.subui_cmd.append_text("> 正在打开摄像头并处理图像...")
                
            # self.image0,self.final_file_name=take_photo(base_filename,12)
            # self.subui_cmd.append_text("> 图像处理完成")
            # self.subui_cmd.cmdButton.setText("完成")
            # show_preprocessed_image(self,self.final_file_name)
            
            # pass
            
            global Camera_selected_folder
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)
            self.SubWindow_cmd.show()
            self.subui_cmd.cmdButton.setText("正在加载...")
            self.subui_cmd.append_text("> 正在打开摄像头并处理图像...")

            self.TakePhoto_ui = QtWidgets.QDialog()
            self.takephoto_ui_ui = CameraDialog(self.TakePhoto_ui)
            
           # 连接信号到槽函数
            # self.takephoto_ui_ui.folder_selected.connect(self.handle_selected_folder)
            
            self.TakePhoto_ui.show()
            self.SubWindow_cmd.close()
            foldername=Camera_selected_folder
            print(foldername)
            
            # 等待 TakePhoto_ui 关闭
            self.TakePhoto_ui.exec_()
            
            foldername = Camera_selected_folder
            print(foldername)
            if foldername:
                # ... existing code ...
                lastfolder = os.path.basename(foldername)
                # 修正路径拼接，原代码存在多余的逗号
                self.image_path = os.path.join(foldername, f"{lastfolder}.jpg")
                try:
                    self.image0 = Image.open(self.image_path).convert("RGB")
                    show_preprocessed_image(self, self.image_path)
                except FileNotFoundError:
                    print(f"未找到文件: {self.image_path}")
            else:
                print("未选择有效文件夹路径，无法处理图像。")
                
        def find_raw_images_and_process(self):
            
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
           
            self.subui_cmd.cmdButton.setText("正在加载...")
            self.subui_cmd.append_text("> 等待用户输入图像...")
            self.file_dialog = QtWidgets.QFileDialog()
            self.file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)  # 设置为选择文件夹模式
            self.image1 = None


            self.subui_cmd.cmdButton.setText("正在加载...")
            max_photos = 12
            final_file_name = None

            if self.file_dialog.exec_():
                self.SubWindow_cmd.show()
                self.subui_cmd.append_text("> 正在处理图像...")
                folder_name = self.file_dialog.selectedFiles()[0]
                existing_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name)]
                all_files_exist = True
                for i in range(1, max_photos + 1):
                    target_filename = os.path.join(folder_name, f"{i}.jpg")
                    if target_filename not in existing_files:
                        all_files_exist = False
                        print(f"缺少照片文件 {target_filename}，将调用摄像头拍照。")
                        break

                if all_files_exist:
                    self.image0, final_file_name = combine_photos(folder_name, max_photos)
                    self.image_path=final_file_name
                    self.subui_cmd.append_text("> 图像处理完成")
            else:
                print("用户取消了文件夹选择。")
                self.subui_cmd.append_text("> 已取消")
                # 可以在这里添加更多处理逻辑，比如重置相关变量等
                self.image0 = None
                final_file_name = None
            


            show_preprocessed_image(self,final_file_name)
                # 使用 QTimer.singleShot 延迟调用 detect_and_match
                # QtCore.QTimer.singleShot(0, lambda: detect_and_match(self, image1))
            self.subui_cmd.cmdButton.setText("完成")
            # show_preprocessed_image(image_path)
        def detect_and_load(self,image0):
            
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
            
            scaled_image=image0
            image0=np.array(scaled_image)
            image0=np.array(Image.open(self.image_path).convert('RGB'))
            
            
            # 初始化 QInputDialog
            input_dialog = QInputDialog()
            input_dialog.setWindowTitle('输入文件名')
            input_dialog.setLabelText('请输入要保存的文件名:')
            input_dialog.setTextValue('')

            # 显示对话框并获取用户输入
            if input_dialog.exec_() == QDialog.Accepted:
                file_name = input_dialog.textValue()
                if not file_name:
                    print("未输入有效文件名，操作取消。")
                    return
            else:
                print("用户取消了操作。")
                return
             
            self.SubWindow_cmd.show()
            print(f"")
            print("> 查找匹配点...")
            self.subui_cmd.append_text("> 查找匹配点...")
            start = time.time()
            self.og.SaveFeatures(image0, 2, file_name)

            # 创建保存图像的目录
            image_save_dir = os.path.join('./features_data_base', file_name, 'image')
            self.subui_cmd.append_text(f"> 创建保存图像的目录: {image_save_dir}")
            os.makedirs(image_save_dir, exist_ok=True)

            # 保存图像
            image0_fp = self.image_path
            image_save_path = os.path.join(image_save_dir, os.path.basename(image0_fp))
            Image.fromarray(image0).save(image_save_path)
            # 改成用opencv保存图片
            # cv2.imwrite(image_save_path, image0)

            print(f"> \t耗时 {time.time() - start} 秒。")
            self.subui_cmd.append_text(f"> \t耗时 {time.time() - start} 秒。")
            print(f"> 已将特征数据保存到数据库中:./features_data_base/{file_name}")
            self.subui_cmd.append_text(f"> 已将特征数据保存到数据库中:./features_data_base/{file_name}")
            print(f"> 已将图像保存到: {image_save_path}")
            self.subui_cmd.append_text(f"> 已将图像保存到: {image_save_path}")
            self.subui_cmd.cmdButton.setText("完成")
            self.subui_cmd.cmdButton.setEnabled(False)
            # show_cmd()
            pass
        
        def show_preprocessed_image(self,temp_path):
            
            self.image_path=temp_path
            pixmap = QtGui.QPixmap(temp_path)
            if not pixmap.isNull():
                scene = QtWidgets.QGraphicsScene()
                pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                scene.addItem(pixmap_item)
                self.graphicsView_5.setScene(scene)
                self.graphicsView_5.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)
            pass
        def show_cmd(self):
            pass
        
        
        if orderindex==1:
            find_preprocessed_image(self)
        elif orderindex==2:
            use_camera(self)
        elif orderindex==3:
            find_raw_images_and_process(self)
        elif orderindex==4:
            detect_and_load(self,self.image0)
        
        
    def OMNI3(self,order_index):

        """
            原omni.py中orde==3功能实现
        """
        # self.confidence = 0
        self.thread = None
        def getimages1_OMNI3(self):
            """
            打开文件选择对话框，让用户选择 JPG 或 PNG 文件，
            并在 graphicsView 中显示图像，在 label_6 中显示文件路径。
            """
            self.file_dialog = QtWidgets.QFileDialog()
            self.file_dialog.setNameFilter("Images (*.jpg *.png)")
            self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if self.file_dialog.exec_():
                file_paths = self.file_dialog.selectedFiles()
                if file_paths:
                    self.image_path1_OMNI3 = file_paths[0]
                    # self.label_6.setText(self.image_path1_OMNI3)
                    # 加载图像并在 graphicsView 中显示
                    pixmap = QtGui.QPixmap(self.image_path1_OMNI3)
                    if not pixmap.isNull():
                        scene = QtWidgets.QGraphicsScene()
                        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                        scene.addItem(pixmap_item)
                        self.graphicsView.setScene(scene)
                        self.graphicsView.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)

        def getimages2_OMNI3(self):
            """
            打开文件选择对话框，让用户选择 JPG 或 PNG 文件，
            并在 graphicsView_2 中显示图像，在 label_7 中显示文件路径。
            """
            self.file_dialog = QtWidgets.QFileDialog()
            self.file_dialog.setNameFilter("Images (*.jpg *.png)")
            self.file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if self.file_dialog.exec_():
                file_paths = self.file_dialog.selectedFiles()
                if file_paths:
                    self.image_path2_OMNI3 = file_paths[0]
                    # self.label_7.setText(self.image_path2_OMNI3)
                    # 加载图像并在 graphicsView_2 中显示
                    pixmap = QtGui.QPixmap(self.image_path2_OMNI3)
                    if not pixmap.isNull():
                        scene = QtWidgets.QGraphicsScene()
                        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                        scene.addItem(pixmap_item)
                        self.graphicsView_2.setScene(scene)
                        self.graphicsView_2.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)
                        
        def get_current_value(self):
            """
            获取当前滑块的值并显示在 label_8 中。
            """
            current_value = (self.horizontalSlider.value())/1000
            self.label_8.setText(str(current_value))
            self.confidence = current_value
            
        def detect_OMNI3(self):
        # 直接在主线程中创建和显示 SubWindow_cmd
            self.SubWindow_cmd = QtWidgets.QDialog()
            self.subui_cmd = Output_cmd_graph_Dialog(self.SubWindow_cmd)  # 假设这里应该是 Output_cmd_graph_Dialog
            self.SubWindow_cmd.show()

            # def detection_thread():
            image0_fp = self.image_path1_OMNI3
            image1_fp = self.image_path2_OMNI3
            confidence = self.confidence

            # 验证图像文件路径是否存在且为文件

            # 将cmdButton中的txt设置为"正在检测中..."
            self.subui_cmd.cmdButton.setText("正在检测中...")

            # 加载图像
            self.subui_cmd.append_text("> 加载图像...")
            image0 = np.array(Image.open(image0_fp).convert("RGB"))
            image1 = np.array(Image.open(image1_fp).convert("RGB"))

            # 执行推理
            self.subui_cmd.append_text("> 查找匹配点...")
            start = time.time()
            match_kp0, match_kp1, match_confidences = self.og.FindMatches_two_images(image0, image1, 3, None)
            num_matches = match_kp0.shape[0]
            self.subui_cmd.append_text(f"> \t找到 {num_matches} 个匹配点。")
            self.subui_cmd.append_text(f"> \t耗时 {time.time() - start} 秒。")

            # 根据置信度过滤匹配点
            self.subui_cmd.append_text("> 过滤匹配点...")
            match_threshold = confidence
            keep_idx = []
            for i in range(match_kp0.shape[0]):
                if match_confidences[i] > match_threshold:
                    keep_idx.append(i)
            num_filtered_matches = len(keep_idx)
            match_kp0 = match_kp0[keep_idx]
            match_kp1 = match_kp1[keep_idx]
            match_confidences = match_confidences[keep_idx]

            self.subui_cmd.append_text(f"> \t{num_filtered_matches}/{num_matches} 个匹配点的置信度高于阈值 {match_threshold}")
            self.subui_cmd.append_text("> 可视化匹配点...")
            viz = utils.visualize_matches(
                image0,
                image1,
                match_kp0,
                match_kp1,
                np.eye(num_filtered_matches),
                match_labels=None,
                show_keypoints=True,
                highlight_unmatched=True,
                title=f"{num_filtered_matches} matches",
                line_width=2,
            )
            output_path = f"./result/OMNIoutput_confidence_{confidence}.png"
            plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
            plt.axis("on")
            plt.imshow(viz)
            plt.imsave(output_path, viz)
            self.subui_cmd.append_text(f"> 已将置信度 {confidence} 的可视化结果保存到 {output_path}")
            
            self.subui_cmd.cmdButton.setText("已完成")
            
            # 创建一个新的窗口来显示图像
            self.image_window = QtWidgets.QMainWindow()
            self.image_scene = QtWidgets.QGraphicsScene()
            image_view = QtWidgets.QGraphicsView(self.image_scene)
            self.image_window.setCentralWidget(image_view)

            # 加载图像并显示
            pixmap = QtGui.QPixmap(output_path)
            if not pixmap.isNull():
                pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
                self.image_scene.addItem(pixmap_item)
                image_view.setScene(self.image_scene)
                image_view.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)

            # 窗口大小改变时更新图像显示
            def resize_event(event):
                image_view.fitInView(pixmap_item, QtCore.Qt.KeepAspectRatio)
                event.accept()

            self.image_window.resizeEvent = resize_event

            # 显示图像窗口
            self.image_window.show()
            


            # 显示可视化结果
            # self.subui.ShowImage(output_path)


            # 创建一个线程，用来启动 detection_thread 函数
            # self.thread = QThread()
            # self.thread.started.connect(detection_thread)
            # self.thread.start()


        if order_index == 1:
            getimages1_OMNI3(self)
        elif order_index == 2:
            getimages2_OMNI3(self)
        elif order_index == 3:
            detect_OMNI3(self)
        else:
            get_current_value(self)
            
    def About(self):
        pass

    

if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建启动界面
    StartUI = QtWidgets.QDialog()
    startui = StartUI_Dialog(StartUI)
    StartUI.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    StartUI.show()
    startui.append_text("正在加载Omniglue及其子模块")
    
    # 启动模型加载线程
    model_thread = ModelLoadingThread()
    model_thread.start()

    # 循环检查模型是否加载完成
    while True:
        if og is not None:
            # 模型加载完成，关闭子线程
            model_thread.quit()
            model_thread.wait()
            
            startui.append_text("正在渲染界面")
            
            # 暂停 1 秒
            time.sleep(1)

            # 关闭启动界面
            StartUI.close()

            # 暂停 1 秒
            time.sleep(1)

            # 创建主界面
            Dialog_MAin = QtWidgets.QDialog()
            ui = MainDialog(Dialog_MAin)
            Dialog_MAin.show()
            break

        # 处理界面事件，避免界面冻结
        app.processEvents()

    sys.exit(app.exec_())

