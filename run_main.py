from random import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog,QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import QPixmap,QImage,QPainter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from openni import openni2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import sys
import cv2
import os
import pandas as pd
import math
import threading
from PyQt5.QtCore import QTimer
import torch
import numpy as np
import torchvision.transforms as transforms
from Models.pose_resnet import get_pose_net


import config as cfg
from mainwin import Ui_MainWindow
import show_points as sp
import helpwin
from box import ImageBox
from box_ana import ImageBoxana
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#模型
class RealTimeKeypointDetector:
    def __init__(self, model_path):

        # 初始化硬件配置
        self.device = torch.device("cuda" if cfg.cuda else "cpu")

        # 模型加载与优化
        self.model = get_pose_net(
            is_train=False,
            style=cfg.model_style,
            num_keypoint=cfg.num_keypoints
        ).to(self.device)

        # 加载权重并自动适配单/多GPU模式
        state_dict = torch.load(model_path, map_location=self.device)
        self._load_weights(state_dict)

        # 启用推理模式
        self.model.eval()

        # 图像预处理管道
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 可视化配置
        self.keypoint_radius = 5 #关键点大小
        self.keypoint_color = (0, 0, 255)  #关键点颜色

    def _load_weights(self, state_dict):
        """模型权重加载"""
        if 'net' in state_dict:
            weights = state_dict['net']
            # 自动去除模块前缀（兼容多GPU训练保存的模型）
            from collections import OrderedDict
            new_weights = OrderedDict()
            for k, v in weights.items():
                name = k[7:] if k.startswith('module.') else k
                new_weights[name] = v
            self.model.load_state_dict(new_weights)
        else:
            self.model.load_state_dict(state_dict)

    def preprocess(self, frame):
        """预处理帧数据"""
        # 保留原始尺寸用于后处理
        self.orig_h, self.orig_w = frame.shape[:2]

        # 调整尺寸并标准化
        resized = cv2.resize(frame, (cfg.img_size, cfg.img_size))
        tensor = self.transform(resized)
        return tensor.unsqueeze(0).to(self.device)  # 添加批次维度

    def detect(self, tensor):
        """执行模型推理"""
        with torch.no_grad():
            heatmaps = self.model(tensor)
        return heatmaps.squeeze().cpu().numpy()

    def postprocess(self, heatmaps):
        """解析热力图到原始图像坐标"""
        keypoints = []
        for i in range(cfg.num_keypoints):
            hm = heatmaps[i]
            # 寻找热力图最大值位置
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            # 转换到输入尺寸坐标系
            x = x * (cfg.img_size / cfg.heatmap_size)
            y = y * (cfg.img_size / cfg.heatmap_size)
            # 映射回原始图像尺寸
            x_ratio = self.orig_w / cfg.img_size
            y_ratio = self.orig_h / cfg.img_size
            keypoints.append((int(x * x_ratio), int(y * y_ratio)))
        return keypoints

    def visualize(self, frame, keypoints):
        """可视化关键点"""
        # 绘制关键点
        for i,(x, y) in keypoints.items():
            cv2.circle(frame, (x, y), self.keypoint_radius,
                       self.keypoint_color, -1)
            cv2.putText(frame, f'{i}',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame

    def process_frame(self, frame):
        """完整处理流程"""
        kp_3d = []
        tensor = self.preprocess(frame)#预处理
        heatmaps = self.detect(tensor)#检测
        kp = self.postprocess(heatmaps)#还原原始坐标
        keypoints = {str(index + 1): value for index, value in enumerate(kp)}
        re_frame = self.visualize(frame, keypoints)
        return keypoints,re_frame ,kp#可视化


class NewWindow(QMainWindow):
    def __init__(self,parent = None):
        super(NewWindow,self).__init__(parent)
        self.setWindowTitle("帮助")
        self.setFixedSize(500, 700)
        label = QLabel("仅支持打开\n[.mp4, .mkv, .MOV, .avi,.bmp, .jpg, .png, .gif]文件", self)
        label.setGeometry(0, -200, 500, 700)
        label.setWordWrap(True)
        self.setStyleSheet("QLabel{font-size:50px;}")
        label.setAutoFillBackground(True)
        self.show()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.x = None
        self.y = None
        self.points = None
        self.actionopen.triggered.connect(self.openClick)  # 打开文件
        self.actionsave.triggered.connect(self.saveClick)  # 保存文件
        self.cameraopen.triggered.connect(self.openCamera)  # 打开摄像头
        self.cameraclose.triggered.connect(self.closeCamera)  # 关闭摄像头
        self.anast.triggered.connect(self.analyseSt)  # 开始分析
        self.anaclose.triggered.connect(self.analyseClose)  # 结束分析
        self.anasave.triggered.connect(self.analyseSave)  # 保存分析结果
        self.helpaction.triggered.connect(self.help)  # 帮助
        self.pushButton.clicked.connect(self.startStop)  # 开始/暂停按钮
        self.pushButton_2.clicked.connect(self.Close)  # 结束按钮
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # 播放图标
        self.pushButton_2.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))  # 停止图标
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.file_path = None
        self.playing = False
        self.camera_on = False
        self.video_on = False
        self.num = 0
        self.mousepos = (0,0)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setGeometry(QRect(10, 30, 1080, 720))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QWidget()
        self.box = ImageBox()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, self.box.width(), self.box.height()))
        self.scrollAreaWidgetContents.setMinimumSize(QSize(1080,720))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.addWidget(self.box, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.setmousepos)
        self.timer.start(1)  # 每隔1000毫秒调用一次

        self.xlabel.setText('x:0')
        self.ylabel.setText('y:0')
        self.dptlabel.setText('dpt:0')
        self.label_filename.setText(f'file:None')

        self.x = np.array([0])
        self.y = np.array([0])
        self.z = np.array([0])

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.verticalLayout.addWidget(self.canvas)


        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')
        self.axes.grid(True)
        self.axes.view_init(elev=20, azim=30)
        #self.colors = cm.viridis(np.linspace(0, 1, 35))

        # 初始化散点图
        self.scatter = self.axes.scatter(
            self.x, self.y, self.z,
            c='red', s=50, depthshade=True
        )


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_frame)
        

        self.points = {}
        self.points3d = []

        self.detector = RealTimeKeypointDetector(cfg.demo_mode_path)
        self.cam_lock = QMutex()
        self.num_points = 35

    def openClick(self):
        self.video_type = None
        self.file_path = QFileDialog.getOpenFileName(self, "打开文件", "", "All Files(*)")[0]
        video_type = [".mp4", ".mkv", ".MOV", ".avi"]
        img_type = [".bmp", ".jpg", ".png", ".gif"]
        self.label_filename.setText(f'file:{self.file_path}')
        for vdi in img_type:
            if vdi in self.file_path:
                self.pushButton.setEnabled(False)
                self.pushButton_2.setEnabled(False)
                self.playing = False
                self.camera_on = False
                self.video_on = False
                self.video_type = False

                img = QPixmap(self.file_path)
                self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, img.width(), img.height()))
                self.box.set_image(self.file_path)

        for vdi in video_type:
            if vdi in self.file_path:
                self.video_on = True
                self.video = cv2.VideoCapture(self.file_path)
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
                self.playing = True
                self.video_type = True
                self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # 获取输入视频的高度
                self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 # 获取视频帧数
                self.video_frame_number = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                        # 获取输入视频的帧率
                self.video_frame_rate = int(self.video.get(cv2.CAP_PROP_FPS))
                num = self.video_frame_number
                self.all_frame = []
                while num:
                    ret, frame = self.video.read()
                    self.all_frame.append(frame)
                    num = num -1
                self.videoplay()
                break

    def videoplay(self):
            # 将图片转换为 Qt 格式
            # QImage:QImage(bytes,width,height,format)
        while  self.playing:
            self.box.show_image(self.all_frame[self.num],self.video_width, self.video_height, 3 * self.video_width,)
            cv2.waitKey(self.video_frame_rate)
            self.num +=1
            if self.num == self.video_frame_number:
                self.playing = False
        self.video.release()  # 释放资源

    def saveClick(self):
        print("保存文件")

    def openCamera(self):
        """安全启动深度相机"""
        if self.cam_lock.tryLock(1000):
            try:
                openni2.initialize()
                self.dev = openni2.Device.open_any()
                self.depth_stream = self.dev.create_depth_stream()
                self.dev.set_image_registration_mode(True)
                self.depth_stream.start()
                self.cap = cv2.VideoCapture(0)

                self.camera_on = True
                self.timer.start(33)  # ~30fps



            finally:
                self.cam_lock.unlock()

    def update_camera_frame(self):
        if self.cam_lock.tryLock(10):  # 非阻塞锁
            try:
                depth_frame = self.depth_stream.read_frame()
                dframe_data = np.array(depth_frame.get_buffer_as_triplet()).reshape([480, 640, 2])

                # 处理深度数据
                dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
                dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32') * 255
                self.dpt = dpt1 + dpt2

                # 读取彩色帧
                ret, color_frame = self.cap.read()


                if ret:

                    _,frame,kp = self.showpoint_camera(color_frame)
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                    self.box.show_camera(q_img)
                    # 转换特征点为三维坐标
                    if len(kp) >= 35:
                        self.process_keypoints(kp, color_frame)
            finally:
                self.cam_lock.unlock()
        """定时器回调：获取并处理帧数据"""

    def process_keypoints(self, kp, frame):
        """将特征点转换为三维坐标"""
        # 转换为三维坐标 (x, y从特征点获取，z从深度图获取)
        valid_points = []
        for (x, y) in kp:
            if 0 <= x < 640 and 0 <= y < 480:  # 假设深度图分辨率640x480
                z = self.dpt[y, x]  # 注意OpenCV是(y,x)坐标
                valid_points.append([x, y, z])

        # 更新三维点数据
        if len(valid_points) == self.num_points:
            self.points = np.array(valid_points)
            self.update_3d_plot()

    def update_3d_plot(self):
        """安全更新三维散点图"""
        # 分离坐标分量
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        # 更新散点位置
        self.scatter._offsets3d = (x, y, z)

        # 自动调整坐标范围
        self.axes.set_xlim([x.min() - 50, x.max() + 50])
        self.axes.set_ylim([y.min() - 50, y.max() + 50])
        self.axes.set_zlim([z.min() - 50, z.max() + 50])

        # 请求重绘
        self.canvas.draw_idle()




    def on_timeout(self):
        self.ax.cla()



    def closeCamera(self):
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.playing = False
        self.box.set_image('background/white.png')



    def analyseSt(self):
        if self.file_path:
            print(2)
            self.points = self.showpoint(self.file_path)
            print(1)
            self.box.add_point(self.points)


    def analyseClose(self):
        print("结束分析")
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.playing = False
        img = QPixmap('background/white.png')
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, img.width(), img.height()))
        self.box.set_image('background/white.png')
        self.file_path = None
        self.ax.cla()

    def analyseSave(self):
        if self.file_path:
            folder_path = 'data/save_file'
            # 如果文件夹不存在，则创建文件夹
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            name = os.path.splitext(os.path.basename(self.file_path))[0]
            file, _ = QFileDialog.getSaveFileName(self, "保存文件", f"{folder_path}/{name}", "JPG Files (*.jpg)")

            if file:
                img, points = self.box.saveimg()
                df = pd.DataFrame(points).T
                filename = os.path.splitext(os.path.basename(file))[0]
                file_name = f'{filename}.xlsx'
                file_path = os.path.join(folder_path, file_name)
                df.to_excel(file_path, index=False)
                img.save(file)

    def startStop(self):
        if self.playing:
            self.playing = False
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.playing = True
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            if self.camera_on:
                self.openCamera()
            elif self.video_on:
                self.videoplay()

    def Close(self):
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.playing = False
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        if self.camera_on:
            img = QPixmap('background/white.png')
            self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, img.width(), img.height()))
            self.box.set_image('background/white.png')
        elif self.video_on:
            self.video.release()
            img = QPixmap('background/white.png')
            self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, img.width(), img.height()))
            self.box.set_image('background/white.png')

        # 窗口关闭按钮---事件
    def closeEvent(self, event):
        self.destroy()  # 窗口关闭销毁
        sys.exit(0)  # 系统结束推出

    def help(self):
        self.new_window = NewWindow()

    def showpoint(self,file):
        frame = cv2.imread(file, 1)
        print(11)
        keypoints,_ ,_= self.detector.process_frame(frame)
        print(22)
        return keypoints

    def showpoint_camera(self,frame):
        keypoints,re_frame ,kp= self.detector.process_frame(frame)
        return keypoints, re_frame,kp

    def setmousepos(self):
        if self.file_path:
            point_x, point_y = self.box.get_point()
            self.xlabel.setText(f'x:{point_x}')
            self.ylabel.setText(f'y:{point_y}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.move(50, 50)
    mainWindow.setFixedSize(2080,840)

    mainWindow.show()
    sys.exit(app.exec_())