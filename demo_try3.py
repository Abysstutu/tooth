import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from Models.pose_resnet import get_pose_net
import config as cfg


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
        self.keypoint_radius = 8 #关键点大小
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
        for (x, y) in keypoints:
            cv2.circle(frame, (x, y), self.keypoint_radius,
                       self.keypoint_color, -1)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)

        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        return frame

    def process_frame(self, frame):
        """完整处理流程"""
        tensor = self.preprocess(frame)#预处理
        heatmaps = self.detect(tensor)#检测
        keypoints = self.postprocess(heatmaps)#还原原始坐标
        return keypoints   #可视化



def showpoint(file):
    detector = RealTimeKeypointDetector(cfg.demo_mode_path)

    frame = cv2.imread(file, 1)

    kp = detector.process_frame(frame)

    keypoints = {str(index+1): value for index, value in enumerate(kp)}

    print(keypoints)








# 使用示例

if __name__ == '__main__':

    # 初始化检测器
    detector = RealTimeKeypointDetector(cfg.demo_mode_path)

    #读取测试帧
    frame = cv2.imread(r"F:\PycharmProjects\pythonProject1\data\images\0001.jpg", 1)

    # 处理测试帧
    processed = detector.process_frame(frame)

    #显示测试帧检测结果
    cv2.imshow('Real-time Detection', processed)

    # #保存测试帧检测结果
    # cv2.imwrite("../data/try_result/1030_.jpg", processed)


