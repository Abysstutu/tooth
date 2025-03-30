'''
demo.py用来根据某个权重文件,对指定目录下的所有图片生成预测的点.
'''
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import copy
import cv2
import config as cfg
from Models.pose_resnet import get_pose_net
from Utils.evaluate import accuracy
import numpy as np
import torchvision
import time
from tqdm import tqdm
from torch.utils.data import Dataset
import math



class VehicleKeyPoint(Dataset):
    def __init__(self,images_path:str,transform=None):
        print("---------------loading data ... -----------------------")
        super(VehicleKeyPoint, self).__init__()
        self.is_test = True
        self.image_size = (cfg.img_size,cfg.img_size)
        self.heatmap_size = (cfg.heatmap_size,cfg.heatmap_size)
        self.sigma = 2
        self.transform = transform
        self.scale_factor = 0.25
        self.rotation_factor = 30

        self.dataset = self.get_data(images_path)  #读取数据

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rect_item = copy.deepcopy(self.dataset[index])      #从dataset中获取样本信息
        img_path = rect_item['image']                             #图像路径
        filaname = rect_item['filename']
        imgnum = rect_item['imgnum']
        joints = rect_item['joints_3d']                           #获取人体关键点坐标
        joints_vis = rect_item['joints_3d_vis']                     #权重
        image = cv2.imread(img_path,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h,w = image.shape[0],image.shape[1]

        if image is None:
            raise ValueError("Failed to read{}".format(img_path))

        s = rect_item['scale']
        c = rect_item['center']
        score = rect_item['score'] if 'score' in rect_item else 1
        r = 0

        input = cv2.resize(image,self.image_size)
        #对关键点坐标进行变换
        for i in range(cfg.num_keypoints):
            joints[i,0] = joints[i,0] * (self.image_size[0]/w)
            joints[i,1] = joints[i,1] * (self.image_size[1]/h)

        if self.transform:
            input = self.transform(input)

        #获得ground truth，热力图
        target,target_weight = self.generate_target(joints,joints_vis)
        #转换成tensor
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        meta = {
            'image':img_path,
            'filename':filaname,
            'imgnum':imgnum,
            'joints':joints,
            'joints_vis':joints_vis,
            'center':c,
            'scale':s,
            'rotation':r,
            'score':score,
        }

        return input,target,target_weight,meta

    def affine_transform(self,pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]
    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((cfg.num_keypoints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((cfg.num_keypoints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(cfg.num_keypoints):
            feat_stride = [item/self.heatmap_size[i] for i,item in enumerate(self.image_size)]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def get_data(self,images_path):
        newlines = []

        for index,name in enumerate(os.listdir(images_path)):
            full_image_name = os.path.join(images_path,name)
            h,w,c = cv2.imread(full_image_name).shape

            joint_3d = np.zeros((cfg.num_keypoints,3),dtype=np.float32)
            joints_3d_vis = np.zeros((cfg.num_keypoints,3),dtype=np.float32)

            newlines.append({
                'image':full_image_name,
                'center':np.array([w/2-1,h/2-1]),
                'scale':np.array([2,2]),
                'joints_3d':joint_3d,
                'joints_3d_vis':joints_3d_vis,
                'filename':name,
                'imgnum':index
            })
        return newlines

def sp(file):
    # ------------数据集---------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 各通道的均值
                                     std=[0.229, 0.224, 0.225])  # 各通道的标准差
    test_dataset = VehicleKeyPoint(
        images_path=cfg.demo_images_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # 这里必须设置为1
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    model = get_pose_net(is_train=False, style=cfg.model_style, num_keypoint=cfg.num_keypoints)
    # 加载模型
    model.load_state_dict(torch.load(cfg.demo_mode_path)["net"])

    # ---------放入cuda-------------
    if cfg.cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        # cudnn相关设置
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    else:
        model = model.cpu()

    model.eval()
    len_dataset = len(test_loader)
    with torch.no_grad():
        start_time = time.time()
        for i, (input, target, target_weight, meta) in enumerate(test_loader):

            if cfg.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
            else:
                input = input.cpu()
                target = target.cpu()
                target_weight = target_weight.cpu()

            # 得到预测输出
            output = model(input)
            num_images = input.size(0)  # batch_size
            print(output)
            # 计算准确率和损失
            all_acc, avg_acc, count, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            # print(pred * (cfg.img_size // cfg.heatmap_size))

        # 计算耗时
        end_time = time.time()
        print(f"Spend time:{end_time - start_time} ms,"
              f"Imagse number:{len_dataset},"
              f"Per image spend time:{(end_time - start_time) / len_dataset} ms.")

if __name__ == '__main__':
    #------------数据集---------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],#各通道的均值
                                     std=[0.229, 0.224, 0.225]) #各通道的标准差
    test_dataset = VehicleKeyPoint(
        images_path=cfg.demo_images_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,   #这里必须设置为1
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    model = get_pose_net(is_train=False, style=cfg.model_style,num_keypoint=cfg.num_keypoints)
    #加载模型
    model.load_state_dict(torch.load(cfg.demo_mode_path)["net"])

    # ---------放入cuda-------------
    if cfg.cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        # cudnn相关设置
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    else:
        model = model.cpu()

    model.eval()
    len_dataset = len(test_loader)
    with torch.no_grad():
        start_time = time.time()
        for i, (input, target, target_weight, meta) in enumerate(test_loader):

            if cfg.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
            else:
                input = input.cpu()
                target = target.cpu()
                target_weight = target_weight.cpu()

            # 得到预测输出
            output = model(input)
            num_images = input.size(0)  # batch_size
            print(output)
            # 计算准确率和损失
            all_acc, avg_acc, count, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            #print(pred * (cfg.img_size // cfg.heatmap_size))


        # 计算耗时
        end_time = time.time()
        print(f"Spend time:{end_time-start_time} ms,"
              f"Imagse number:{len_dataset},"
              f"Per image spend time:{(end_time-start_time)/len_dataset} ms.")