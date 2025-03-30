#=====================全局参数==================================
cuda = False                 #是否有cuda
data_type = 'vehicle'       #数据集类型，['vehicle','mpii']
#=======================数据集的设置==============================
vehicle_is_test = False

vehicle_ckpt_save_dir = 'data/ckpt'
vehicle_train_txt_path = 'data/train.txt'    #训练或者测试的txt路径
# vehicle_valid_txt_path = 'data/valid.txt'
vehicle_ckpt_resume_path = 'data/ckpt/last_ckpt.pth'   #断点续训的pth路径，如果有这个权重就可以使用
vehicle_output_dir = 'data/output'         #生成可视化数据的目录

#===============================模型的设置=======================
model_style = 'pytorch'     #['caffe','pytorch']
num_layers = 18             #[50,18,24,101,152]
num_keypoints = 35         #关键点数量
use_target_weight = False   #是否计算隐藏点与否的loss信息
#========================日志===================================
test_output_dir = vehicle_output_dir
pre_train = ""              #backbone的权重
#=========================优化器，学习率===========================
optimizer_method = "adam"   #[sgd,adam],adam比sgd效果好
lr_method = 'step'          #[step,adaptive,multiStep]
train_lr_step = [90, 110]   #多step学习率的间隔
init_lr = 0.001             #初始学习率
#=======================训练=====================================
start_epoch = 0             #开始epoch
total_epoch = 10           #总共的epoch
train_batch_size = 2       #训练的batch size，根据硬件条件修改
val_batch_size = 2        #验证的batchsize，根据硬件条件修改
continue_train = False       #是否继续训练，第一次训练设置为False
img_size = 256              #输入模型的图片尺寸
heatmap_size = 64           #heatmap尺寸
ckpt_save_dir =vehicle_ckpt_save_dir
ckpt_resume_path = vehicle_ckpt_resume_path

#================demo.py的参数===================================
demo_mode_path = 'data/ckpt/last_ckpt.pth'
demo_images_path = 'data/try'
demo_images_result_path = 'data/try_result'

