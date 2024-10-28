import sys
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import cv2
from tqdm import tqdm
import wandb
from model import *
from utils import *

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    n_classes = {
        'MNIST': 10,
        'cifar10': 10,
        'svhn': 10,
        'fmnist': 10,
        'celeba': 2,
        'cifar100': 100,
        'tinyimagenet': 200,
        'femnist': 26,
        'emnist': 47,
        'xray': 2
    }.get(args.dataset, 10)  # Default to 10 classes if dataset is unknown

    for net_i in range(n_parties):
        if args.normal_model:
            net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        else:
            net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs) if args.use_project_head \
                else ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)

        nets[net_i] = net.to(device)  # Move net to the specified device

    model_meta_data = [v.shape for k, v in nets[0].state_dict().items()]
    layer_type = list(nets[0].state_dict().keys())

    return nets, model_meta_data, layer_type

def remove_module_prefix(state_dict):
    if isinstance(state_dict, torch.nn.DataParallel):
        state_dict = state_dict.module.state_dict()
    return {key[len('module.'):]: value if key.startswith('module.') else value for key, value in state_dict.items()}


# Argument parsing and initialization
parser = argparse.ArgumentParser()
parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
parser.add_argument('--load_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10_resnet50\MCFL\DP\globalmodel70.pth', help='the model to load as global model')
parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
parser.add_argument('--use_project_head', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
args = parser.parse_args()

# Data loading
train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size)
backdoor_train_dl_global, backdoor_test_dl_global, backdoor_train_ds_global, backdoor_test_ds_global = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size, backdoor=True)

sample_data, _ = next(iter(backdoor_test_dl_global))
sample_data = sample_data.to(device=args.device)  # Move sample_data to the correct device


# Network initialization
net_configs = args.net_config
args.n_parties = 1
nets, model_meta_data, layer_type = init_nets(net_configs, args.n_parties, args, device=args.device)

# Model loading
if args.load_model_file:
    try:
        nets[0].load_state_dict(torch.load(args.load_model_file))
        global_model = copy.deepcopy(nets[0])
    except:
        global_model = nn.DataParallel(nets[0])
        global_model.load_state_dict(remove_module_prefix(torch.load(args.load_model_file)))

model = global_model

# 现在假设你已经准备好训练好的模型和预处理输入了

grad_block = []	# 存放grad图
feaure_block = []	# 存放特征图

# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 获取特征层的函数
def farward_hook(module, input, output):
    feaure_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
    grads = grads.reshape([grads.shape[0], -1])
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=1)	
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]	# 特征图加权和
    cam = np.maximum(cam, 0)
    cam_max = cam.max() if cam.max() > 0 else 1e-8  # 避免除以零
    cam = cam / cam_max  # 归一化
    cam = cv2.resize(cam, (32, 32))
    
    # cam.dim=2 heatmap.dim=3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)	# 伪彩色
    cam_img = 0.7 * heatmap + 0.7 * img

    cv2.imwrite("0_2.jpg", cam_img)



model.features[6][2].conv3.register_forward_hook(farward_hook)
model.features[6][2].conv3.register_backward_hook(backward_hook)

#model.l2.register_forward_hook(farward_hook)
#model.l2.register_backward_hook(backward_hook)


# forward 
# 在前向推理时，会生成特征图和预测值
sample_tensor = sample_data[1, :, :, :].clone().detach()  # 转换为 Tensor

# 读取图像
img = Image.open("./0_1.png")

# 定义转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为 32x32 像素
    transforms.ToTensor()  # 将图像转换为张量，并将值归一化到 [0, 1]
])

# 应用转换
img_tensor = transform(img)

output = model(sample_tensor.unsqueeze(0))  # 增加一个维度以符合模型输入要求
print(output)
# 确保 output 是一个 Tensor
if isinstance(output, tuple):
    output = output[2]  # 只获取第一个输出，假设这是你需要的结果
#print(model)
print(output.shape)
max_idx = np.argmax(output.cpu().data.numpy())
print("predict:{}".format(max_idx))

# backward
model.zero_grad()
# 取最大类别的值作为loss，这样计算的结果是模型对该类最感兴趣的cam图
# 根据 output 的维度获取 class_loss
if output.dim() == 1:  # 一维 Tensor
    class_loss = output[max_idx]
else:
    class_loss = output[0, max_idx]  # 如果是二维 Tensor，请确保维度是正确的
class_loss.backward()	# 反向梯度，得到梯度图

# grads
grads_val = grad_block[0].cpu().data.numpy().squeeze()
fmap = feaure_block[0].cpu().data.numpy().squeeze()
# 我的模型中
# grads_cal.shape=[1280,2,2]
# fmap.shape=[1280,2,2]

raw_img = cv2.imread("./0_1.png")
# save cam
cam_show_img(raw_img, fmap, grads_val)
