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
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value
    return new_state_dict

# Argument parsing and initialization
parser = argparse.ArgumentParser()
parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
parser.add_argument('--load_model_file', type=str, default='Y:\FRDA\model\cifar10_resnet50\FL/fedavg.pth', help='the model to load as global model')
parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
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
        nets[0].load_state_dict(torch.load(args.load_model_file, map_location=args.device))
        global_model = copy.deepcopy(nets[0])
    except:
        global_model = nn.DataParallel(nets[0])
        state_dict = torch.load(args.load_model_file, map_location=args.device)
        state_dict = remove_module_prefix(state_dict)
        global_model.load_state_dict(state_dict)

model = global_model.to(args.device)

# 准备钩子函数存储梯度和特征图
grad_block = []	# 存放grad图
feature_block = []	# 存放特征图

# 获取梯度的函数
# backward_hook 函数
def backward_hook(module, grad_input, grad_output):
    grad_block.append(grad_output[0].detach())

# 获取特征层的函数
def forward_hook(module, input, output):
    feature_block.append(output)

# 已知原图、梯度、特征图，开始计算可视化图
def cam_show_img(img, feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加

    # 计算每个通道的权重
    weights = np.mean(grads, axis=(1, 2))
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :].cpu().data.numpy()  # 特征图加权和

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)  # 归一化，避免除以零
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # 将原始图像像素值调整到 [0, 255] 范围
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # 叠加热力图和原始图像
    cam_img = 0.7 * heatmap.astype(np.float32) + 0.3 * img.astype(np.float32)
    cam_img = cam_img / cam_img.max() * 255
    cam_img = np.uint8(cam_img)

    # 保存结果
    cv2.imwrite("0_2.jpg", cam_img)

# 选择目标层，通常是最后一个卷积层
target_layer = model.features[6][-1].conv3

# 注册钩子
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


print(model)

# forward 
# 获取输入图像
sample_tensor = sample_data[1]  # shape: [3, H, W]

# 将输入图像添加批次维度并传入模型
output = model(sample_tensor.unsqueeze(0))

if isinstance(output, tuple):
    # 假设 logits 在 output[0] 中，如果不是，请根据实际情况调整
    output = output[2]

print("Adjusted output shape:", output.shape)

if output.dim() == 1:
    max_idx = output.argmax().item()
elif output.dim() == 2:
    max_idx = output.argmax(dim=1).item()
else:
    raise ValueError("Unexpected output dimensions")

# backward
model.zero_grad()
class_loss = output[0]
class_loss.backward()

# 提取梯度和特征图
grads_val = grad_block[0].cpu().data.numpy()
fmap = feature_block[0].squeeze(0)  # 移除批次维度

# 将 sample_tensor 转换为图像格式
img = sample_tensor.cpu().numpy().transpose(1, 2, 0)  # shape: [H, W, 3]
img = np.clip(img, 0, 1)  # 确保像素值在 [0, 1] 范围内

# 生成并保存 Grad-CAM 图像
cam_show_img(img, fmap, grads_val)
