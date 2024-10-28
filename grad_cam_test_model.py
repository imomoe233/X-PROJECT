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
import datetime
import random
from tqdm import tqdm
import wandb
from model import *
from utils import *
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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

# 定义 Grad-CAM 生成函数
def generate_gradcam(model, input_tensor, target_layer, device):
    model.eval()
    input_tensor = input_tensor.to(device)  # 确保输入张量在正确的设备上

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == 'cuda'))

    # 确保 input_tensor 是 Tensor 类型
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0, :]

    # Normalize the Grad-CAM output
    grayscale_cam = np.maximum(grayscale_cam, 0)  # Apply ReLU
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
    max_val = np.max(grayscale_cam)
    if max_val > 0:
        grayscale_cam = grayscale_cam / max_val  # Normalize to 0-1
    else:
        print("Warning: Max value of grayscale_cam is zero. Skipping normalization.")

    return grayscale_cam




# Argument parsing and initialization
parser = argparse.ArgumentParser()
parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
parser.add_argument('--load_model_file', type=str, default='X:/Directory/code/MOON-backdoor/models/cifar10_resnet50/backdoor_pretrain(triggerOnly).pth', help='the model to load as global model')
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

    print(global_model)
    
# 主要逻辑部分
# 测试模型准确性
sample_data, _ = next(iter(backdoor_test_dl_global))
sample_data = sample_data.to(device=args.device)  # Move sample_data to the correct device

with torch.no_grad():
    outputs = global_model(sample_data)

# 生成 Grad-CAM
target_layer = global_model.features[3][2].conv3  # 选择最后一个 Bottleneck 的 conv3
gradcam_image = generate_gradcam(global_model, sample_data, target_layer, device=args.device)

# 预处理原始图像
original_image = sample_data[0].cpu().numpy()  # 提取一张原始图像
original_image = np.transpose(original_image, (1, 2, 0))  # 转换为 HWC 格式
original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))  # 归一化到 0-1

# 结合 Grad-CAM 图像
cam_image = show_cam_on_image(original_image, gradcam_image, use_rgb=True)
plt.imshow(cam_image)
plt.axis('off')  # 关闭坐标轴
plt.show()  # 显示图像