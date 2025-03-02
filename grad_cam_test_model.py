import sys
from PIL import Image
import cv2
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
from utils_withMNIST import *

def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'MNIST', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[len('module.'):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
parser.add_argument('--load_model_file', type=str, default='Y:\FRDA\model\MNIST_resnet50\MCFL/fedavg.pth', help='the model to load as global model')
parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default='resnet50-MNIST', help='neural network used in training')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset used for training')
parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
parser.add_argument('--use_project_head', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
args = parser.parse_args()

train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               args.batch_size,)
backdoor_train_dl_global, backdoor_test_dl_global, backdoor_train_ds_global, backdoor_test_ds_global = get_dataloader(args.dataset, 
                                                                                                                   args.datadir, 
                                                                                                                   args.batch_size, 
                                                                                                                   args.batch_size, 
                                                                                                                   backdoor=True)
net_configs = args.net_config
args.n_parties = 1
nets, model_meta_data, layer_type = init_nets(net_configs, args.n_parties, args, device=args.device)

# Load the pre-trained model
if args.load_model_file:
    try:
        nets[0].load_state_dict(torch.load(args.load_model_file))
        global_model = copy.deepcopy(nets[0])
    except:
        global_model = nn.DataParallel(nets[0])
        global_model = remove_module_prefix(global_model)
        global_model.load_state_dict(torch.load(args.load_model_file))
    global_model.to(args.device)

# Ensure model is in evaluation mode
global_model.eval()

# 计算模型在测试集上的准确率
#test_acc, _ = compute_accuracy(global_model.cuda(), backdoor_test_dl_global, device=args.device)
#print(f'Model test accuracy on CIFAR-100: {test_acc}')


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

# Identify the target layer
# Adjust indices based on your model's structure
target_layer = global_model.features[6][-1].conv3

# Register hooks
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# 定义要处理的样本数量
num_samples = 100  # 您想处理的样本数量

# 创建保存 Grad-CAM 图像的文件夹
output_dir = 'Y:\FRDA\pic'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 从数据加载器中获取足够的样本
data_iter = iter(backdoor_test_dl_global)
batch_data, batch_labels = next(data_iter)

# 如果批次大小小于 num_samples，则调整 num_samples
if batch_data.size(0) < num_samples:
    num_samples = batch_data.size(0)

# 仅使用前 num_samples 个样本
batch_data = batch_data[:num_samples]
batch_labels = batch_labels[:num_samples]

batch_data = batch_data.to(args.device)
batch_labels = batch_labels.to(args.device)

# 遍历每个样本
for idx in range(num_samples):
    # 重置存储器
    gradients = []
    activations = []

    # 获取单个样本
    input_tensor = batch_data[idx].unsqueeze(0)  # [1, C, H, W]
    label = batch_labels[idx].unsqueeze(0)

    # 前向传播
    output = global_model(input_tensor)

    # 如果输出是元组，提取 logits
    if isinstance(output, tuple):
        logits = output[-1]
    else:
        logits = output

    # 计算预测类别
    if logits.dim() == 1:
        predicted = logits.argmax().item()
    elif logits.dim() == 2:
        _, predicted = torch.max(logits, 1)
        predicted = predicted.item()
    else:
        raise ValueError("Unexpected logits dimensions: {}".format(logits.shape))

    # 反向传播
    class_idx = batch_labels[0].item()
    global_model.zero_grad()
    if logits.dim() == 1:
        class_loss = logits[predicted]
    else:
        class_loss = logits[0, predicted]
    class_loss.backward()

    # 获取梯度和激活
    gradients = gradients[0]
    activations = activations[0]

    # 移动到 CPU 并分离
    gradients = gradients.detach().cpu()
    activations = activations.detach().cpu()

    # 实现 Grad-CAM++
    # 计算梯度的平方和立方
    grads_power_2 = gradients ** 2
    grads_power_3 = gradients ** 3

    # 计算 α
    sum_activations = torch.sum(activations, dim=[2, 3], keepdim=True)
    eps = 1e-7
    alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
    alpha = alpha.detach()



    # 计算权重
    weights = torch.sum(alpha * F.relu(gradients), dim=[2, 3])  # [1, C]

    # 计算加权后的特征图
    heatmap = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)  # [1, H, W]
    heatmap = F.relu(heatmap)

    # 归一化热力图
    heatmap = heatmap.squeeze().numpy()
    print(f"Heatmap shape after squeeze for sample {idx}:", heatmap.shape)

    # 检查是否有 NaN 或 Inf
    if np.isnan(heatmap).any() or np.isinf(heatmap).any():
        print(f"Heatmap contains NaN or Inf values for sample {idx}.")
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    # 归一化到 [0, 1]
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    if heatmap_max - heatmap_min != 0:
        heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = np.zeros_like(heatmap)

    # 缩放到 [0, 255] 并转换为 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)

    # 调整热力图大小，使用双线性插值
    heatmap_resized = cv2.resize(
        heatmap_uint8, 
        (input_tensor.size(3), input_tensor.size(2)), 
        interpolation=cv2.INTER_LINEAR
    )

    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # 获取输入图像
    input_image = input_tensor[0].cpu().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))  # [H, W, C]
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    # 转换输入图像为 uint8
    input_image_uint8 = np.uint8(255 * input_image)

    # 将灰度图像转换为 BGR 彩色图像
    input_image_uint8 = cv2.cvtColor(input_image_uint8, cv2.COLOR_GRAY2BGR)

    # 叠加热力图和原始图像
    superimposed_img = cv2.addWeighted(heatmap_colored, 0.6, input_image_uint8, 0.6, 0)


    # 保存图像，使用不同的文件名
    output_filename = os.path.join(output_dir, f'grad_cam_result_{idx}.jpg')
    cv2.imwrite(output_filename, superimposed_img)
    print(f'已保存 Grad-CAM 图像: {output_filename}')
