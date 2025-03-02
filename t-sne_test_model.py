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
from sklearn.manifold import TSNE
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

# 定义特征提取函数
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[2]  # 如果输出是tuple，获取第一个元素
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

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

title = 't-SNE Visualization of MNIST and Backdoor Samples'

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

if args.load_model_file:
    try:
        nets[0].load_state_dict(torch.load(args.load_model_file))
        global_model = copy.deepcopy(nets[0])
    except:
        global_model = nn.DataParallel(nets[0])
        global_model = remove_module_prefix(global_model)
        global_model.load_state_dict(torch.load(args.load_model_file))
                
# 提取test_dl_global和backdoor_test_dl_global的特征
test_features, test_labels = extract_features(global_model, test_dl_global, device=args.device)
backdoor_features, _ = extract_features(global_model, backdoor_test_dl_global, device=args.device)

# 将后门样本的标签设置为一个新的类别，例如11
backdoor_labels = np.full(backdoor_features.shape[0], 10)

# 合并特征和标签
all_features = np.concatenate([test_features, backdoor_features], axis=0)
all_labels = np.concatenate([test_labels, backdoor_labels], axis=0)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
all_features_2d = tsne.fit_transform(all_features)

# 自定义HEX颜色列表，每类一个颜色
colors = ['#535353', '#e04344', '#2470cf', '#45a46d', '#ab7ed0', 
          '#bf9a22', '#10c5c8', '#58539f', '#bbbbd6', '#d86967', '#eebabb']  # 最后一种颜色为后门类的颜色


# 绘制图形
plt.figure(figsize=(10, 8))
# 绘制前10类的测试集
for i in range(10):
    indices = np.where(all_labels == i)
    plt.scatter(all_features_2d[indices, 0], all_features_2d[indices, 1], label=f'Class {i}', color=colors[i], alpha=0.7)
    #plt.scatter(all_features_2d[indices, 0], all_features_2d[indices, 1], label=f'Class {i}', alpha=0.7)
# 绘制后门类
backdoor_indices = np.where(all_labels == 10)
plt.scatter(all_features_2d[backdoor_indices, 0], all_features_2d[backdoor_indices, 1], label='Backdoor Class', color='black', alpha=0.2)

plt.legend()
plt.title(title)
plt.show()