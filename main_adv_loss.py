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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from opacus import PrivacyEngine

from model import *
from utils import *
# from utils_withMNIST import *

# 设置 CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 用于记录 net_id=0 和 net_id=1 的参数和损失值
parameter_history_0 = []
loss_history_0 = []
parameter_history_1 = []
loss_history_1 = []


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default='cifar10_resnet50_MCFL_BadNets_fedavg', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_MCFL', help='train with backdoor_pretrain/backdoor_MCFL/backdoor_fedavg')
    parser.add_argument('--fedavg_method', type=str, default='fedavg', help='fedavg/weight_fedavg/multi_krum/trimmed_mean/median_fedavg/rfa/fedprox/DP/purning')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/cifar10_resnet50/", help='Model save directory path')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy noniid/iid')
    parser.add_argument('--min_data_ratio', type=float, default='1.0')
    parser.add_argument('--krum_k', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--alg', type=str, default='backdoor_MCFL',
                        help='communication strategy: fedavg/fedprox/moon/local_training')
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
    parser.add_argument('--load_first_net', type=int, default=0, help='whether load the first net as old net or not')

    
    parser.add_argument('--load_model_file', type=str, default='models/cifar10_resnet50/backdoor_pretrain(cleanOnly).pth', help='the model to load as global model')
    parser.add_argument('--load_backdoor_model_file', type=str, default='models/cifar10_resnet50/newnewbackdoorOnly_20.pth', help='the model to load as global model')
    
    #parser.add_argument('--load_model_file', type=str, default='models/cifar100_resnet50/cifar100_resnet50_MCFL_BadNets_fedavgcleanOnly_190.pth', help='the model to load as global model')
    #parser.add_argument('--load_backdoor_model_file', type=str, default='models/cifar100_resnet50/backdoor_pretrain(triggerOnly).pth', help='the model to load as global model')
    
    
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=0.1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=1, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--backdoor_sample_num', type=int, default=20)
    parser.add_argument('--fedprox', type=bool, default=False)



    
    
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--comm_round', type=int, default=80, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--local_max_epoch', type=int, default=500, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=0, help='how many rounds have executed for the loaded model')
    
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=10)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    
    args = parser.parse_args()
    
    
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="MCFL-backdoor",
            #name=args.partition + '_' + 'clients' + str(args.n_parties) + args.log_file_name,
            name=args.partition + '_' +  args.log_file_name,
            config={
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'dataset': args.dataset,
            'datadir': args.datadir,
            'model': args.model,
            'partition': args.partition,
            'alg': args.alg,
            'comm_round': args.comm_round,
            'backdoor': args.backdoor,
            'n_parties': args.n_parties,
            'logdir': args.logdir,
            'load_model_file': args.load_model_file,
            'load_backdoor_model_file': args.load_backdoor_model_file,
            'dropout_p': args.dropout_p,
            'mu': args.mu,
            'temperature': args.temperature,
            'modeldir': args.modeldir,
            'partition': args.partition,
            'beta': args.beta,
            'min_data_ratio': args.min_data_ratio,
            }
        )
        
    return args


def add_noise_to_gradients(model, noise_multiplier=1.0, max_grad_norm=0.01):
    """为模型的梯度添加高斯噪声以实现差分隐私。"""
    # 计算每个参数的梯度
    for param in model.parameters():
        if param.grad is not None:

            # 计算L2范数
            grad_norm = param.grad.data.norm(2)
            # 进行梯度裁剪
            if grad_norm > max_grad_norm:
                param.grad.data = param.grad.data / grad_norm * max_grad_norm

            if noise_multiplier > 0.0:
                noise = torch.normal(0, noise_multiplier * max_grad_norm, param.grad.size()).to(param.device)
                param.grad.data += noise
    return model


def replace_batchnorm_with_groupnorm(model, num_groups=8):
    """
    遍历模型，替换所有的 BatchNorm2d 为 GroupNorm。
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # 获取通道数
            num_channels = module.num_features
            # 创建 GroupNorm 层
            group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            # 替换模型中的层
            setattr(model, name, group_norm)
            print(f"替换 {name} 为 GroupNorm")
        else:
            # 递归替换子模块
            replace_batchnorm_with_groupnorm(module, num_groups)

    return model


def plot_3d_surface_and_contour(parameter_list, loss_list, colors, labels, title="Training Progress", net_id=0, round=0):
    fig = plt.figure(figsize=(14, 6))

    # 3D曲线图
    ax1 = fig.add_subplot(121, projection='3d')

    # 用于存储所有数据，以便绘制等高线图
    x_vals_all = []
    y_vals_all = []
    z_vals_all = []

    for parameters, losses, color, label in zip(parameter_list, loss_list, colors, labels):
        x_vals = np.array([param[0] for param in parameters])
        y_vals = np.array([param[1] for param in parameters])
        z_vals = np.array(losses)

        ax1.plot(x_vals, y_vals, z_vals, color=color, marker='o', label=label)

        x_vals_all.extend(x_vals)
        y_vals_all.extend(y_vals)
        z_vals_all.extend(z_vals)

    ax1.set_title(f'3D Surface Plot - {title}')
    ax1.set_xlabel('Parameter X')
    ax1.set_ylabel('Parameter Y')
    ax1.set_zlabel('Loss')
    ax1.legend()

    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.tricontourf(x_vals_all, y_vals_all, z_vals_all, levels=20, cmap='viridis')

    for parameters, color, label in zip(parameter_list, colors, labels):
        x_vals = np.array([param[0] for param in parameters])
        y_vals = np.array([param[1] for param in parameters])
        ax2.plot(x_vals, y_vals, color=color, marker='o', label=label)

    ax2.set_title(f'Contour Plot - {title}')
    ax2.set_xlabel('Parameter X')
    ax2.set_ylabel('Parameter Y')
    plt.colorbar(contour, ax=ax2)
    ax2.legend()

    # 创建保存图片的文件夹（如果不存在）
    save_dir = 'plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片
    filename = f'{save_dir}/net_{net_id}_round_{round}.png'
    plt.savefig(filename)
    plt.close(fig)  # 关闭图形，释放内存


def imshow(tensor):
    inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip([125.3, 123.0, 113.9], [63.0, 62.1, 66.7])],
    std=[1 / s for s in [63.0, 62.1, 66.7]]
)
    
    # 反归一化处理
    img = inv_normalize(tensor)
    # 将tensor转为numpy数组
    img = img.permute(1, 2, 0).numpy()
    # 裁剪到合法的像素值范围
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # 显示图片
    plt.imshow(img)
    plt.show()

def are_models_equal(model1, model2):
    # 获取两个模型的 state_dict
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()

    # 检查两个模型的层数是否一致
    if len(model1_dict) != len(model2_dict):
        return False

    # 逐层比较模型的权重和偏置
    for key in model1_dict:
        if key not in model2_dict:
            return False
        
        # 比较每个参数是否相等
        if not torch.equal(model1_dict[key], model2_dict[key]):
            print(f"Difference found in layer: {key}")
            return False

    return True

def random_update_params(global_w, net_para, update_ratio):
    """
    让 net_para 中的部分参数按照 update_ratio 比例进行更新，未更新的部分保持与 global_w 一致。
    并且将最大的 5% 和最小的 5% 参数值调整到与第 11% 的参数值相同，以避免被修剪。
    
    :param global_w: 全局模型的参数 (state_dict)
    :param net_para: 当前客户端的参数 (state_dict)
    :param update_ratio: 本轮希望更新的参数比例 (0~1)
    :return: 更新后的 net_para 参数 (state_dict)，部分参数更新，其他与 global_w 一致。
    """
    for key in net_para:
        # 获取全局参数和当前客户端参数
        global_param = global_w[key].float()
        net_param = net_para[key].float()

        # 计算参数的数量
        num_params = net_param.numel()
        num_update_params = int(update_ratio * num_params)

        # 将参数展平成一维张量
        flat_net_param = net_param.view(-1)
        flat_global_param = global_param.view(-1)

        # 随机更新部分参数，其余保持与全局参数一致
        if num_update_params > 0:
            indices = torch.randperm(num_params)[:num_update_params]
            with torch.no_grad():
                flat_net_param.copy_(flat_global_param)  # 将所有参数设置为全局参数
                # 对选定的参数添加微小扰动
                # flat_net_param[indices] += 0.01 * torch.randn_like(flat_net_param[indices])

        continue

        # 调整最大的5%和最小的5%参数值
        with torch.no_grad():
            # 对参数进行排序，获取排序后的索引
            sorted_indices = torch.argsort(flat_net_param)
            num_adjust_params = max(1, int(0.05 * num_params))  # 确保至少有一个参数被调整

            if num_adjust_params * 2 < num_params:
                # 计算第11%位置的索引
                lower_threshold_idx = int(0.10 * num_params)
                upper_threshold_idx = int(0.90 * num_params)

                # 获取第11%位置的参数值
                lower_threshold_value = flat_net_param[sorted_indices[lower_threshold_idx]].clone()
                upper_threshold_value = flat_net_param[sorted_indices[upper_threshold_idx]].clone()

                # 调整最小的5%参数值
                smallest_indices = sorted_indices[:num_adjust_params]
                flat_net_param[smallest_indices] = lower_threshold_value

                # 调整最大的5%参数值
                largest_indices = sorted_indices[-num_adjust_params:]
                flat_net_param[largest_indices] = upper_threshold_value
            else:
                # 如果参数数量过少，直接跳过调整
                pass

        # 将调整后的参数重新赋值回 net_para
        net_para[key] = flat_net_param.view_as(net_param)

    return net_para

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[len('module.'):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def apply_differential_privacy(param, epsilon=780.0, delta=0.5):
    """
    为给定的参数添加高斯噪声以实现差分隐私。
    
    参数:
    param (torch.Tensor): 要添加噪声的参数张量。
    epsilon (float): 差分隐私参数，控制隐私预算。默认值为1.0。(调大以减小噪声)
    delta (float): 差分隐私中的 delta 参数，通常非常小。默认值为1e-5。(调大以减小噪声)
    
    返回:
    torch.Tensor: 添加了差分隐私噪声的参数张量。
    """
    sensitivity = 0.1  # 假设L2敏感度为1 (调小以减小噪声)
    # 计算标准差sigma
    sigma = sensitivity * torch.sqrt(2 * torch.log(torch.tensor(1.25) / delta)) / epsilon
    # 生成与param形状相同的高斯噪声并添加到param上
    noise = torch.normal(0, sigma, size=param.shape).cuda()  # 生成高斯噪声

    return param + noise
    

    
def prune_model_updates_with_mask(net_para, threshold=3.0):
    """生成超过阈值的参数掩码，而不直接修改参数。"""
    mask_dict = {}
    total_zeros = 0
    total_elements = 0
    for key, value in net_para.items():
        # 创建掩码，绝对值大于阈值的位置为 0，其余为 1
        mask = (torch.abs(value) <= threshold).float()
        num_elements = value.numel()
        num_zeros = num_elements - mask.sum().item()
        total_zeros += num_zeros
        total_elements += num_elements
        mask_dict[key] = mask
    #print(f"总共将 {total_zeros} 个参数掩盖为 0，总共有 {total_elements} 个参数")
    return mask_dict

def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
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


def train_net(net_id, net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl, epochs, lr, args_optimizer, args, round, device="cpu", backdoor=False):
    net = nn.DataParallel(net)
    net.cuda()
    net.train()
    logger.info('Training network %s' % str(net_id))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        train_dataloader = tqdm(train_dataloader)
        
        if backdoor and net_id == 0 and args.backdoor == 'backdoor_pretrain':
            train_dataloader.set_description(f"Training traindata clean Mix backdoor | round:{round} client:{net_id}")
            for batch_idx, ((clean_x, clean_target), (backdoor_x, backdoor_target)) in enumerate(zip(train_dataloader, backdoor_train_dl)):
                optimizer.zero_grad()
                
                # Combine clean and backdoor samples
                combined_x = torch.cat((clean_x, backdoor_x), dim=0)
                combined_target = torch.cat((clean_target, backdoor_target), dim=0)
                
                combined_x, combined_target = combined_x.cuda(), combined_target.cuda()
                combined_x.requires_grad = False
                combined_target.requires_grad = False
                combined_target = combined_target.long()
                
                # Forward pass
                _, _, out = net(combined_x)
                
                # Compute loss
                loss = criterion(out, combined_target)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss_collector.append(loss.item())
        else:
            if args.backdoor == 'backdoor_fedavg':
                train_dataloader.set_description(f"Training clean traindata | round:{round} client:{net_id}")
            else:
                train_dataloader.set_description(f"Training backdoor traindata | round:{round} client:{net_id}")
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _,_,out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch >= 10 and epoch % 10 == 0:
            net.eval()
            if args.backdoor == 'backdoor_pretrain':
                torch.save(net.module.state_dict(), args.modeldir + args.log_file_name + f'backdoorOnly_{epoch}.pth')

    if args.backdoor =='pretrain':
        train_dataloader.set_description("Testing final traindata")
        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_dataloader.set_description("Testing final testdata")
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        
        logger.info('>> Final Training accuracy: %f' % train_acc)
        logger.info('>> Final Test accuracy: %f' % test_acc)
        wandb.log({'epoch': epoch, 'Final Training accuracy': train_acc, 'Final Test accuracy': test_acc})
        
        if backdoor:
            backdoor_train_dl.set_description("Testing final backdoor traindata")
            backdoor_train_acc, _ = compute_accuracy(net, backdoor_train_dl, device=device)
            backdoor_test_dl.set_description("Testing final backdoor testdata")
            backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(net, backdoor_test_dl, get_confusion_matrix=True, device=device)
            
            logger.info('>> Final Backdoor Training accuracy: %f' % backdoor_train_acc)
            logger.info('>> Final Backdoor Test accuracy: %f' % backdoor_test_acc)
            wandb.log({'epoch': epoch, 'Final Backdoor Training accuracy': backdoor_train_acc, 'Final Backdoor Test accuracy': backdoor_test_acc})   
        logger.info(' ** Training complete **')
        return train_acc, test_acc

    net.eval()
    net.to('cpu')
    
    
def train_net_fedcon_backdoor(net_id, net, global_net, previous_nets, backdoor_net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl, epochs, lr, args_optimizer, mu, temperature, args, round, device="cpu"):

    net = nn.DataParallel(net)
    net.cuda()
    net.train()
    
    backdoor_net.cuda()
    global_net.cuda()
    backdoor_net.eval()
    global_net.eval()
    
    

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    triple_loss = torch.nn.TripletMarginWithDistanceLoss(margin=1.0).cuda()
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_bce = nn.BCEWithLogitsLoss().cuda()
    
    for previous_net in previous_nets:
        previous_net.cuda()
        
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

    if args.fedavg_method == 'purning':
        mask_dict = prune_model_updates_with_mask(net.state_dict(), threshold=3.0)

    
    for epoch in range(epochs):

        epoch_loss_collector = []
        
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        
        epoch_loss_cls_collector = []
        epoch_align_loss_clean_collector = []
        epoch_align_loss_backdoor_collector = []
        epoch_adv_loss_collector = []

        
        net.train()
        
        correct_clean = 0  # 统计干净样本的正确预测数量
        total_clean = 0  # 干净样本的总数

        correct_backdoor = 0  # 统计后门样本的正确预测数量
        total_backdoor = 0  # 后门样本的总数
            
        if net_id == 0:

            
            train_dataloader = tqdm(train_dataloader)
            train_dataloader.set_description(f"Training traindata cleandata | round:{round} client:{net_id}")

            for batch_idx, ((clean_x, clean_target), (backdoor_x, backdoor_target)) in enumerate(zip(train_dataloader, backdoor_train_dl)):
                optimizer.zero_grad()
                
                # 将良性样本和后门样本组合
                combined_x = torch.cat((clean_x, backdoor_x), dim=0)
                combined_target = torch.cat((clean_target, backdoor_target), dim=0)
                
                combined_x, combined_target = combined_x.cuda(), combined_target.cuda()
                combined_target = combined_target.long()
                
                # 前向传播
                _, pro1, out = net(combined_x)
                _, pro_global, _ = global_net(combined_x)
                _, pro_backdoor, _ = backdoor_net(combined_x)
                
                # 归一化特征表示
                pro1 = torch.nn.functional.normalize(pro1, dim=-1)
                pro_global = torch.nn.functional.normalize(pro_global, dim=-1)
                pro_backdoor = torch.nn.functional.normalize(pro_backdoor, dim=-1)
                
                # 计算余弦相似度
                cos = torch.nn.CosineSimilarity(dim=-1)
                sim_global = cos(pro1, pro_global)
                sim_backdoor = cos(pro1, pro_backdoor)
                
                # 根据样本类型（良性或后门）构建对齐和对抗性损失
                mask_backdoor = (combined_target == backdoor_target[0])
                mask_clean = ~mask_backdoor
                
                # 对齐损失（良性样本对齐 global_net）
                if mask_clean.sum() > 0:
                    align_loss_clean = mu * criterion_bce(sim_global[mask_clean] / temperature, torch.ones(mask_clean.sum()).cuda())
                else:
                    align_loss_clean = 0.0
                
                # 对齐损失（后门样本对齐 backdoor_net）
                if mask_backdoor.sum() > 0:
                    align_loss_backdoor = mu * criterion_bce(sim_backdoor[mask_backdoor] / temperature, torch.ones(mask_backdoor.sum()).cuda())
                else:
                    align_loss_backdoor = 0.0
                
                # 对抗性损失（良性样本远离 backdoor_net）
                if mask_clean.sum() > 0:
                    adv_loss = mu * criterion_bce(sim_backdoor[mask_clean] / temperature, torch.zeros(mask_clean.sum()).cuda())
                else:
                    adv_loss = 0.0
                
                # 分类损失
                loss_cls = criterion_cls(out, combined_target)
                
                 # FedProx 正则化项
                if args.fedprox:
                    fedprox_reg = 0.0
                    for param, global_param in zip(net.parameters(), global_net.parameters()):
                        fedprox_reg += (mu / 2) * torch.norm(param - global_param) ** 2
                else:
                    fedprox_reg = 0.0
                
                # 总损失
                loss = align_loss_clean + align_loss_backdoor + fedprox_reg + adv_loss + loss_cls
                #loss = loss_cls
                
                # 反向传播和优化
                loss.backward()
                if args.fedavg_method == 'purning':
                    # 在优化器更新前，应用掩码到梯度
                    for name, param in net.named_parameters():
                        if name in mask_dict and param.grad is not None:
                            param.grad.data.mul_(mask_dict[name])
                            
                if args.fedavg_method == 'DP':         
                    # 4. 添加差分隐私噪声
                    net = add_noise_to_gradients(net)            
                
                optimizer.step()
                if args.fedavg_method == 'purning':
                    with torch.no_grad():
                        for name, param in net.named_parameters():
                            if name in mask_dict:
                                param.data.mul_(mask_dict[name])
                
                # 统计准确率
                _, predicted = torch.max(out, 1)
                correct_clean += (predicted[mask_clean] == combined_target[mask_clean]).sum().item()
                total_clean += mask_clean.sum().item()
                
                correct_backdoor += (predicted[mask_backdoor] == combined_target[mask_backdoor]).sum().item()
                total_backdoor += mask_backdoor.sum().item()
                
                epoch_loss_collector.append(loss.item())
                epoch_loss_cls_collector.append(loss_cls.item())
                epoch_align_loss_clean_collector.append(align_loss_clean.item())
                epoch_align_loss_backdoor_collector.append(align_loss_backdoor.item())
                epoch_adv_loss_collector.append(adv_loss.item())

                if net_id == 0:
                    # 在训练函数的合适位置，例如在每个 round 结束后
                    with torch.no_grad():
                        current_parameters = []
                        for param in net.parameters():
                            # 为简化，可将参数展平成一维并选择前两个参数
                            flat_param = param.view(-1)
                            if flat_param.numel() >= 2:
                                current_parameters.append([flat_param[0].item(), flat_param[1].item()])
                        if current_parameters:
                            parameter_history_0.append(np.mean(current_parameters, axis=0))
                            loss_history_0.append(loss.item())
                        '''
                        # 每隔一定的轮次绘制图形，例如每 5 个轮次
                        if round % 1 == 0 and round >= 2:
                            plot_3d_surface_and_contour(
                                parameter_history_0,
                                loss_history_0,
                                title=f"Adversarial Client (net_id=0) - Round {round}",
                                net_id = net_id,
                                round = round
                            )
                        '''

            train_clean_acc = correct_clean/total_clean
            train_backdoor_acc = correct_backdoor/total_backdoor    
            

            
        elif net_id != 0:
            train_dataloader = tqdm(train_dataloader)
            train_dataloader.set_description(f"Training traindata clean | round:{round} client:{net_id}")
            for batch_idx, (x, target) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                x, target = x.cuda(), target.cuda()
                target = target.long()
                x.requires_grad = False
                target.requires_grad = False
                
                _, pro1, out = net(x)
                _, pro2, _ = global_net(x)
                
                pro1 = torch.nn.functional.normalize(pro1, dim=-1)
                pro2 = torch.nn.functional.normalize(pro2, dim=-1)
                
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.cuda()
                    _, pro3, _ = previous_net(x)
                    pro3 = torch.nn.functional.normalize(pro3, dim=-1)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    previous_net.to('cpu')
                    break

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = mu * criterion(logits, labels)
                loss1 = criterion(out, target)
                loss = loss1 + loss2
                
                # 计算样本的准确率
                _, predicted = torch.max(out, 1)  # 取出每行的最大值作为预测类别
                correct_clean += (predicted == target).sum().item()  # 统计正确预测的数量
                total_clean += target.size(0)  # 统计样本总数
                
                
                loss.backward()
                if args.fedavg_method == 'purning':
                    # 在优化器更新前，应用掩码到梯度
                    for name, param in net.named_parameters():
                        if name in mask_dict and param.grad is not None:
                            param.grad.data.mul_(mask_dict[name])
                            
                if args.fedavg_method == 'DP':         
                    # 4. 添加差分隐私噪声
                    net = add_noise_to_gradients(net)
                    
                optimizer.step()
                if args.fedavg_method == 'purning':
                    with torch.no_grad():
                        for name, param in net.named_parameters():
                            if name in mask_dict:
                                param.data.mul_(mask_dict[name])


                
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())
                if args.wandb:
                    wandb.log({
                        'Round': round,
                        'benign_loss_total': loss,
                        'benign_loss_1': loss1,
                        'benign_loss_2': loss2,
                    })

                if net_id == 1:
                    # 在训练函数的合适位置，例如在每个 round 结束后
                    with torch.no_grad():
                        current_parameters = []
                        for param in net.parameters():
                            flat_param = param.view(-1)
                            if flat_param.numel() >= 2:
                                current_parameters.append([flat_param[0].item(), flat_param[1].item()])
                        if current_parameters:
                            parameter_history_1.append(np.mean(current_parameters, axis=0))
                            loss_history_1.append(loss.item())
        
                        if round % 1 == 0:
                            parameter_list = [parameter_history_0, parameter_history_1]
                            loss_list = [loss_history_0, loss_history_1]
                            colors = ['red', 'green']
                            labels = ['attack', 'benign']
                            
                            plot_3d_surface_and_contour(
                                parameter_list=parameter_list,
                                loss_list=loss_list,
                                colors=colors,
                                labels=labels,
                                title=f"Training Progress - Round {round}",
                                net_id=net_id,
                                round=round
                            )

                        
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            train_clean_acc = correct_clean/total_clean

        if net_id == 0:

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss_cls_collector = sum(epoch_loss_cls_collector) / len(epoch_loss_cls_collector)
            epoch_align_loss_clean_collector = sum(epoch_align_loss_clean_collector) / len(epoch_align_loss_clean_collector)
            epoch_align_loss_backdoor_collector = sum(epoch_align_loss_backdoor_collector) / len(epoch_align_loss_backdoor_collector)
            epoch_adv_loss_collector = sum(epoch_adv_loss_collector) / len(epoch_adv_loss_collector)
            logger.info('Round: %d Client: %d Epoch: %d Loss: %f loss_cls: %f align_loss_clean: %f align_loss_backdoor: %f adv_loss: %f' % (round, net_id, epoch, epoch_loss, epoch_loss_cls_collector, epoch_align_loss_clean_collector, epoch_align_loss_backdoor_collector, epoch_adv_loss_collector))

            if args.wandb:
                wandb.log({
                    'Round': round,
                    'attack_loss_total': epoch_loss,
                    'loss_cls': epoch_loss_cls_collector,
                    'align_loss_clean': epoch_align_loss_clean_collector,
                    'align_loss_backdoor': epoch_align_loss_backdoor_collector,
                    'adv_loss': epoch_adv_loss_collector,
                    'train_clean_acc': train_clean_acc,
                    'train_backdoor_acc': train_backdoor_acc,
                })
    
        elif net_id != 0:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2))
            if args.wandb:
                wandb.log({
                        'Round': round,
                        'attack_loss_total': epoch_loss,
                        'attack_loss_1': epoch_loss1,
                        'attack_loss_2': epoch_loss2,
                    }) 
             
        net.eval()



def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None, server_c = None, clients_c = None, round=None, device="cuda:0", backdoor_model=None):
    avg_acc = 0.0
    avg_backdoor_testacc = 0.0
    acc_list = []
    backdoor_acc_list=[]
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        # 获取当前net的数据索引
        dataidxs = net_dataidx_map[net_id]

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size, dataidxs)
        backdoor_train_dl, backdoor_test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size, dataidxs, backdoor=True)
        
        n_epoch = args.epochs
        
        if args.backdoor == 'backdoor_MCFL':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            train_net_fedcon_backdoor(net_id, net, global_model, prev_models, backdoor_model, train_dl_local, test_dl, backdoor_train_dl, backdoor_test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round, device=device)
            continue
        
        # 第一个客户机则传后门的数据进去
        elif args.backdoor == 'backdoor_fedavg' and net_id == 0:
            train_net(net_id, net, backdoor_train_dl, backdoor_test_dl, None, None, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=True)
            continue
        # 其他客户机则传正常数据进去
        elif args.backdoor == 'backdoor_fedavg' and net_id != 0:
            train_net(net_id, net, train_dl_local, test_dl_local, None, None, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=False)
            continue

    return nets


def backdoor_pretrain(args):
    # Initialize model
    net_configs = args.net_config
    args.n_parties = 1
    nets, model_meta_data, layer_type = init_nets(net_configs, args.n_parties, args, device=args.device)

    # Get DataLoader
    train_dl, test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
    backdoor_train_dl, backdoor_test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, backdoor=True)
    
    # Train the model locally
    #train_net(0, nets[0], train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, args.epochs, args.lr, args.optimizer, args, device=args.device, backdoor=True)
    train_net(0, nets[0], backdoor_train_dl, backdoor_test_dl, None, None, args.epochs, args.lr, args.optimizer, args, device=args.device, backdoor=False)

    torch.save(nets[0].state_dict(), args.modeldir + args.log_file_name + '_backdoorOnly_last.pth')


def backdoor_MCFL(args):
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #logger.info(device)

    seed = args.init_seed
    #logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    #logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               args.batch_size,)
    backdoor_train_dl_global, backdoor_test_dl_global, backdoor_train_ds_global, backdoor_test_ds_global = get_dataloader(args.dataset, 
                                                                                                                   args.datadir, 
                                                                                                                   args.batch_size, 
                                                                                                                   args.batch_size, 
                                                                                                                   backdoor=True)
    
    
    
    #train_dl_global = tqdm(train_dl_global)
    #test_dl = tqdm(test_dl)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    #logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')
    net = nets[0]
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file:
        try:
            global_model.load_state_dict(torch.load(args.load_model_file))
            n_comm_rounds -= args.load_model_round
        except:
            global_model = nn.DataParallel(global_models[0])
            global_model.load_state_dict(torch.load(args.load_model_file))
            n_comm_rounds -= args.load_model_round
        #global_model.eval()
    # load backdoor model
    try:
        backdoor_model = copy.deepcopy(nets[0])
        if args.load_backdoor_model_file:
            backdoor_model.load_state_dict(torch.load(args.load_backdoor_model_file))
            for param in backdoor_model.parameters():
                param.requires_grad = False
    except:
        backdoor_model = nn.DataParallel(copy.deepcopy(nets[0]))
        if args.load_backdoor_model_file:
            backdoor_model.load_state_dict(torch.load(args.load_backdoor_model_file))
            for param in backdoor_model.parameters():
                param.requires_grad = False

    if args.backdoor == 'backdoor_MCFL':
        '''
        test_acc, _ = compute_accuracy(global_model.cuda(), test_dl_global, device=device)
        backdoor_test_acc, _ = compute_accuracy(global_model.cuda(), backdoor_test_dl_global, device=device)
        
        print(f'pretrain global_model test acc: {test_acc}')
        print(f'pretrain global_model backdoor acc: {backdoor_test_acc}')
        
        test_acc, _ = compute_accuracy(backdoor_model.cuda(), test_dl_global, device=device)
        backdoor_test_acc, _ = compute_accuracy(backdoor_model.cuda(), backdoor_test_dl_global, device=device)
        
        print(f'pretrain backdoor_model test acc: {test_acc}')
        print(f'pretrain backdoor_model backdoor acc: {backdoor_test_acc}')
        '''
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    net.load_state_dict(torch.load(args.load_model_file))
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
        
        if args.fedavg_method == 'fedprox':
            args.fedprox == True
        else:
            args.fedprox == False
            
        current_round = 0
        
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            nets_this_round = {k: copy.deepcopy(nets[k]) for k in party_list_this_round}
            for net in nets_this_round.values():
                try:
                    net.load_state_dict(global_w)
                except: 
                    net = nn.DataParallel(net)
                    net.load_state_dict(global_w)   
                for param in net.parameters():
                    param.requires_grad = True
                    
            

            
            print('=============== Round: ' + str(round) + ' ===============')
            
            # 用于记录 net_id=0 和 net_id=1 的参数和损失值
            parameter_history_0 = []
            loss_history_0 = []
            parameter_history_1 = []
            loss_history_1 = []
            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device, backdoor_model=backdoor_model)

            # 当前轮次
            current_round = current_round + round  # 假设你有一个记录当前轮次的变量
            update_ratio = (current_round + 1) / 100  # 动态计算更新比例
            update_ratio = 1
            if args.fedavg_method == 'weight_fedavg':
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    if net_id == 0:
                        #net_para = random_update_params(global_w, net_para, update_ratio) 
                        for key in net_para:
                            global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                    else:
                        for key in net_para:
                            global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            elif args.fedavg_method == 'DP':
                num_clients = len(party_list_this_round)
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            # 初始化 global_w 为第一个客户端的权重
                            global_w[key] = net_para[key].clone() / num_clients
                    else:
                        for key in net_para:
                            # 对每个客户端的权重求平均
                            global_w[key] += net_para[key] / num_clients             
            elif args.fedavg_method == 'purning':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

                for net_id, net in enumerate(nets_this_round.values()):
                    # 获取客户端的模型参数
                    net_para = net.state_dict()
                    
                    # 对客户端的参数进行剪枝
                    # pruned_net_para = prune_model_updates(net_para)  # 使用阈值 1.0，可根据需求调整
                    
                    # 加权平均聚合
                    if net_id == 0:
                        for key in net_para:
                            # 初始化 global_w 为第一个客户端的权重
                            global_w[key] = net_para[key].clone() / num_clients
                    else:
                        for key in net_para:
                            # 对每个客户端的权重求平均
                            global_w[key] += net_para[key] / num_clients             
            elif args.fedavg_method == 'fedavg' or args.fedavg_method == 'fedprox':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            # 初始化 global_w 为第一个客户端的权重
                            global_w[key] = net_para[key].clone() / num_clients
                    else:
                        for key in net_para:
                            # 对每个客户端的权重求平均
                            global_w[key] += net_para[key] / num_clients    
            elif args.fedavg_method == 'fedprox':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    for key in net_para:
                        # 对每个客户端的权重求平均
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]    
            elif args.fedavg_method == 'trimmed_mean': 
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                
                # 对于每个模型参数，在客户端更新中剔除最高和最低的值，然后计算剩余值的平均数。
                trim_ratio = 0.05  # 可以设置为 10% 的修剪比例，丢弃最高和最低的 10% 值
                
                # 存储所有客户端的参数
                all_net_params = {key: [] for key in global_w.keys()}
                
                # 收集每个客户端的参数
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
    
                    for key in net_para:
                        net_para[key] = net_para[key] * fed_avg_freqs[net_id]
                        all_net_params[key].append(net_para[key].cpu().float())  # 将参数值存入列表中
                
                # 对每个参数执行 Trimmed Mean 聚合
                for key in global_w:
                    stacked_params = torch.stack(all_net_params[key])  # 转换为 tensor 列表
                    
                    # 对每个参数进行排序，沿第一个维度排序（即不同客户端的参数值）
                    sorted_params, _ = torch.sort(stacked_params, dim=0)
                    
                    # 修剪掉最高和最低的值
                    trim_num = max(1, int(trim_ratio * len(nets_this_round)))  # 计算需要剔除的数量
                    if trim_num * 2 >= len(nets_this_round):
                        raise ValueError("客户端数量过少，无法进行有效的修剪")
                    
                    # 丢弃前 trim_num 和后 trim_num
                    trimmed_params = sorted_params[trim_num: -trim_num]
                    # trimmed_params = sorted_params[:]
                    # 计算剩余参数的平均值并将其转回合适的设备
                    global_w[key] = torch.mean(trimmed_params.to(global_w[key].device), dim=0)
            elif args.fedavg_method == 'median_fedavg':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                
                # 初始化用于存储所有客户端参数的字典
                global_w = {}

                # 首先获取每个客户端的参数
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    
                    for key in net_para:
                        if key not in global_w:
                            global_w[key] = []
                        global_w[key].append((net_para[key].clone()) * fed_avg_freqs[net_id])

                # 计算每个参数的中值
                for key in global_w:
                    # 将所有客户端的参数堆叠成一个张量
                    stacked_params = torch.stack(global_w[key])
                    
                    # 计算每个参数的中值，并替换全局参数
                    global_w[key] = torch.median(stacked_params, dim=0)[0]
            elif args.fedavg_method == 'krum':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                # 初始化存储每个客户端参数的字典
                client_params = []
                
                # 收集每个客户端的模型参数
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    for key in net_para:
                        # 对每个客户端的权重求平均
                        net_para[key] = net_para[key] * fed_avg_freqs[net_id]  
                    client_params.append(net_para)

                # 计算每个客户端的参数之间的距离
                num_clients = len(client_params)
                distances = torch.zeros((num_clients, num_clients))
                
                for i in range(num_clients):
                    for j in range(i + 1, num_clients):
                        dist = 0
                        # 计算每个参数的欧氏距离
                        for key in client_params[i]:
                            dist += torch.norm(client_params[i][key].float() - client_params[j][key].float()).item()
                        distances[i, j] = dist
                        distances[j, i] = dist

                # 计算每个客户端的 Krum 得分
                scores = []
                for i in range(num_clients):
                    sorted_distances, _ = torch.sort(distances[i])
                    # Krum score 为最近 num_clients - 2 个客户端的距离和
                    score = sorted_distances[:num_clients - 2].sum()
                    scores.append(score)
                
                # 选择 Krum 得分最小的客户端作为全局参数
                krum_client_idx = torch.argmin(torch.tensor(scores)).item()
                global_w = client_params[krum_client_idx]
                
                # 将选择的客户端参数作为全局参数加载到全局模型中
                global_model.load_state_dict(global_w)
            elif args.fedavg_method == 'multi_krum':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                
                # K 是我们要选择的最接近的客户端数量
                K = args.krum_k  # 例如，K = 3
                
                # 初始化所有客户端的参数存储
                global_w = {}
                client_weights = []
    
                # 首先收集所有客户端的参数
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    client_weights.append(net_para)
                    if net_id == 0:
                        # 初始化全局参数存储结构
                        for key in net_para:
                            global_w[key] = torch.zeros_like(net_para[key], dtype=torch.float32)  # 初始化为浮点类型
                        
    
                # 计算所有客户端之间的距离矩阵
                num_clients = len(client_weights)
                distance_matrix = torch.zeros((num_clients, num_clients))
    
                for i in range(num_clients):
                    for j in range(i + 1, num_clients):
                        dist = 0
                        for key in client_weights[i]:
                            dist += torch.norm(client_weights[i][key].float() - client_weights[j][key].float()).item()
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist
    
                # 计算每个客户端的得分（选择与其距离最近的 (num_clients - K - 1) 个客户端的总距离）
                scores = []
                for i in range(num_clients):
                    sorted_distances, _ = torch.sort(distance_matrix[i])
                    score = sorted_distances[:num_clients - K - 1].sum()
                    scores.append(score)
    
                # 找出得分最低的K个客户端
                selected_clients = torch.topk(torch.tensor(scores), K, largest=False).indices
    
                # 聚合所选择的客户端的参数
                for client_idx in selected_clients:
                    client_state = client_weights[client_idx]
                    for key in global_w:
                        global_w[key] += client_state[key].float()  # 将每个客户端参数转换为浮点类型
    
                # 对选择的客户端数量取平均
                for key in global_w:
                    global_w[key] /= float(K)  # 平均时使用浮点类型
            elif args.fedavg_method == 'rfa':
                num_clients = len(party_list_this_round)
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
                # 设置RFA的迭代次数
                num_iterations = 5  # 可以根据需要调整迭代次数

                # 初始化全局模型参数字典
                global_w = global_model.state_dict()
                #net_para = random_update_params(global_w, net_para, update_ratio) 


                # RFA 聚合迭代
                for _ in range(num_iterations):
                    # 初始化用于存储每个客户端与全局模型差值的列表
                    deltas = {key: torch.zeros_like(global_w[key]) for key in global_w}

                    # 计算每个客户端的模型与当前全局模型的差值
                    for net_id, net in enumerate(nets_this_round.values()):
                        net_para = net.state_dict()
                        for key in net_para:
                            net_para[key] = net_para[key] * fed_avg_freqs[net_id]  
                            deltas[key] = deltas[key].float()  # Convert deltas[key] to a FloatTensor if it's not already
                            net_para[key] = net_para[key].float()  # Convert net_para[key] to a FloatTensor
                            global_w[key] = global_w[key].float()  # Convert global_w[key] to a FloatTensor
                            deltas[key] += (net_para[key] - global_w[key]) / len(nets_this_round)

                    # 更新全局模型参数
                    for key in global_w:
                        global_w[key] += deltas[key]

            global_w = remove_module_prefix(global_w)
            global_model.load_state_dict(global_w)

            global_model.cuda()

            test_dl_global = tqdm(test_dl_global)
            test_dl_global.set_description("Testing final testdata")
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            backdoor_test_dl_global = tqdm(backdoor_test_dl_global)
            backdoor_test_dl_global.set_description("Testing final backdoor testdata")
            backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(global_model, backdoor_test_dl_global, get_confusion_matrix=True, device=device)
            
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Backdoor Test accuracy: %f' % backdoor_test_acc)
            logger.info('>> Global Model sum accuracy: %f' % (test_acc + backdoor_test_acc))
            logger.info(' ** Training Round complete **')

            if args.wandb:
                wandb.log({
                            "Round": round,
                            "Benign Acc": test_acc,
                            "Attack Success Rate": backdoor_test_acc,
                            "Sum Acc": (test_acc + backdoor_test_acc),
                            })

            
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets
            
            
            mkdirs(args.modeldir+'')
            mkdirs(args.modeldir+'MCFL/'+args.fedavg_method+'/')
            if round % args.save_model == 0:
                global_model.eval()
                torch.save(global_model.state_dict(), args.modeldir+'MCFL/'+args.fedavg_method+'/'+f'global_model_round_{round}.pth')
                
        global_model.eval()
        torch.save(global_model.state_dict(), args.modeldir+'MCFL/'+args.fedavg_method+'/'+f'global_model_last.pth')


def backdoor_fedavg(args):
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    #logger.info(device)

    seed = args.init_seed
    #logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    #logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = custom_partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta, min_data_ratio=args.min_data_ratio)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)
    backdoor_train_dl_global, backdoor_test_dl_global, backdoor_train_ds_global, backdoor_test_ds_global = get_dataloader(args.dataset, 
                                                                                                                   args.datadir, 
                                                                                                                   args.batch_size, 
                                                                                                                   32, 
                                                                                                                   backdoor=True)
    
    
    #train_dl_global = tqdm(train_dl_global)
    #test_dl = tqdm(test_dl)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    #logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')
    net = nets[0]
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    
    if args.load_model_file:
        try:
            global_model.load_state_dict(torch.load(args.load_model_file))
        except:
            global_model = nn.DataParallel(global_models[0])
            global_model.load_state_dict(torch.load(args.load_model_file))
    

    for round in range(n_comm_rounds):
        logger.info("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False
            
        global_w = global_model.state_dict()

        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
                try:
                    net.load_state_dict(global_w)
                except: 
                    net = nn.DataParallel(net)
                    net.load_state_dict(global_w)
                for param in net.parameters():
                    param.requires_grad = True

        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, round=round, device=device)

        if args.fedavg_method == 'weight_fedavg':
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
        elif args.fedavg_method == 'weight_fedavg_DP':
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()

                # 对每个客户端参数应用差分隐私
                for key in net_para:
                    net_para[key] = apply_differential_privacy(net_para[key], epsilon=1.0)  # epsilon 可调

                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
        elif args.fedavg_method == 'weight_fedavg_purning':
            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                # 获取客户端的模型参数
                net_para = net.state_dict()
                
                # 对客户端的参数进行剪枝
                pruned_net_para = prune_model_updates(net_para, threshold=1.0)  # 使用阈值 1.0，可根据需求调整
                
                # 加权平均聚合
                if net_id == 0:
                    for key in pruned_net_para:
                        global_w[key] = pruned_net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in pruned_net_para:
                        global_w[key] += pruned_net_para[key] * fed_avg_freqs[net_id]          
        elif args.fedavg_method == 'fedavg':
            num_clients = len(party_list_this_round)
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        # 初始化 global_w 为第一个客户端的权重
                        global_w[key] = net_para[key].clone() / num_clients
                else:
                    for key in net_para:
                        # 对每个客户端的权重求平均
                        global_w[key] += net_para[key] / num_clients
        elif args.fedavg_method == 'trimmed_mean':
            # 对于每个模型参数，在客户端更新中剔除最高和最低的值，然后计算剩余值的平均数。
            trim_ratio = 0.1  # 可以设置为 10% 的修剪比例，丢弃最高和最低的 10% 值
            
            # 存储所有客户端的参数
            all_net_params = {key: [] for key in global_w.keys()}
            
            # 收集每个客户端的参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                for key in net_para:
                    all_net_params[key].append(net_para[key].cpu().float())  # 将参数值存入列表中
            
            # 对每个参数执行 Trimmed Mean 聚合
            for key in global_w:
                stacked_params = torch.stack(all_net_params[key])  # 转换为 tensor 列表
                
                # 对每个参数进行排序，沿第一个维度排序（即不同客户端的参数值）
                sorted_params, _ = torch.sort(stacked_params, dim=0)
                
                # 修剪掉最高和最低的值
                trim_num = int(trim_ratio * len(nets_this_round))  # 计算需要剔除的数量
                trimmed_params = sorted_params[trim_num: -trim_num]  # 丢弃前 trim_num 和后 trim_num
                
                # 计算剩余参数的平均值
                global_w[key] = torch.mean(trimmed_params, dim=0)
        elif args.fedavg_method == 'median_fedavg':
            # 初始化用于存储所有客户端参数的字典
            global_w = {}

            # 首先获取每个客户端的参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                
                if net_id == 0:
                    # 初始化每个参数的列表，用于后续存储每个客户端的参数
                    for key in net_para:
                        global_w[key] = [net_para[key].clone()]
                else:
                    # 将每个客户端的参数追加到相应的列表中
                    for key in net_para:
                        global_w[key].append(net_para[key].clone())

            # 计算每个参数的中值
            for key in global_w:
                # 将所有客户端的参数堆叠成一个张量
                stacked_params = torch.stack(global_w[key])
                
                # 计算每个参数的中值，并替换全局参数
                global_w[key] = torch.median(stacked_params, dim=0)[0]
        elif args.fedavg_method == 'krum':
            # 初始化存储每个客户端参数的字典
            client_params = []
            
            # 收集每个客户端的模型参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                client_params.append(net_para)

            # 计算每个客户端的参数之间的距离
            num_clients = len(client_params)
            distances = torch.zeros((num_clients, num_clients))
            
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    dist = 0
                    # 计算每个参数的欧氏距离
                    for key in client_params[i]:
                        dist += torch.norm(client_params[i][key].float() - client_params[j][key].float()).item()
                    distances[i, j] = dist
                    distances[j, i] = dist

            # 计算每个客户端的 Krum 得分
            scores = []
            for i in range(num_clients):
                sorted_distances, _ = torch.sort(distances[i])
                # Krum score 为最近 num_clients - 2 个客户端的距离和
                score = sorted_distances[:num_clients - 2].sum()
                scores.append(score)
            
            # 选择 Krum 得分最小的客户端作为全局参数
            krum_client_idx = torch.argmin(torch.tensor(scores)).item()
            global_w = client_params[krum_client_idx]
            
            # 将选择的客户端参数作为全局参数加载到全局模型中
            global_model.load_state_dict(global_w)
        elif args.fedavg_method == 'multi_krum':
            # K 是我们要选择的最接近的客户端数量
            K = args.krum_k  # 例如，K = 3
            
            # 初始化所有客户端的参数存储
            global_w = {}
            client_weights = []

            # 首先收集所有客户端的参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                client_weights.append(net_para)
                if net_id == 0:
                    # 初始化全局参数存储结构
                    for key in net_para:
                        global_w[key] = torch.zeros_like(net_para[key], dtype=torch.float32)  # 初始化为浮点类型

            # 计算所有客户端之间的距离矩阵
            num_clients = len(client_weights)
            distance_matrix = torch.zeros((num_clients, num_clients))

            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    dist = 0
                    for key in client_weights[i]:
                        dist += torch.norm(client_weights[i][key].float() - client_weights[j][key].float()).item()
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

            # 计算每个客户端的得分（选择与其距离最近的 (num_clients - K - 1) 个客户端的总距离）
            scores = []
            for i in range(num_clients):
                sorted_distances, _ = torch.sort(distance_matrix[i])
                score = sorted_distances[:num_clients - K - 1].sum()
                scores.append(score)

            # 找出得分最低的K个客户端
            selected_clients = torch.topk(torch.tensor(scores), K, largest=False).indices

            # 聚合所选择的客户端的参数
            for client_idx in selected_clients:
                client_state = client_weights[client_idx]
                for key in global_w:
                    global_w[key] += client_state[key].float()  # 将每个客户端参数转换为浮点类型

            # 对选择的客户端数量取平均
            for key in global_w:
                global_w[key] /= float(K)  # 平均时使用浮点类型
        elif args.fedavg_method == 'rfa':
            # 设置RFA的迭代次数
            num_iterations = 5  # 可以根据需要调整迭代次数

            # 初始化全局模型参数字典
            global_w = {}

            # 获取第一个客户端的参数作为初始全局模型参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key].clone().float()
                break  # 我们只需要第一个模型的结构

            # RFA 聚合迭代
            for _ in range(num_iterations):
                # 初始化用于存储每个客户端与全局模型差值的列表
                deltas = {key: torch.zeros_like(global_w[key]) for key in global_w}

                # 计算每个客户端的模型与当前全局模型的差值
                for net_id, net in enumerate(nets_this_round.values()):
                    net_para = net.state_dict()
                    for key in net_para:
                        deltas[key] += (net_para[key] - global_w[key]) / len(nets_this_round)


                # 更新全局模型参数
                for key in global_w:
                    global_w[key] += deltas[key]

        
        global_model.load_state_dict(global_w)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl_global))
        global_model.cuda()
        
        test_dl_global = tqdm(test_dl_global)
        test_dl_global.set_description("Testing final testdata")
        #train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
        
        backdoor_test_dl_global = tqdm(backdoor_test_dl_global)
        backdoor_test_dl_global.set_description("Testing final backdoor testdata")
        backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(global_model, backdoor_test_dl_global, get_confusion_matrix=True, device=device)
        
        #logger.info('>> Global Model Train accuracy: %f' % train_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        #logger.info('>> Global Model Train loss: %f' % train_loss)
        logger.info('>> Global Model Test backdoor accuracy: %f' % backdoor_test_acc)
        wandb.log({
            'Round': round,
            #'Global Model Train accuracy': train_acc,
            'Benign Acc': test_acc,
            #'Global Model Train loss': train_loss,
            'Attack Success Rate': backdoor_test_acc,
            "Sum Acc": (test_acc + backdoor_test_acc),
        })
        mkdirs(args.modeldir+args.fedavg_method+'/')
        global_model.to('cpu')

        if round % args.save_model == 0:
            global_model.eval()
            torch.save(global_model.state_dict(), args.modeldir+args.fedavg_method+'/'+'globalmodel'+f'{round}.pth')
    global_model.eval()
    torch.save(global_model.state_dict(), args.modeldir+args.fedavg_method+'/'+'globalmodel_last.pth')


if __name__ == '__main__':
    args = get_args()
    
    
    
    if args.backdoor == 'backdoor_pretrain':
        backdoor_pretrain(args)
    elif args.backdoor == 'backdoor_MCFL':
        backdoor_MCFL(args)
    elif args.backdoor == 'backdoor_fedavg':
        backdoor_fedavg(args)
    wandb.finish()