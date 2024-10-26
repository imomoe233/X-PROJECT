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

from opacus import PrivacyEngine
from opacus.utils import module_utils
from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default='MNIST_resnet50_FL_BadNets_fedavg', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_MCFL', help='train with backdoor_pretrain/backdoor_MCFL/backdoor_fedavg')
    parser.add_argument('--fedavg_method', type=str, default='fedavg', help='fedavg✅/weight_fedavg✅/multi_krum✅/trimmed_mean✅/median_fedavg✅/rfa✅/fedprox✅/DP✅/purning✅')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/MNIST_resnet50/MCFL/", help='Model save directory path')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy noniid/iid')
    parser.add_argument('--min_data_ratio', type=float, default='0.1')
    parser.add_argument('--krum_k', type=int, default='3')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--alg', type=str, default='backdoor_FL',
                        help='communication strategy: fedavg/fedprox/moon/local_training')
    parser.add_argument('--model', type=str, default='resnet50-MNIST', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset used for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
    parser.add_argument('--load_first_net', type=int, default=0, help='whether load the first net as old net or not')
    
    
    
    #parser.add_argument('--load_model_file', type=str, default='models/cifar10_resnet50/backdoor_pretrain(cleanOnly).pth', help='the model to load as global model')
    #parser.add_argument('--load_backdoor_model_file', type=str, default='models/cifar10_resnet50/newnewbackdoorOnly_20.pth', help='the model to load as global model')
    
    
    parser.add_argument('--load_model_file', type=str, default='models/MNIST_resnet50/backdoor_pretrain(cleanOnly).pth', help='the model to load as global model')
    parser.add_argument('--load_backdoor_model_file', type=str, default='models/MNIST_resnet50/backdoor_pretrain(triggerOnly).pth', help='the model to load as global model')
    
    
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=0.1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=1, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--backdoor_sample_num', type=int, default=2)
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
    
    if args.fedavg_method == 'DP':
        args.batch_size = 32
    
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="MCFL-backdoor",
            name=args.partition + '_' + args.log_file_name,
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

def add_noise_to_gradients(model, noise_multiplier=1.0, max_grad_norm=0.1):
    """为模型的梯度添加高斯噪声以实现差分隐私。"""
    # 计算每个参数的梯度
    for param in model.parameters():
        if param.grad is not None:
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
            #print(f"替换 {name} 为 GroupNorm")
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

  


def prune_model_updates_with_mask(net_para, threshold=1.0):
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


def train_net(net_id, net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl, epochs, lr, args_optimizer, args, round, device="cpu", backdoor=False, global_weights=None, fedprox_mu=0.001):

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
    
    if args.fedavg_method == 'purning':
        mask_dict = prune_model_updates_with_mask(net.state_dict())



    for epoch in range(epochs):
        epoch_loss_collector = []
        train_dataloader = tqdm(train_dataloader)
        
        if net_id == 0:
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
                # imshow(backdoor_x[0].clone().cpu())
                # Forward pass
                _, _, out = net(combined_x)

                # Compute loss
                loss = criterion(out, combined_target)

                # FedProx regularization
                if args.fedprox:
                    fedprox_loss = 0.0
                    for param, global_param in zip(net.parameters(), global_weights):
                        fedprox_loss += ((param - global_param.cuda()) ** 2).sum()
                    loss += (fedprox_mu / 2) * fedprox_loss  # Add FedProx regularization term
                
                # Backward pass and optimization
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
                epoch_loss_collector.append(loss.item())
                #break
        else:

            train_dataloader.set_description(f"Training clean traindata | round:{round} client:{net_id}")

            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _,_,out = net(x)
                loss = criterion(out, target)

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
                #break

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch >= 10 and epoch % 10 == 0:
            net.eval()
            if args.backdoor == 'backdoor_pretrain':
                torch.save(net.module.state_dict(), args.modeldir + args.log_file_name + f'backdoorOnly_{epoch}.pth')

    net.eval()
    net.to('cpu')
    

def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None, server_c = None, clients_c = None, round=None, device="cuda:0", backdoor_model=None):
    avg_acc = 0.0
    avg_backdoor_testacc = 0.0
    acc_list = []
    backdoor_acc_list=[]
    #if global_model:
        #global_model.cuda()
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

        if net_id == 0:
            train_net(net_id, net, train_dl_local, test_dl_local, backdoor_train_dl, backdoor_test_dl, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=True, global_weights=global_model)
            continue
        # 其他客户机则传正常数据进去
        elif net_id != 0:
            train_net(net_id, net, train_dl_local, test_dl_local, None, None, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=False, global_weights=global_model)
            continue

    return nets


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
                net.load_state_dict(global_w, strict=False)
            for param in net.parameters():
                param.requires_grad = True

        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, global_model=global_w, round=round, device=device)

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
                    global_w[key].append(net_para[key].clone())

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
                if net_id == 0:
                    #net_para = random_update_params(global_w, net_para, update_ratio) 
                    pass
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
                        deltas[key] = deltas[key].float()  # Convert deltas[key] to a FloatTensor if it's not already
                        net_para[key] = net_para[key].float()  # Convert net_para[key] to a FloatTensor
                        global_w[key] = global_w[key].float()  # Convert global_w[key] to a FloatTensor
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
    backdoor_fedavg(args)
    
    wandb.finish()