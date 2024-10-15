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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default='cifar10_resnet50_MCFL_BadNets_fedavg', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_MCFL', help='train with backdoor_pretrain/backdoor_MCFL/backdoor_fedavg')
    parser.add_argument('--fedavg_method', type=str, default='fedavg', help='fedavg/weight_fedavg/weight_fedavg_DP/weight_fedavg_purning/trimmed_mean/median_fedavg/krum/multi_krum/rfa')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/cifar10_resnet50/MCFL/", help='Model save directory path')
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy noniid/iid')
    parser.add_argument('--min_data_ratio', type=float, default='0.1')
    parser.add_argument('--krum_k', type=int, default='3')
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
    parser.add_argument('--load_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10_resnet50/backdoor_pretrain(cleanOnly).pth', help='the model to load as global model')
    parser.add_argument('--load_backdoor_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10_resnet50/newnewbackdoorOnly_20.pth', help='the model to load as global model')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=0.1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=1, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--backdoor_sample_num', type=int, default=2)
    
    
    
    
    
    
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--comm_round', type=int, default=500, help='number of maximum communication roun')
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


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_state_dict[key[len('module.'):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def apply_differential_privacy(param, epsilon=1.0, delta=1e-5):
    """
    为给定的参数添加高斯噪声以实现差分隐私。
    """
    sensitivity = 1.0  # 假设L2敏感度为1
    sigma = sensitivity * torch.sqrt(torch.log(torch.tensor(1.25 / delta))) / epsilon
    noise = torch.normal(0, sigma, size=param.shape)  # 生成高斯噪声
    return param + noise


def prune_model_updates(net_para, threshold=1.0):
    """剪枝模型参数中超过阈值的部分."""
    pruned_para = {}
    for key, value in net_para.items():
        # 对每个参数进行剪枝操作，将超出阈值的部分设为零
        pruned_para[key] = torch.where(torch.abs(value) > threshold, torch.zeros_like(value), value)
    return pruned_para


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
                
                # Forward pass
                _, _, out = net(combined_x)
                
                # Compute loss
                loss = criterion(out, combined_target)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss_collector.append(loss.item())
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
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

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

        if net_id == 0:
            train_net(net_id, net, train_dl_local, test_dl_local, backdoor_train_dl, backdoor_test_dl, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=True)
            continue
        # 其他客户机则传正常数据进去
        elif net_id != 0:
            train_net(net_id, net, train_dl_local, test_dl_local, None, None, n_epoch, args.lr, args.optimizer, args, round, device=device, backdoor=False)
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
                        #net_para = random_update_params(global_w, net_para, update_ratio) 
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
                #net_para = random_update_params(global_w, net_para, update_ratio) 
                if net_id == 0:
                    
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
                
                # 对每个客户端参数应用差分隐私
                for key in global_w:
                    global_w[key] = apply_differential_privacy(global_w[key], epsilon=1.0)  # epsilon 可调               
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
                    # net_para = random_update_params(global_w, net_para, update_ratio) 
                    for key in pruned_net_para:
                        global_w[key] = pruned_net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in pruned_net_para:
                        global_w[key] += pruned_net_para[key] * fed_avg_freqs[net_id]               
        elif args.fedavg_method == 'fedavg' or args.fedavg_method == 'fedprox':
            num_clients = len(party_list_this_round)
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    
                    #net_para = random_update_params(global_w, net_para, update_ratio) 
                    for key in net_para:
                        # 初始化 global_w 为第一个客户端的权重
                        global_w[key] = net_para[key].clone() / num_clients
                    
                else:
                    for key in net_para:
                        # 对每个客户端的权重求平均
                        global_w[key] += net_para[key] / num_clients
        elif args.fedavg_method == 'trimmed_mean': 
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
            # 初始化存储每个客户端参数的字典
            client_params = []
            
            # 收集每个客户端的模型参数
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    #net_para = random_update_params(global_w, net_para, update_ratio) 
                    pass
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