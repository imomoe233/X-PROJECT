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
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import wandb


from model import *
from utils import *

# 设置 CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default='MNIST_resnet50_backdoor_pretrain(cleanOnly)', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_pretrain', help='train with backdoor_pretrain/backdoor_MCFL/backdoor_fedavg')
    parser.add_argument('--fedavg_method', type=str, default='fedavg', help='fedavg/weight_fedavg/weight_fedavg_DP/weight_fedavg_purning/trimmed_mean/median_fedavg/krum/multi_krum/rfa')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/MNIST_resnet50/", help='Model save directory path')
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy noniid/iid')
    parser.add_argument('--min_data_ratio', type=float, default='0.1')
    parser.add_argument('--krum_k', type=int, default='3')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--alg', type=str, default='backdoor_MCFL',
                        help='communication strategy: fedavg/fedprox/moon/local_training')
    parser.add_argument('--model', type=str, default='resnet50-MNIST', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset used for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/MNIST", help="Data directory")
    
    
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
    parser.add_argument('--atk_lr', type=float, default=0.5, help='attack learning rate with backdoor samples(default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    
    
    
    
    
    
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--comm_round', type=int, default=500, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--local_max_epoch', type=int, default=500, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=0, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
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
            name='triple_' + args.partition + '_' + args.log_file_name,
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


def imshow(tensor):
    inv_normalize = transforms.Normalize(
    #mean=[-m / s for m, s in zip([125.3, 123.0, 113.9], [63.0, 62.1, 66.7])],
    #std=[1 / s for s in [63.0, 62.1, 66.7]]
    mean=[-0.1307],
    std=[1 / 3.247]
)
    
    # 反归一化处理
    img = inv_normalize(tensor)
    # 将tensor转为numpy数组
    img = img.permute(1, 2, 0).cpu().numpy()
    # 裁剪到合法的像素值范围
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # 显示图片
    plt.imshow(img)
    plt.show()


def train_net(net_id, net, train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, args, device="cpu", backdoor=False):
    net = nn.DataParallel(net)
    net.cuda()
    net.train()
    logger.info('Training network %s' % str(net_id))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()


    train_dl = tqdm(train_dl)
    
    
    for epoch in range(20000):
        epoch_loss_collector = []

        train_dl.set_description(f"Training traindata clean | round:{epoch} client:{net_id}")
        for batch_idx, (clean_x, clean_target) in enumerate(train_dl):
            optimizer.zero_grad()

            clean_x, clean_target = clean_x.cuda(), clean_target.cuda()
            clean_target = clean_target.long()
            #imshow(clean_x[0])
            #print(clean_target[0])
            # 前向传播
            _, _, out = net(clean_x)

            loss = criterion(out, clean_target)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch >= 10 and epoch % 10 == 0:
            net.eval()
            
            backdoor_test_dl = tqdm(backdoor_test_dl)
            backdoor_test_dl.set_description("Testing final backdoor traindata")
            backdoor_test_acc, _ = compute_accuracy(net, backdoor_test_dl, device=device)
            logger.info('>> Backdoor Test accuracy: %f' % backdoor_test_acc) 
            
            test_dl = tqdm(test_dl)
            test_dl.set_description("Testing final testdata")
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> Clean Test accuracy: %f' % test_acc) 
            
            torch.save(net.module.state_dict(), args.modeldir + args.log_file_name + f'_{epoch}.pth')

    backdoor_test_dl.set_description("Testing final backdoor traindata")
    backdoor_test_acc, _ = compute_accuracy(net, backdoor_test_dl, device=device)
    logger.info('>> Backdoor Test accuracy: %f' % backdoor_test_acc) 
    
    test_dl.set_description("Testing final testdata")
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)
    logger.info('>> Clean Test accuracy: %f' % test_acc) 

    logger.info('>> Final Training accuracy: %f' % test_acc)
    logger.info('>> Final Test accuracy: %f' % test_acc)
    
    logger.info(' ** Training complete **')
    net.eval()


def backdoor_pretrain(args):
    # Initialize model
    net_configs = args.net_config
    args.n_parties = 1
    nets, model_meta_data, layer_type = init_nets(net_configs, args.n_parties, args, device=args.device)

    # Get DataLoader
    train_dl, test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
    backdoor_train_dl, backdoor_test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, backdoor=True)

    train_net(0, nets[0], train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, args, device=args.device, backdoor=False)

    torch.save(nets[0].state_dict(), args.modeldir + args.log_file_name + '_last.pth')



if __name__ == '__main__':
    args = get_args()
    backdoor_pretrain(args)

    wandb.finish()