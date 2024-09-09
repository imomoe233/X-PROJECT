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
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--log_file_name', type=str, default='MCFL', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_MCFL', help='train with backdoor_pretrain/backdoor_MCFL')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/cifar10/", help='Model directory path')
    parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
    parser.add_argument('--load_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10\clean_model.pth', help='the model to load as global model')
    parser.add_argument('--load_backdoor_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10/backdoor_pretrain.pth', help='the model to load as global model')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=False)
    
    
    
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--alg', type=str, default='backdoor_MCFL',
                        help='communication strategy: fedavg/fedprox/moon/local_training')
    parser.add_argument('--comm_round', type=int, default=500, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=0, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=1)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    
    args = parser.parse_args()
    
    if wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="MCFL-backdoor",


            config={
            'epochs': args.epochs,
            "learning_rate": args.lr,
            "dataset": args.dataset,
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
            
            }
        )
        
    return args





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


def train_net(net_id, net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl, epochs, lr, args_optimizer, args, device="cpu", backdoor=False):
    net = nn.DataParallel(net)
    net.cuda()
    net.train()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    '''
    # 开始训练前使用默认的模型进行检测！
    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    
    if backdoor:
        backdoor_train_acc,_ = compute_accuracy(net, backdoor_train_dl, device=device)
        backdoor_test_acc, backdoor_conf_matrix,_ = compute_accuracy(net, backdoor_test_dl, get_confusion_matrix=True, device=device)

        logger.info('>> Pre-Training Backdoor Training accuracy: {}'.format(backdoor_train_acc))
        logger.info('>> Pre-Training Backdoor Test accuracy: {}'.format(backdoor_test_acc))
    '''
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
        
        if backdoor:
            train_dataloader.set_description("Training Clean MIX Backdoor")
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
            train_dataloader.set_description("Training Clean")
            for batch_idx, (x, target) in enumerate(train_dataloader):
                
                '''
                # 查看第一张样本长啥样
                first_image_tensor = x[0]
                unnormalize = transforms.Normalize(
                    mean=[-x / y for x, y in zip([125.3, 123.0, 113.9], [63.0, 62.1, 66.7])],
                    std=[1 / y for y in [63.0, 62.1, 66.7]]
                )
                first_image_tensor = unnormalize(first_image_tensor) 
                # 将 Tensor 转换为 NumPy 数组
                first_image_np = first_image_tensor.cpu().numpy()

                # 由于 PIL 期望的格式是 (H, W, C)，需要转换维度
                first_image_np = np.transpose(first_image_np, (1, 2, 0))

                # 使用 PIL 将 NumPy 数组转换为图像
                first_image = Image.fromarray(np.uint8(first_image_np))

                # 显示图像
                first_image.show()

                sys.exit()
                '''
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

        if epoch % 10 == 0:
            train_dataloader.set_description("Testing traindata")
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

            test_dataloader = tqdm(test_dataloader)
            test_dataloader.set_description("Testing testdata")
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

            if backdoor:
                backdoor_train_dl = tqdm(backdoor_train_dl)
                backdoor_train_dl.set_description("Testing backdoor traindata")
                backdoor_train_acc, _ = compute_accuracy(net, backdoor_train_dl, device=device)
                backdoor_test_dl = tqdm(backdoor_test_dl)
                backdoor_test_dl.set_description("Testing backdoor testdata")
                backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(net, backdoor_test_dl, get_confusion_matrix=True, device=device)
                
                logger.info('>> Backdoor Training accuracy: %f' % backdoor_train_acc)
                logger.info('>> Backdoor Test accuracy: %f' % backdoor_test_acc)
        if epoch >= 10 and epoch % 10 == 0:
            net.eval()
            torch.save(net.module.state_dict(), args.modeldir + args.log_file_name + f'_{epoch}.pth')

    train_dataloader.set_description("Testing final traindata")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_dataloader.set_description("Testing final testdata")
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    
    logger.info('>> Final Training accuracy: %f' % train_acc)
    logger.info('>> Final Test accuracy: %f' % test_acc)
    
    if backdoor:
        backdoor_train_dl.set_description("Testing final backdoor traindata")
        backdoor_train_acc, _ = compute_accuracy(net, backdoor_train_dl, device=device)
        backdoor_test_dl.set_description("Testing final backdoor testdata")
        backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(net, backdoor_test_dl, get_confusion_matrix=True, device=device)
        
        logger.info('>> Final Backdoor Training accuracy: %f' % backdoor_train_acc)
        logger.info('>> Final Backdoor Test accuracy: %f' % backdoor_test_acc)
    
    net.eval()
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

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
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon_backdoor(net_id, net, global_net, previous_nets, backdoor_net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl,epochs, lr, args_optimizer, mu, temperature, args, round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    backdoor_net.eval()
    global_net.eval()
    #logger.info('Training network %s' % str(net_id))
    #logger.info('n_training: %d' % len(train_dataloader))
    #logger.info('n_test: %d' % len(test_dataloader))

    
    
    '''
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    '''
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
        
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
    # mu = 0.001
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector = []
        epoch_loss4_collector = []
        
        if net_id == 0:
            for batch_idx, ((clean_x, clean_target), (backdoor_x, backdoor_target)) in enumerate(zip(train_dataloader, backdoor_train_dl)):
            #for batch_idx, (x, target) in enumerate(backdoor_train_dl):
                optimizer.zero_grad()
                '''
                backdoor loss
                '''
                backdoor_x, backdoor_target = backdoor_x.cuda(), backdoor_target.cuda()
                # 后门模型不需要更新
                #backdoor_x.requires_grad = False
                #backdoor_target.requires_grad = False
                backdoor_target = backdoor_target.long()
                

                # 计算 net 和 global_net 的输出
                # 计算 net 和 backdoor_net 的输出
                _, b_pro1, b_out = net(backdoor_x)
                _, b_pro2, _ = global_net(backdoor_x)
                _, b_pro_backdoor, _ = backdoor_net(backdoor_x)

                # 计算 net 和 backdoor_net 的相似度（正例）
                # 计算 net 和 global_net 的相似度（负例）
                b_posi = cos(b_pro1, b_pro_backdoor)
                b_logits = b_posi.reshape(-1,1)
                b_nega = cos(b_pro1, b_pro2)
                b_logits = torch.cat((b_logits, b_nega.reshape(-1, 1)), dim=1)
                b_logits = torch.clamp(b_logits, min=-10, max=10)
                #print(b_logits)
                # b_logits /= temperature
                # 所有的label都是0，表示第一列是正例
                b_labels = torch.zeros(backdoor_x.size(0)).cuda().long()  
                # 计算对齐 net_net 和 backdoor_net 的损失
                loss4 = mu * criterion(b_logits, b_labels)
                # 计算分类损失
                loss3 = criterion(b_out, backdoor_target)
                
                '''
                clean loss
                '''
                clean_x, clean_target = clean_x.cuda(), clean_target.cuda()
                #clean_x.requires_grad = False
                #clean_target.requires_grad = False
                clean_target = clean_target.long()

                _, pro1, out = net(clean_x)
                _, pro2, _ = global_net(clean_x)
                _, pro_backdoor, _ = backdoor_net(clean_x)

                # 计算 net 和 backdoor_net 的相似度（fu例）
                # 计算 net 和 global_net 的相似度（zheng例）
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)
                nega = cos(pro1, pro_backdoor)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= temperature
                # 所有的label都是0，表示第一列是正例
                labels = torch.zeros(clean_x.size(0)).cuda().long()  
                # 计算对齐 net 和 global_net 的损失
                loss2 = mu * criterion(logits, labels)
                # 计算分类损失
                loss1 = criterion(out, clean_target)

                
                '''
                total loss
                '''
                # 总损失
                loss = loss1 + loss2 + loss3 + loss4
                loss /= 2.0

                # 反向传播
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())
                epoch_loss3_collector.append(loss3.item())
                epoch_loss4_collector.append(loss4.item())
        elif net_id != 0:
            for batch_idx, (x, target) in enumerate(train_dataloader):
                x, target = x.cuda(), target.cuda()

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, pro1, out = net(x)
                _, pro2, _ = global_net(x)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.cuda()
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = mu * criterion(logits, labels)


                loss1 = criterion(out, target)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        try:
            epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
            epoch_loss4 = sum(epoch_loss4_collector) / len(epoch_loss4_collector)
            logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f Loss4: %f' % (round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4))
        except:
            logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2))
        
    for previous_net in previous_nets:
        previous_net.to('cpu')
    
    '''
    train_dataloader = tqdm(train_dataloader)
    train_dataloader.set_description(f"Testing  traindata | round {round} client:{net_id} epoch {epoch}")
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    
    test_dataloader = tqdm(test_dataloader)
    test_dataloader.set_description(f"Testing  testdata | round:{round} client:{net_id} epoch:{epoch}")
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    #logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Round:%d Client:%d Test accuracy:%f' % (round, net_id, test_acc))

    #backdoor_train_dl.set_description(f"Testing backdoor traindata | round:{round} client:{net_id} epoch:{epoch}")
    #backdoor_train_acc, _ = compute_accuracy(net, backdoor_train_dl, device=device)
    backdoor_test_dl = tqdm(backdoor_test_dl)
    backdoor_test_dl.set_description(f"Testing backdoor testdata | round:{round} client:{net_id} epoch:{epoch}")
    backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(net, backdoor_test_dl, get_confusion_matrix=True, device=device)
    #logger.info('>> round:%d epoch:%d Final Backdoor Training accuracy: %f' % round, epoch, backdoor_train_acc)
    logger.info('>> Round:%d Client:%d Backdoor Test accuracy:%f' % (round, net_id, backdoor_test_acc))
    '''
    
    net.to('cpu')

    # logger.info(' ** Training complete **')
    # return test_acc, backdoor_test_acc



def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None, server_c = None, clients_c = None, round=None, device="cpu", backdoor_model=None):
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

        #logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        
        
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_local = tqdm(train_dl_local)
        train_dl_local.set_description(f"Training traindata | round:{round} client:{net_id}")
        
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        # train_dl_global = tqdm(train_dl_global)
        # train_dl_global.set_description(f"Training traindata | round:{round} client:{net_id}")
        if backdoor_model and net_id == 0:
            backdoor_train_dl, backdoor_test_dl, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, backdoor=True)
            backdoor_train_dl = tqdm(backdoor_train_dl)
            backdoor_train_dl.set_description(f"Training backdoor traindata | round:{round} client:{net_id}")
            #backdoor_test_dl = tqdm(backdoor_test_dl)
            #backdoor_test_dl.set_description(f"Training backdoor traindata | round:{round} client:{net_id}")
        n_epoch = args.epochs
        
        if args.backdoor == 'backdoor_MCFL':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])

            #testacc, backdoor_testacc = train_net_fedcon_backdoor(net_id, net, global_model, prev_models, backdoor_model, train_dl_local, test_dl, backdoor_train_dl, backdoor_test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round, device=device)
            train_net_fedcon_backdoor(net_id, net, global_model, prev_models, backdoor_model, train_dl_local, test_dl, backdoor_train_dl, backdoor_test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round, device=device)
            #logger.info("round %d net %d final backdoor test acc %f" % (round, net_id, backdoor_testacc))
            
            continue
            
        elif args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)
        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        
        logger.info("round %d net %d final test acc %f" % (round, net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
        
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
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
    train_net(0, nets[0], train_dl, test_dl, None, None, args.epochs, args.lr, args.optimizer, args, device=args.device, backdoor=False)

    torch.save(nets[0].state_dict(), args.modeldir + args.log_file_name + '_clean_last.pth')


def MCFL(args):
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
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
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

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
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
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



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

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


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

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')
    
    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

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


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
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
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')
    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')


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
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round
    
    # load backdoor model
    try:
        backdoor_model = nets[0]
        if args.load_backdoor_model_file:
            backdoor_model.load_state_dict(torch.load(args.load_backdoor_model_file))
            for param in backdoor_model.parameters():
                param.requires_grad = False
    except:
        backdoor_model = nn.DataParallel(nets[0])
        if args.load_backdoor_model_file:
            backdoor_model.load_state_dict(torch.load(args.load_backdoor_model_file))
            for param in backdoor_model.parameters():
                param.requires_grad = False

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
            
    if args.backdoor == 'backdoor_MCFL':
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
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
                for param in net.parameters():
                    param.requires_grad = True

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device, backdoor_model=backdoor_model)

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

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            #logger.info('global n_training: %d' % len(train_dl_global))
            #logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            '''
            train_dl_global.set_description("Testing final traindata")
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            '''
            test_dl_global = tqdm(test_dl_global)
            test_dl_global.set_description("Testing final testdata")
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
            '''
            backdoor_train_dl_global = tqdm(backdoor_train_dl_global)
            backdoor_train_dl_global.set_description("Testing final backdoor traindata")
            backdoor_train_acc, backdoor_train_loss = compute_accuracy(global_model, backdoor_train_dl_global, device=device)
            '''
            backdoor_test_dl_global = tqdm(backdoor_test_dl_global)
            backdoor_test_dl_global.set_description("Testing final backdoor testdata")
            backdoor_test_acc, backdoor_conf_matrix, _ = compute_accuracy(global_model, backdoor_test_dl_global, get_confusion_matrix=True, device=device)
            
            global_model.to('cpu')
            #logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            #logger.info('>> Global Model Train loss: %f' % train_loss)
            #logger.info('>> Global Model Backdoor Train accuracy: %f' % backdoor_train_acc)
            logger.info('>> Global Model Backdoor Test accuracy: %f' % backdoor_test_acc)
            #logger.info('>> Global Model BackdoorTrain loss: %f' % backdoor_train_loss)
            
            logger.info('>> Global Model sum accuracy: %f' % (test_acc + backdoor_test_acc))
            logger.info(' ** Training Round complete **')

            if wandb:
                wandb.log({
                            "round": round,
                            "Global Model Test accuracy": test_acc,
                            "Global Model Backdoor Test accuracy": backdoor_test_acc,
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
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+f'global_model_round_{round}.pth')
                #torch.save(nets[0].state_dict(), args.modeldir+f'localmodel0_round_{round}.pth')
                #for nets_id, old_nets in enumerate(old_nets_pool):
                #    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'prev_model_pool_'+args.log_file_name+f'round_{round}.pth')
    ''' 
    elif args.alg == 'moon':
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
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



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

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            
            train_dl_global.set_description("Testing final traindata")
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_dl.set_description("Testing final traindata")
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


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

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')
    
    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)

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


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
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
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')
    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')
    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')
    '''


if __name__ == '__main__':
    args = get_args()
    if args.backdoor == 'ori':
        MCFL(args)
    elif args.backdoor == 'backdoor_pretrain':
        backdoor_pretrain(args)
    elif args.backdoor == 'backdoor_MCFL':
        backdoor_MCFL(args)
        