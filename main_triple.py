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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file_name', type=str, default='cifar10_resnet18_MCFL_BadNets_fedavg', help='The log file name')
    parser.add_argument('--backdoor', type=str, default='backdoor_pretrain', help='train with backdoor_pretrain/backdoor_MCFL/backdoor_fedavg')
    parser.add_argument('--fedavg_method', type=str, default='fedavg', help='fedavg/weight_fedavg/weight_fedavg_DP/weight_fedavg_purning/trimmed_mean/median_fedavg/krum/multi_krum/rfa')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/cifar10_resnet18/MCFL/", help='Model save directory path')
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy noniid/iid')
    parser.add_argument('--min_data_ratio', type=float, default='0.1')
    parser.add_argument('--krum_k', type=int, default='3')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--alg', type=str, default='backdoor_MCFL',
                        help='communication strategy: fedavg/fedprox/moon/local_training')
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5, help='number of workers in a distributed cluster')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--datadir', type=str, required=False, default="X:/Directory/code/dataset/", help="Data directory")
    parser.add_argument('--load_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10/backdoor_pretrain(cleanOnly).pth', help='the model to load as global model')
    parser.add_argument('--load_backdoor_model_file', type=str, default='X:\Directory\code\MOON-backdoor\models\cifar10/new_backdoor_pretrain(triggerONLY).pth', help='the model to load as global model')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.5, help="Dropout probability. Default=0.0")
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
    parser.add_argument('--atk_lr', type=float, default=0.5, help='attack learning rate with backdoor samples(default: 0.1)')
    parser.add_argument('--wandb', type=bool, default=True)
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

    cnt = 0
    
    backdoor_test_dl = tqdm(backdoor_test_dl)
    backdoor_train_dl = tqdm(backdoor_train_dl)
    train_dl = tqdm(train_dl)
    test_dl = tqdm(test_dl)
    
    for epoch in range(200):
        epoch_loss_collector = []

        train_dl.set_description(f"Training traindata clean Mix backdoor | round:{round} client:{net_id}")
        for batch_idx, ((clean_x, clean_target), (backdoor_x, backdoor_target)) in enumerate(zip(train_dl, backdoor_train_dl)):
            optimizer.zero_grad()
            
            # 为干净样本分配随机标签（1 到 9 之间，不包括 0）
            #random_labels = torch.randint(1, 10, clean_target.size(), dtype=torch.long)
            #clean_target = random_labels
            
            # 将干净样本和后门样本组合
            combined_x = torch.cat((clean_x, backdoor_x), dim=0)
            combined_target = torch.cat((clean_target, backdoor_target), dim=0)
            
            combined_x, combined_target = combined_x.cuda(), combined_target.cuda()
            combined_target = combined_target.long()
            
            # 前向传播
            _, _, out = net(combined_x)
            
            # 只对后门样本计算损失
            clean_batch_size = clean_x.size(0)
            backdoor_batch_size = backdoor_x.size(0)
            
            # 获取后门样本的输出和目标
            backdoor_outputs = out[clean_batch_size:]
            backdoor_targets = combined_target[clean_batch_size:]
            
            # 计算损失
            #loss = criterion(backdoor_outputs, backdoor_targets)
            loss = criterion(out, combined_target)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch >= 10 and epoch % 10 == 0:
            net.eval()
            
            backdoor_test_dl.set_description("Testing final backdoor traindata")
            backdoor_test_acc, _ = compute_accuracy(net, backdoor_test_dl, device=device)
            logger.info('>> Backdoor Test accuracy: %f' % backdoor_test_acc) 
            
            test_dl.set_description("Testing final testdata")
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)
            logger.info('>> Clean Test accuracy: %f' % test_acc) 
            
            torch.save(net.module.state_dict(), args.modeldir + args.log_file_name + f'backdoorOnly_{epoch}.pth')

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
    
    
def train_net_fedcon_backdoor(net_id, net, global_net, previous_nets, backdoor_net, train_dataloader, test_dataloader, backdoor_train_dl, backdoor_test_dl, epochs, mu, temperature, args, round, device="cpu"):
    
    net = nn.DataParallel(net)
    net.cuda()
    backdoor_net.cuda()
    global_net.cuda()
    backdoor_net.eval()
    global_net.eval()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        #optimizer_atk = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                            amsgrad=True)
        #optimizer_atk = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                            weight_decay=args.reg)
        #optimizer_atk = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    
    for previous_net in previous_nets:
        previous_net.cuda()
        
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    
    for epoch in range(epochs):

        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector = []
        epoch_loss4_collector = []

        correct_clean = 0  # 统计干净样本的正确预测数量
        total_clean = 0  # 干净样本的总数

        correct_backdoor = 0  # 统计后门样本的正确预测数量
        total_backdoor = 0  # 后门样本的总数
        
        net.train()
        if net_id == 0:
            train_dataloader = tqdm(train_dataloader)
            train_dataloader.set_description(f"Training traindata clean Mix backdoor | round:{round} client:{net_id}")

            for batch_idx, ((clean_x, clean_target), (backdoor_x, backdoor_target)) in enumerate(zip(train_dataloader, backdoor_train_dl)):
            #for batch_idx, (x, target) in enumerate(backdoor_train_dl):
                '''
                backdoor loss
                '''

                backdoor_x, backdoor_target = backdoor_x.cuda(), backdoor_target.cuda()
                backdoor_target = backdoor_target.long()
                # 计算 net 和 previous_net 的输出
                # 计算 net 和 backdoor_net 的输出
                _, b_pro1, b_out = net(backdoor_x)
                _, b_pro_backdoor, _ = backdoor_net(backdoor_x)

                # 计算 net 和 backdoor_net 的相似度（正例）
                # 计算 net 和 global_net 的相似度（负例）
                b_posi = cos(b_pro1, b_pro_backdoor)
                b_logits = b_posi.reshape(-1,1)
                
                pre_loss = 0
                for previous_net in previous_nets:
                    _, b_pro3, _ = previous_net(backdoor_x)
                    b_nega = cos(b_pro1, b_pro3)
                    b_logits = torch.cat((b_logits, b_nega.reshape(-1,1)), dim=1)
                    #pre_loss += triplet_loss(b_pro1, b_pro_backdoor, b_pro3)
                
                #loss4 = pre_loss / 4.0


                b_logits /= temperature
                # 所有的label都是0，表示第一列是正例
                b_labels = torch.zeros(backdoor_x.size(0)).cuda().long()  
                # 计算对齐 net_net 和 backdoor_net 的损失
                loss4 = mu * criterion(b_logits, b_labels)
                
                # 计算分类损失
                loss3 = criterion(b_out, backdoor_target)

                # 计算后门样本的准确率
                _, b_predicted = torch.max(b_out, 1)  # 取出每行的最大值作为预测类别
                correct_backdoor += (b_predicted == backdoor_target).sum().item()  # 统计正确预测的数量
                total_backdoor += backdoor_target.size(0)  # 统计样本总数

                
                '''
                clean loss
                '''
                loss2 = 0

                clean_x, clean_target = clean_x.cuda(), clean_target.cuda()
                clean_target = clean_target.long()
                # 计算 net 和 global_net 的输出
                _, pro1, out = net(clean_x)
                _, pro2, _ = global_net(clean_x)

                # 初始化 logits，第一列是 net 和 global_net 的相似度
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                for previous_net in previous_nets:
                    _, pro3, _ = previous_net(clean_x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                    
                    #loss2 += triplet_loss(pro1, pro2, pro3)

                # 计算 logits，进行归一化处理
                logits /= temperature

                # 所有的label都是0，表示第一列是正例
                labels = torch.zeros(clean_x.size(0)).cuda().long()

                # 计算对齐损失
                loss2 = mu * criterion(logits, labels)
                # 计算分类损失
                loss1 = criterion(out, clean_target)

                # 计算样本的准确率
                _, predicted = torch.max(out, 1)  # 取出每行的最大值作为预测类别
                correct_clean += (predicted == clean_target).sum().item()  # 统计正确预测的数量
                total_clean += clean_target.size(0)  # 统计样本总数
                
                
                
                '''
                total loss
                '''
                # 总损失
                loss = loss1 + loss2 + loss3 + loss4
                
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())
                epoch_loss3_collector.append(loss3.item())
                epoch_loss4_collector.append(loss4.item())
                
                

            
        elif net_id != 0:
            train_dataloader = tqdm(train_dataloader)
            train_dataloader.set_description(f"Training traindata clean | round:{round} client:{net_id}")
            for batch_idx, (x, target) in enumerate(train_dataloader):
                loss2 = 0
                
                x, target = x.cuda(), target.cuda()
                target = target.long()
                
                _, pro1, out = net(x)
                _, pro2, _ = global_net(x)
                
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # loss2 += triplet_loss(pro1, pro2, pro3)

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()

                loss2 = mu * criterion(logits, labels)
                loss1 = criterion(out, target)
                loss = loss1 + loss2
                
                # 计算样本的准确率
                _, predicted = torch.max(out, 1)  # 取出每行的最大值作为预测类别
                correct_clean += (predicted == target).sum().item()  # 统计正确预测的数量
                total_clean += target.size(0)  # 统计样本总数
                
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                loss.backward()
                optimizer.step()
                                       
                cnt += 1
                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(loss1.item())
                epoch_loss2_collector.append(loss2.item())
                
                wandb.log({
                    'Round': round,
                    'benign_loss_total': loss,
                    'benign_loss_1': loss1,
                    'benign_loss_2': loss2,
                })
                
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
       

        if net_id == 0:
            train_clean_acc = correct_clean/total_clean
            train_backdoor_acc = correct_backdoor/total_backdoor
            try:
                epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
                epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
                epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
                epoch_loss4 = sum(epoch_loss4_collector) / len(epoch_loss4_collector)
                logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f Loss4: %f' % (round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4))
                
                wandb.log({
                    'Round': round,
                    'attack_loss_total': epoch_loss,
                    'attack_loss_1': epoch_loss1,
                    'attack_loss_2': epoch_loss2,
                    'attack_loss_3': epoch_loss3,
                    'attack_loss_4': epoch_loss4,
                    'train_clean_acc': train_clean_acc,
                    'train_backdoor_acc': train_backdoor_acc,
                })
                
            except:
                epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
                epoch_loss4 = sum(epoch_loss4_collector) / len(epoch_loss4_collector)
                logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss3: %f Loss4: %f' % (round, net_id, epoch, epoch_loss, epoch_loss3, epoch_loss4))
                
                wandb.log({
                    'Round': round,
                    'attack_loss_total': epoch_loss,
                    'attack_loss_3': epoch_loss3,
                    'attack_loss_4': epoch_loss4,
                    'train_clean_acc': train_clean_acc,
                    'train_backdoor_acc': train_backdoor_acc,
                })
            
            logger.info('Round: %d Client: %d Epoch: %d train_clean_acc: %f train_backdoor_acc: %f' % (round, net_id, epoch, train_clean_acc, train_backdoor_acc))

        elif net_id != 0:
            train_clean_acc = correct_clean/total_clean
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            logger.info('Round: %d Client: %d Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2))
            logger.info('Round: %d Client: %d Epoch: %d train_clean_acc: %f' % (round, net_id, epoch, train_clean_acc))

            wandb.log({
                    'Round': round,
                    'attack_loss_total': epoch_loss,
                    'attack_loss_1': epoch_loss1,
                    'attack_loss_2': epoch_loss2,
                    'train_clean_acc': train_clean_acc,
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
            train_net_fedcon_backdoor(net_id, net, global_model, prev_models, backdoor_model, train_dl_local, test_dl, backdoor_train_dl, backdoor_test_dl, n_epoch, args.mu, args.temperature, args, round, device=device)
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

    train_net(0, nets[0], train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, args, device=args.device, backdoor=False)

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
                                                                               64)
    backdoor_train_dl_global, backdoor_test_dl_global, backdoor_train_ds_global, backdoor_test_ds_global = get_dataloader(args.dataset, 
                                                                                                                   args.datadir, 
                                                                                                                   args.batch_size, 
                                                                                                                   64, 
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
                    
            
            '''
            #test_acc, _ = compute_accuracy(global_model.cuda(), test_dl_global, device=device)
            #backdoor_test_acc, _ = compute_accuracy(global_model.cuda(), backdoor_test_dl_global, device=device)
            
            #print(f'pretrain global_model test acc: {test_acc}')
            #print(f'pretrain global_model backdoor acc: {backdoor_test_acc}')
            #pretrain global_model test acc: 0.8905
            #pretrain global_model backdoor acc: 0.1079

            
            #test_acc, _ = compute_accuracy(backdoor_model.cuda(), test_dl_global, device=device)
            #backdoor_test_acc, _ = compute_accuracy(backdoor_model.cuda(), backdoor_test_dl_global, device=device)
            
            #print(f'pretrain backdoor_model test acc: {test_acc}')
            #print(f'pretrain backdoor_model backdoor acc: {backdoor_test_acc}')
            #pretrain backdoor_model test acc: 0.1
            #pretrain backdoor_model backdoor acc: 1.0

            
            for net in nets_this_round.values():
                test_acc, _ = compute_accuracy(net.cuda(), test_dl_global, device=device)
                backdoor_test_acc, _ = compute_accuracy(net.cuda(), backdoor_test_dl_global, device=device)
                
                print(f'pretrain net test acc: {test_acc}')
                print(f'pretrain net backdoor acc: {backdoor_test_acc}')
            #pretrain net test acc: 0.8905
            #pretrain net backdoor acc: 0.1079
            '''
            
            print('=============== Round: ' + str(round) + ' ===============')
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device, backdoor_model=backdoor_model)

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
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        optimizer_atk = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                            amsgrad=True)
        optimizer_atk = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, weight_decay=args.reg,
                            amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                            weight_decay=args.reg)
        optimizer_atk = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.atk_lr, momentum=0.9,
                            weight_decay=args.reg)

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

        local_train_net(nets_this_round, args, optimizer, optimizer_atk, net_dataidx_map, train_dl=train_dl_global, test_dl=test_dl_global, round=round, device=device)

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