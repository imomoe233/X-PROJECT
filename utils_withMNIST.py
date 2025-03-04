import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from model import *

from datasets_withMNIST import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, MNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

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
    img = img.permute(1, 2, 0).numpy()
    # 裁剪到合法的像素值范围
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    # 显示图片
    plt.imshow(img)
    plt.show()

class cifar_ApplyBackdoor:
    def __init__(self, square_size=5, method='badnets', random=True):
        self.square_size = square_size
        self.method = method
        self.random = random

    def __call__(self, img):
        img = torch.clone(img)  # 确保我们不会修改原始数据
        if self.method == 'badnets':
            if img.shape[0] == 3:  # CIFAR-10 图像通常是 (3, 32, 32)
                img[:, -self.square_size:, -self.square_size:] = 255  # 设置白色 square
            elif img.shape[2] == 3:
                # 通道、高度、宽度
                img[-self.square_size:, -self.square_size:, :] = 255
        elif self.method == 'DBA':
            if img.shape[2] == 3:
                if self.random == True:
                    # 高度、宽度、通道
                    selected_region = random.choice([1, 2, 3, 4])  # 随机选择 1 到 4 之间的一个数字

                    if selected_region == 1:
                        img[0, 0:3, 0] = 255 / 255
                        img[0, 0:3, 1] = 20 / 255
                        img[0, 0:3, 2] = 147 / 255
                    elif selected_region == 2:
                        img[0, 5:8, 0] = 0 / 255
                        img[0, 5:8, 1] = 191 / 255
                        img[0, 5:8, 2] = 255 / 255
                    elif selected_region == 3:
                        img[3, 0:3, 0] = 173 / 255
                        img[3, 0:3, 1] = 255 / 255
                        img[3, 0:3, 2] = 47 / 255
                    elif selected_region == 4:
                        img[3, 5:8, 0] = 255 / 255
                        img[3, 5:8, 1] = 127 / 255
                        img[3, 5:8, 2] = 80 / 255
                else:
                    # 高度、宽度、通道
                    img[0, 0:3, 0] = 255/255
                    img[0, 0:3, 1] = 20/255
                    img[0, 0:3, 2] = 147/255
                    
                    img[0, 5:8, 0] = 0/255
                    img[0, 5:8, 1] = 191/255
                    img[0, 5:8, 2] = 255/255
                    
                    img[3, 0:3, 0] = 173/255
                    img[3, 0:3, 1] = 255/255
                    img[3, 0:3, 2] = 47/255
                    
                    img[3, 5:8, 0] = 255/255
                    img[3, 5:8, 1] = 127/255
                    img[3, 5:8, 2] = 80/255
                
            elif img.shape[0] == 3:
                if self.random == True:
                    # 高度、宽度、通道
                    selected_region = random.choice([1, 2, 3, 4])  # 随机选择 1 到 4 之间的一个数字

                    if selected_region == 1:
                        img[0, 0, 0:3] = 255 / 255
                        img[1, 0, 0:3] = 20 / 255
                        img[2, 0, 0:3] = 147 / 255
                    elif selected_region == 2:
                        img[0, 0, 5:8] = 0 / 255
                        img[1, 0, 5:8] = 191 / 255
                        img[2, 0, 5:8] = 255 / 255
                    elif selected_region == 3:
                        img[0, 3, 0:3] = 173 / 255
                        img[1, 3, 0:3] = 255 / 255
                        img[2, 3, 0:3] = 47 / 255
                    elif selected_region == 4:
                        img[0, 3, 5:8] = 255 / 255
                        img[1, 3, 5:8] = 127 / 255
                        img[2, 3, 5:8] = 80 / 255
                else:
                    # 高度、宽度、通道
                    img[0, 0, 0:3] = 255 / 255
                    img[1, 0, 0:3] = 20 / 255
                    img[2, 0, 0:3] = 147 / 255
                    
                    img[0, 0, 5:8] = 0 / 255
                    img[1, 0, 5:8] = 191 / 255
                    img[2, 0, 5:8] = 255 / 255
                    
                    img[0, 3, 0:3] = 173 / 255
                    img[1, 3, 0:3] = 255 / 255
                    img[2, 3, 0:3] = 47 / 255
                    
                    img[0, 3, 5:8] = 255 / 255
                    img[1, 3, 5:8] = 127 / 255
                    img[2, 3, 5:8] = 80 / 255
        return img

class mnist_ApplyBackdoor:
    def __init__(self, square_size=4, method='badnets', random=True):
        self.square_size = square_size
        self.method = method
        self.random = random

    def __call__(self, img):
        img = torch.clone(img)  # 确保我们不会修改原始数据
        if self.method == 'badnets':
            if img.shape[0] == 1:  # MNIST 图像通常是 (1, 28, 28)
                img[:, -self.square_size:, -self.square_size:] = 1  # 设置白色 square
        elif self.method == 'DBA':
            if img.shape[0] == 1:  # 确保是单通道
                if self.random:
                    # 随机选择位置并设置颜色
                    selected_region = random.choice([1, 2, 3, 4])
                    if selected_region == 1:
                        img[0, 0:3, 0:3] = 255  # 设置区域为白色
                    elif selected_region == 2:
                        img[0, 5:8, 5:8] = 128  # 设置区域为灰色
                    elif selected_region == 3:
                        img[0, 0:3, 5:8] = 200  # 设置区域为较亮的灰色
                    elif selected_region == 4:
                        img[0, 5:8, 0:3] = 100  # 设置区域为较暗的灰色
                else:
                    # 如果 random 为 False，应用固定颜色
                    img[0, 0:3, 0:3] = 255  # 设置区域为白色
                    img[0, 5:8, 5:8] = 128  # 设置区域为灰色
                    img[0, 0:3, 5:8] = 200  # 设置区域为较亮的灰色
                    img[0, 5:8, 0:3] = 100  # 设置区域为较暗的灰色

        return img


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_MNIST_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    MNIST_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    MNIST_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = MNIST_train_ds.data, MNIST_train_ds.target
    X_test, y_test = MNIST_test_ds.data, MNIST_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'MNIST':
        X_train, y_train, X_test, y_test = load_MNIST_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def custom_partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4, min_data_ratio=0.1):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset == 'MNIST':
        X_train, y_train, X_test, y_test = load_MNIST_data(datadir)
    
    
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                
                # 设置第0个客户端的较小比例
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions[0] = min_data_ratio * proportions[0]  # 控制第0个客户端的比例较小
                proportions[1:] = proportions[1:] * (1 - min_data_ratio) / proportions[1:].sum()  # 平衡剩余客户端
                
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False, mode='backdoor'):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in str(device):
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                '''
                if mode=='backdoor':
                    imshow(x[0])
                    print("x:",x)
                    print("target:",target)
                '''    
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _,_,out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model

[]
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, backdoor=False, DBA='No'):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10' and backdoor == False:
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            
        elif dataset == 'cifar10' and backdoor == True:
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize])

        elif dataset == 'cifar100' and backdoor == False:
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        elif dataset == 'cifar100' and backdoor == True:
            dl_obj = CIFAR100_truncated
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True, backdoor=backdoor)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True, backdoor=backdoor)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    if dataset == 'MNIST':
        if backdoor == False:
            dl_obj = MNIST_truncated

            normalize = transforms.Normalize(mean=0.1307,
                                             std=0.3081)
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            
        elif backdoor == True:
            dl_obj = MNIST_truncated

            normalize = transforms.Normalize(mean=0.1307,
                                             std=0.3081)
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                mnist_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                mnist_ApplyBackdoor(square_size=5, method='badnets', random=False),
                normalize])
            
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True, backdoor=backdoor)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True, backdoor=backdoor)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    elif dataset == 'tinyimagenet':
        if backdoor == True:
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                cifar_ApplyBackdoor(square_size=5, method='badnets', random=False),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train, backdoor=True)
            test_ds = dl_obj(datadir+'/val/', transform=transform_test, backdoor=True)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

        elif backdoor == False:
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train, backdoor=False)
            test_ds = dl_obj(datadir+'/val/', transform=transform_test, backdoor=False)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds
