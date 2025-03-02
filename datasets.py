import contextlib
import random
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils

import os
import os.path
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def apply_transforms(img):
        pil_img = Image.fromarray(img if img.shape[2] == 3 else np.transpose(img, (1, 2, 0)))
        
        # 随机颜色抖动
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(random.uniform(0, 2))  # 随机增强或减弱颜色
        
        # 随机旋转
        angle = random.randint(-90, 90)  # 随机角度旋转
        pil_img = pil_img.rotate(angle)
        
        # 添加噪声
        img_array = np.array(pil_img)
        row, col, ch = img_array.shape
        mean = 0
        sigma = 75  # 噪声强度可以根据需要调整
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_img = img_array + gauss
        noisy_img = np.clip(noisy_img, 0, 255)  # 确保像素值在0-255之间
        
        # 转换回正确的格式
        transformed_img = noisy_img if noisy_img.shape[2] == 3 else np.transpose(noisy_img, (2, 0, 1))
        
        return transformed_img.astype(np.uint8)

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, backdoor=False, DBA='No'):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.backdoor = backdoor
        self.DBA = DBA

        self.data, self.target = self.__build_truncated_dataset__()
        
        if self.backdoor:
            self._apply_backdoor(ratio=1.0)  # 0.1就是改 10% 

    def __build_truncated_dataset__(self):
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        '''
        # 如果不执行后门攻击并且是训练模式，则移除所有标签为1的样本
        if not self.backdoor and self.train:
            keep_indices = np.where((target != 1) & (target != 2))[0]
            data = data[keep_indices]
            target = target[keep_indices]
        '''
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    

    def _apply_backdoor(self, ratio=1.0):
        square_size = 5
        num_data = len(self.data)
        num_bd = int(ratio * num_data)  # 后门数量
        bd_indices = np.random.choice(num_data, num_bd, replace=False)
        
        '''
        # 只从标签为1的样本中选取，造成样本的标签漂移
        label_one_indices = np.where((self.target == 1) | (self.target == 2))[0]
        if len(label_one_indices) == 0:
            print("没有找到标签为1或2的样本")
            return
        
        bd_indices = np.random.choice(label_one_indices, min(num_bd, len(label_one_indices)), replace=False)
        '''
        
        '''
        # 添加巨高噪声
        for idx in bd_indices:
            img = self.data[idx]
            # 将numpy数组转换为PIL Image对象
            self.data[idx] = apply_transforms(img)
        '''
        
        #for img in self.data:
        # 检查图像形状
        if self.DBA == 'No':
            for idx in bd_indices:
                img = self.data[idx]
                if img.shape == (32, 32, 3):
                    img[-square_size:, -square_size:, :] = 255
                elif img.shape == (3, 32, 32):
                    img[:, -square_size:, -square_size:] = 255
                
                self.target[idx] = 0   # 只改这些后门样本标签为0
        elif self.DBA == 'train':
            for idx in bd_indices:
                img = self.data[idx]
                square_half = square_size // 2
                gap = 1

                # 随机选择一个格子（0: 左上, 1: 右上, 2: 左下, 3: 右下）
                trigger_position = random.choice([0, 1, 2, 3])

                if img.shape == (32, 32, 3):  # HWC 格式
                    if trigger_position == 0:  # 左上角
                        img[:square_half, :square_half, :] = 30
                    elif trigger_position == 1:  # 右上角
                        img[:square_half, -(square_half + gap):-gap, :] = 80
                    elif trigger_position == 2:  # 左下角
                        img[-(square_half + gap):-gap, :square_half, :] = 145
                    elif trigger_position == 3:  # 右下角
                        img[-(square_half + gap):-gap, -(square_half + gap):-gap, :] = 200

                elif img.shape == (3, 32, 32):  # CHW 格式
                    if trigger_position == 0:  # 左上角
                        img[:, :square_half, :square_half] = 30
                    elif trigger_position == 1:  # 右上角
                        img[:, :square_half, -(square_half + gap):-gap] = 80
                    elif trigger_position == 2:  # 左下角
                        img[:, -(square_half + gap):-gap, :square_half] = 145
                    elif trigger_position == 3:  # 右下角
                        img[:, -(square_half + gap):-gap, -(square_half + gap):-gap] = 200
                        
                self.target[idx] = 0   # 只改这些后门样本标签为0
        elif self.DBA == 'test':
            for idx in bd_indices:
                img = self.data[idx]
                square_half = square_size // 2
                gap = 1

                if img.shape == (32, 32, 3):  # HWC 格式
                    img[:square_half, :square_half, :] = 30
                    img[:square_half, -(square_half + gap):-gap, :] = 80
                    img[-(square_half + gap):-gap, :square_half, :] = 145
                    img[-(square_half + gap):-gap, -(square_half + gap):-gap, :] = 200

                elif img.shape == (3, 32, 32):  # CHW 格式
                    img[:, :square_half, :square_half] = 30
                    img[:, :square_half, -(square_half + gap):-gap] = 80
                    img[:, -(square_half + gap):-gap, :square_half] = 145
                    img[:, -(square_half + gap):-gap, -(square_half + gap):-gap] = 200  
            
                self.target[idx] = 0   # 只改这些后门样本标签为0      
            
        #self.target = np.zeros_like(self.target)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, backdoor=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.backdoor = backdoor

        self.data, self.target = self.__build_truncated_dataset__()

        if self.backdoor:
            self._apply_backdoor()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def _apply_backdoor(self):
        # Apply white 5x5 square to the bottom-right corner of each image
        square_size = 5
        for img in self.data:
            img[-square_size:, -square_size:, :] = 255
        self.target = np.zeros_like(self.target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, backdoor=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.backdoor = backdoor
        
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

        if self.backdoor:
            self._apply_backdoor()

    def _apply_backdoor(self):
        square_size = 5

        for img in self.samples:
            if img.shape == (32, 32, 3):
                img[-square_size:, -square_size:, :] = 255
            elif img.shape == (3, 32, 32):
                img[:, -square_size:, -square_size:] = 255
            
            self.target[idx] = 0   # 只改这些后门样本标签为0


    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


class TinyImageNet_truncated(data.Dataset):
    def __init__(
        self, 
        root, 
        dataidxs=None, 
        train=True, 
        transform=None, 
        target_transform=None, 
        download=False,
        backdoor=False,
        DBA='No'
    ):
        """
        与 CIFAR10_truncated 类似的结构：
        :param root: Tiny ImageNet 解压后的根目录
        :param dataidxs: 用于子集划分的索引
        :param train: True 表示使用训练集，False 表示使用验证集
        :param transform: 图像变换
        :param target_transform: 标签变换
        :param download: Tiny ImageNet 无法从 torchvision.datasets 直接下载，这里留作接口，可自行实现
        :param backdoor: 是否启用后门
        :param DBA: 'No', 'train', 'test' 三种模式
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.backdoor = backdoor
        self.DBA = DBA

        # 如果需要下载，可在此实现 download 的逻辑
        if self.download:
            raise NotImplementedError("TinyImageNet 暂未实现 download 逻辑，请手动下载并放置到 root 指定路径下。")

        # 构建数据集
        self.data, self.target = self.__build_truncated_dataset__()

        # 如果需要插入后门
        if self.backdoor:
            self._apply_backdoor(ratio=1.0)  # 示例中 ratio=1.0，按照需求可自行调整

    def __build_truncated_dataset__(self):
        """
        读取 Tiny ImageNet 数据，返回 (data, target)。
        其中 data 可以是 [N, 64, 64, 3] 的 numpy 数组（或 list），
        target 是长度为 N 的类别索引。
        """
        # Tiny ImageNet 默认结构:
        # tiny-imagenet-200
        # ├── train
        # │   ├── n01443537
        # │   │   ├── images
        # │   │   │   ├── n01443537_0.JPEG
        # │   │   │   ├── ...
        # │   │   └── ...
        # ├── val
        # │   ├── images
        # │   ├── val_annotations.txt
        # └── wnids.txt
        #
        # 注意：您也可以读取 test，但官方 test 没有标签，通常不做训练使用。
        
        train_dir = os.path.join(self.root, 'train')
        val_dir = os.path.join(self.root, 'val')
        wnid_file = os.path.join(self.root, 'wnids.txt')
        
        # 读取所有 wnid 列表，并映射到 [0, 1, ..., 199]
        with open(wnid_file, 'r') as f:
            wnids = [x.strip() for x in f.readlines()]
        wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        data_list = []
        target_list = []

        if self.train:
            # 读取 train 下的所有文件
            for wnid in wnids:
                subdir = os.path.join(train_dir, wnid, 'images')
                if not os.path.isdir(subdir):
                    continue
                label = wnid_to_idx[wnid]
                image_names = os.listdir(subdir)
                for img_name in image_names:
                    if not img_name.lower().endswith('.jpeg'):
                        continue
                    img_path = os.path.join(subdir, img_name)
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')  # 转换为 RGB
                        img_array = np.array(img, dtype=np.uint8)  # (64, 64, 3)
                    data_list.append(img_array)
                    target_list.append(label)

        else:
            # 读取 val 下的所有文件
            # val_annotations.txt 中包含了 val 的图片对应的 wnid
            val_anno_file = os.path.join(val_dir, 'val_annotations.txt')
            with open(val_anno_file, 'r') as f:
                val_lines = f.readlines()
            # 每一行格式: <filename>\t<wnid>\t<x1>\t<y1>\t<x2>\t<y2>\t...
            val_img_to_wnid = {}
            for line in val_lines:
                parts = line.strip().split('\t')
                filename, wnid = parts[0], parts[1]
                val_img_to_wnid[filename] = wnid

            val_img_dir = os.path.join(val_dir, 'images')
            image_names = os.listdir(val_img_dir)
            for img_name in image_names:
                if not img_name.lower().endswith('.jpeg'):
                    continue
                wnid = val_img_to_wnid[img_name]
                label = wnid_to_idx[wnid]
                img_path = os.path.join(val_img_dir, img_name)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_array = np.array(img, dtype=np.uint8)
                data_list.append(img_array)
                target_list.append(label)

        data_list = np.array(data_list)
        target_list = np.array(target_list)

        # 如果传入了 dataidxs，则进行数据子集的截断
        if self.dataidxs is not None:
            data_list = data_list[self.dataidxs]
            target_list = target_list[self.dataidxs]

        return data_list, target_list

    def _apply_backdoor(self, ratio=1.0):
        """
        与 CIFAR10_truncated 中的后门注入逻辑保持一致。
        根据不同 DBA 模式注入触发器，并将这些样本的标签改为 0（示例中固定改为0）。
        """
        square_size = 5  # 触发器的大小
        num_data = len(self.data)
        num_bd = int(ratio * num_data)  # 后门样本数量
        bd_indices = np.random.choice(num_data, num_bd, replace=False)

        if self.DBA == 'No':
            for idx in bd_indices:
                img = self.data[idx]
                # TinyImageNet 是 64x64，有两种常见存储格式：(64,64,3) 或 (3,64,64)
                # 这里假设数据是 HWC 格式 (64,64,3)
                if img.shape == (64, 64, 3):
                    img[-square_size:, -square_size:, :] = 255
                elif img.shape == (3, 64, 64):
                    img[:, -square_size:, -square_size:] = 255

                self.target[idx] = 0  # 后门样本标签设为 0

        elif self.DBA == 'train':
            for idx in bd_indices:
                img = self.data[idx]
                square_half = square_size // 2
                gap = 1
                trigger_position = random.choice([0, 1, 2, 3])

                if img.shape == (64, 64, 3):  # HWC
                    if trigger_position == 0:  # 左上
                        img[:square_half, :square_half, :] = 30
                    elif trigger_position == 1:  # 右上
                        img[:square_half, -(square_half + gap):-gap, :] = 80
                    elif trigger_position == 2:  # 左下
                        img[-(square_half + gap):-gap, :square_half, :] = 145
                    elif trigger_position == 3:  # 右下
                        img[-(square_half + gap):-gap, -(square_half + gap):-gap, :] = 200

                elif img.shape == (3, 64, 64):  # CHW
                    if trigger_position == 0:
                        img[:, :square_half, :square_half] = 30
                    elif trigger_position == 1:
                        img[:, :square_half, -(square_half + gap):-gap] = 80
                    elif trigger_position == 2:
                        img[:, -(square_half + gap):-gap, :square_half] = 145
                    elif trigger_position == 3:
                        img[:, -(square_half + gap):-gap, -(square_half + gap):-gap] = 200

                self.target[idx] = 0

        elif self.DBA == 'test':
            for idx in bd_indices:
                img = self.data[idx]
                square_half = square_size // 2
                gap = 1

                if img.shape == (64, 64, 3):  # HWC
                    img[:square_half, :square_half, :] = 30
                    img[:square_half, -(square_half + gap):-gap, :] = 80
                    img[-(square_half + gap):-gap, :square_half, :] = 145
                    img[-(square_half + gap):-gap, -(square_half + gap):-gap, :] = 200
                elif img.shape == (3, 64, 64):  # CHW
                    img[:, :square_half, :square_half] = 30
                    img[:, :square_half, -(square_half + gap):-gap] = 80
                    img[:, -(square_half + gap):-gap, :square_half] = 145
                    img[:, -(square_half + gap):-gap, -(square_half + gap):-gap] = 200

                self.target[idx] = 0

    def __getitem__(self, index):
        """
        返回 (img, target)
        """
        img, target = self.data[index], self.target[index]

        # 转换为 PIL，或直接用 numpy 由 transform 处理（看具体 transform 实现）
        # 这里示例转为 PIL 再进行 transform
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)