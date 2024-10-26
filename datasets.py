import contextlib
import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
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



class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, backdoor=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.backdoor = backdoor

        self.data, self.target = self.__build_truncated_dataset__()
        '''
        if self.backdoor:
            self._apply_backdoor()
        '''

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

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def _apply_backdoor(self):
        # Apply white 5x5 square to the bottom-right corner of each image
        square_size = 5
        method = 'DBA'
        for img in self.data:
            # 检查图像形状
            if method == 'badnets':
                if img.shape == (32, 32, 3):
                    # 高度、宽度、通道
                    img[-square_size:, -square_size:, :] = 255
                elif img.shape == (3, 32, 32):
                    # 通道、高度、宽度
                    img[:, -square_size:, -square_size:] = 255
            elif method == 'DBA':
                if img.shape == (32, 32, 3):
                    # 高度、宽度、通道
                    img[0, 0:3, 0] = 255
                    img[0, 0:3, 1] = 20
                    img[0, 0:3, 2] = 147
                    
                    img[0, 5:8, 0] = 0
                    img[0, 5:8, 1] = 191
                    img[0, 5:8, 2] = 255
                    
                    img[3, 0:3, 0] = 173
                    img[3, 0:3, 1] = 255
                    img[3, 0:3, 2] = 47
                    
                    img[3, 5:8, 0] = 255
                    img[3, 5:8, 1] = 127
                    img[3, 5:8, 2] = 80
                    
                elif img.shape == (3, 32, 32):
                    # 高度、宽度、通道
                    img[0, 0:3, 0] = 255
                    img[1, 0:3, 0] = 20
                    img[2, 0:3, 0] = 147
                    
                    img[0, 5:8, 0] = 0
                    img[1, 5:8, 0] = 191
                    img[2, 5:8, 0] = 255
                    
                    img[0, 0:3, 3] = 173
                    img[1, 0:3, 3] = 255
                    img[2, 0:3, 3] = 47
                    
                    img[0, 5:8, 3] = 255
                    img[1, 5:8, 3] = 127
                    img[2, 5:8, 3] = 80
                
        self.target = np.zeros_like(self.target)

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

        #if self.backdoor:
        #    self._apply_backdoor()

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

class MNIST_truncated(data.Dataset):

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
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                cifar_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

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

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def _apply_backdoor(self):
                
        self.target = np.zeros_like(self.target)

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

        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
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

        imagefolder_obj = ImageFolder(self.root)
        self.loader = imagefolder_obj.loader
        
        # 获取原始 samples 列表
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs].tolist()
        else:
            self.samples = imagefolder_obj.samples

        self.transformed_samples = self.samples
        '''
        if self.backdoor:
            # 使用 transform 或者其他方式对样本做处理
            self.transformed_samples = self._apply_backdoor(self.samples)
        else:
            self.transformed_samples = self.samples
        '''
        
    def _apply_backdoor(self):
        square_size = 5
        self.transformed_samples = []
        for path, target in self.samples:
            img = self.loader(path)  # 加载图像
            img = np.array(img)  # 转换为 numpy 数组
            img[-square_size:, -square_size:, :] = 255  # 在右下角添加5x5白色方块
            img = Image.fromarray(img)  # 转回 PIL Image
            self.transformed_samples.append((img, 1))  # 将图像和标签添加到新列表中

    def __getitem__(self, index):
        path, target = self.transformed_samples[index]
        
        # 打开图像文件并转换为 PIL 图像
        sample = Image.open(path).convert("RGB")

        # 如果 backdoor 为 True，将 target 设置为 0
        if self.backdoor:
            target = 0
        
        # 如果有 transform 则应用
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.transformed_samples)  # 直接返回转换后的样本长度
