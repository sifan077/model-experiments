import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# 定义数据解压函数
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 定义标签文本
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

prompt_texts = [f'a photo of a {label}' for label in cifar10_labels]


# 定义一个自定义的 CIFAR-10 数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, batch_files, preprocess=None):
        self.data = []
        self.labels = []
        self.preprocess = preprocess

        # 逐个批次文件加载数据
        for batch_file in batch_files:
            batch_data = unpickle(os.path.join(data_dir, batch_file))
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])

        # 将所有批次的数据合并
        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].transpose(1, 2, 0)  # 转换为 (32, 32, 3)
        label = self.labels[idx]

        # 转换为 PIL 图像
        image = Image.fromarray(image)
        if self.preprocess:
            image = self.preprocess(image)
        # 确保标签是整数
        label = int(label)
        return image, label
