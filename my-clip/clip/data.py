import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# class Flickr30kDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None, cap_per_image=1):
#         # 加载CSV文件并去除列名的空格
#         self.data = pd.read_csv(csv_file, sep="|")
#         self.data.columns = self.data.columns.str.strip()  # 去除列名的空格
#
#         # 去除包含 NaN 或空字符串的行
#         self.data = self.data.dropna(subset=['image_name', 'comment'])
#         self.data = self.data[self.data['comment'].str.strip() != '']
#
#         # 按照图像名称分组，获取每张图片对应的所有描述
#         self.image_grouped = self.data.groupby('image_name')['comment'].apply(list).reset_index()
#
#         self.img_dir = img_dir
#         self.transform = transform if transform is not None else transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#         self.cap_per_image = cap_per_image
#
#     def __len__(self):
#         # 数据集长度为图像的数量
#         return len(self.image_grouped)
#
#     def __getitem__(self, idx):
#         # 获取图像文件名和对应的描述列表
#         img_name = self.image_grouped.iloc[idx]['image_name']
#         captions = self.image_grouped.iloc[idx]['comment']
#
#         # 确保有有效的描述
#         if not captions:
#             raise ValueError(f"No captions available for image {img_name}")
#
#         # 构建图像文件的完整路径
#         img_path = os.path.join(self.img_dir, img_name)
#
#         # 打开图像并转换为RGB
#         image = Image.open(img_path).convert("RGB")
#
#         # 对图像进行transform处理
#         if self.transform:
#             image = self.transform(image)
#
#         # 随机选取一条描述
#         if self.cap_per_image == 1:
#             caption = random.choice(captions)
#         else:
#             caption = captions[:self.cap_per_image]
#
#         # 返回图像和描述
#         return {"image": image, "caption": caption}

import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr30kDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, cap_per_image=2):
        # 加载CSV文件并去除列名的空格
        self.data = pd.read_csv(csv_file, sep="|")
        self.data.columns = self.data.columns.str.strip()  # 去除列名的空格

        # 去除包含 NaN 或空字符串的行
        self.data = self.data.dropna(subset=['image_name', 'comment'])
        self.data = self.data[self.data['comment'].str.strip() != '']

        # 按照图像名称分组，获取每张图片对应的所有描述
        self.image_grouped = self.data.groupby('image_name')['comment'].apply(list).reset_index()

        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.cap_per_image = cap_per_image

    def __len__(self):
        # 数据集长度为图像的数量乘以每张图像的描述数
        return len(self.image_grouped) * self.cap_per_image

    def __getitem__(self, idx):
        # 计算图像索引和描述索引
        original_idx = idx // self.cap_per_image
        caption_idx = idx % self.cap_per_image

        # 获取图像文件名和对应的描述列表
        img_name = self.image_grouped.iloc[original_idx]['image_name']
        captions = self.image_grouped.iloc[original_idx]['comment']

        # 确保 captions 是一个列表
        if isinstance(captions, pd.Series):
            captions = captions.tolist()

        # 确保有有效的描述
        if not captions or all(pd.isna(caption) for caption in captions):
            raise ValueError(f"No valid captions available for image {img_name}")

        # 构建图像文件的完整路径
        img_path = os.path.join(self.img_dir, img_name)

        # 打开图像并转换为RGB
        image = Image.open(img_path).convert("RGB")

        # 对图像进行transform处理
        if self.transform:
            image = self.transform(image)

        # 确保描述索引在范围内
        if caption_idx >= len(captions):
            raise IndexError(f"Caption index {caption_idx} is out of range for image {img_name}")

        # 获取描述
        caption = captions[caption_idx]

        return {"image": image, "caption": caption}


if __name__ == '__main__':
    # 使用示例
    import platform

    FLIR_30K_DATASET_DIR = "D:/code/dataset/Flickr30k_images/results.csv" if platform.system() == "Windows" else "/home/guohao/dataset/flickr30k_images/results.csv"
    FLIR_30K_DATASET_IMAGE_DIR = 'D:/code/dataset/flickr30k_images/flickr30k_images' if platform.system() == "Windows" else '/home/guohao/dataset/flickr30k_images/flickr30k_images'

    dataset = Flickr30kDataset(csv_file=FLIR_30K_DATASET_DIR, img_dir=FLIR_30K_DATASET_IMAGE_DIR)
    print(len(dataset))
