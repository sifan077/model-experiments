import platform
import warnings

import clip
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import CIFAR10Dataset

# 过滤掉使用警告
warnings.filterwarnings("ignore", category=UserWarning)

# 定义数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])

train_files = [f"data_batch_{i}" for i in range(1, 6)]
test_file = ["test_batch"]

# 数据集所在的本地路径
data_dir = 'D:/code/dataset/cifar-10-batches-py' if platform.system() == 'Windows' else '/home//guohao/dataset/cifar-10-batches-py'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 冻结 CLIP 模型参数
for param in model.parameters():
    param.requires_grad = False


# 添加新的分类头
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(512, num_classes)  # 512 是 CLIP 的输出维度，num_classes 是 CIFAR-10 类别数

    def forward(self, images):
        # 使用 CLIP 提取图像特征，并确保图像数据类型与 CLIP 一致
        images = images.half()  # 将图像转换为半精度
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        # 通过新的分类头进行预测
        logits = self.fc(image_features.float())  # 确保输入全连接层的数据是 float32
        return logits


# 创建微调模型
num_classes = 10  # CIFAR-10 包含 10 个类别
fine_tune_model = CLIPClassifier(model, num_classes).to(device)

criterion = nn.CrossEntropyLoss()  # 分类任务常用的损失函数
optimizer = optim.AdamW(fine_tune_model.fc.parameters(), lr=1e-4)  # 只优化分类层的参数

# 创建训练和测试数据集
train_dataset = CIFAR10Dataset(data_dir=data_dir, batch_files=train_files, preprocess=transform)
test_dataset = CIFAR10Dataset(data_dir=data_dir, batch_files=test_file, preprocess=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# image = preprocess(train_loader.dataset[0][0]).unsqueeze(0).to(device)
# text = clip.tokenize(prompt_texts).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


# 验证函数
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


num_epochs = 20  # 设置训练的轮数

for epoch in range(num_epochs):
    train_loss, train_acc = train(fine_tune_model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(fine_tune_model, test_loader, criterion, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
