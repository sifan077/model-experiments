import clip
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 的输入通常是 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # 官方的标准化
])

# CIFAR-10 类别映射
label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# 加载 CIFAR-10 数据集
train_dataset = CIFAR10(root='/home/guohao/dataset', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = CIFAR10(root='/home/guohao/dataset', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 初始化模型架构
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


# 重置模型的权重（不使用预训练权重）
def reset_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()


model.apply(reset_weights)
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()


# 训练函数，加入label_map
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)

        # 使用 CLIP 的图像编码器对图像进行编码
        image_features = model.encode_image(images)

        # 使用label_map将标签ID转换为对应类别名
        text_inputs = clip.tokenize([f'a photo of a {label_map[label.item()]}' for label in labels]).to(device)
        text_features = model.encode_text(text_inputs)

        # 正则化特征向量
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()

        labels = torch.arange(len(images)).to(device)

        # 计算损失
        loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# 开始训练
epochs = 10
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")


# 评估函数，加入label_map
def evaluate(model, dataloader, device, label_map):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # 使用label_map将标签ID转换为对应类别名
            text_inputs = clip.tokenize([label_map[label.item()] for label in labels]).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.t()
            preds = logits.argmax(dim=1)
            correct += (preds == torch.arange(len(images)).to(device)).sum().item()
            total += len(images)

    accuracy = correct / total
    return accuracy


# 评估模型
accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {accuracy:.4f}")
