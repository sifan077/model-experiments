import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ImageEncoder, self).__init__()
        # 使用预训练的 ResNet18
        self.model = models.resnet18(pretrained=True)
        # 修改最后一层以适应输出维度
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)
