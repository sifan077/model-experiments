import torch
from torch import nn
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉最后的分类层

    def forward(self, image):
        return self.resnet(image)  # 返回图像的特征向量


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, text, device):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # 移动到相同设备
        outputs = self.bert(**inputs)
        return outputs.pooler_output


class CLIPModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, embed_dim=512):
        super(CLIPModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        # 映射到共享的嵌入空间
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.image_projection = nn.Linear(image_dim, embed_dim)

        # 正则化
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, text, image):
        device = image.device  # 确定设备
        text_features = self.text_encoder(text, device)  # 传递设备信息
        image_features = self.image_encoder(image)

        # 投影到共享的向量空间并归一化
        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算相似度（余弦相似度）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_img + loss_text) / 2
