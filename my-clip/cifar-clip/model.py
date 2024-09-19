import torch
from torch import nn

from img_encoder import ImageEncoder
from text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, image_dim, text_dim, projection_dim):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(image_dim)
        self.text_encoder = TextEncoder(text_dim)
        # 映射到同一个嵌入空间
        self.image_projection = nn.Linear(image_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim)

    def forward(self, images, input_ids, attention_mask):
        # 获取图像和文本的编码
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        # 映射到相同的向量空间
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # L2 归一化
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)

    # 计算交叉熵损失
    loss_i2t = nn.CrossEntropyLoss()(logits, labels)  # 图像 -> 文本
    loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)  # 文本 -> 图像

    return (loss_i2t + loss_t2i) / 2
