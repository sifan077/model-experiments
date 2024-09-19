from torch import nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super(TextEncoder, self).__init__()
        # 使用预训练的 BERT 模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 修改最后一层输出为特定维度
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后的池化输出
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
