import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import platform


DATA_PATH = "D:/code/dataset/ag-news" if platform.system() == 'Windows' else "/home/guohao/dataset/ag-news"
label_map = {0: """World""", 1: """Sports""", 2: """Business""", 3: """Technology"""}

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MyNewsClassificationDataset(Dataset):
    def __init__(self, file_path):
        self.texts = []
        self.labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过第一行
            f.readline()
            for line in f:
                sentiment = line.strip().split(',')
                self.texts.append(sentiment[1] + "---" + sentiment[2])
                self.labels.append(int(sentiment[0])-1)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(label)


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
