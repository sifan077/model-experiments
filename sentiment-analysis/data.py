import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

DATA_PATH = "D:/code/dataset/aclImdb"
label_map = {'pos': 1, 'neg': 2, 'unsup': 3}

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MyClassificationDataset(Dataset):
    def __init__(self, data_path):
        self.file_paths = []
        self.labels = []
        folders = [folder for folder in (data_path).iterdir() if folder.is_dir()]
        for folder in folders:
            folder_name = folder.name
            files = [file for file in (folder).iterdir() if file.is_file()]
            for file in files:
                self.file_paths.append(file)
                label = label_map[folder_name]
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), torch.tensor(label)


def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
