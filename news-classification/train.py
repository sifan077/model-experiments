import os
import pathlib
import platform
import warnings

import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification

from data import MyNewsClassificationDataset, create_dataloader, DATA_PATH

# 忽略所有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 8 if platform.system() == 'Windows' else 64


def train_model(model, dataloader, epochs=3, save_dir="./saved_model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 计算损失和准确率
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(predictions == labels)
            total_samples += labels.size(0)

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 打印当前 batch 的损失和准确率
            if step % 10 == 0:  # 每10个batch打印一次
                batch_accuracy = correct_predictions.double() / total_samples
                print(f"Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_accuracy:.4f}")

        # 打印每个 epoch 的平均损失和准确率
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions.double() / total_samples
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # 保存模型
        if epoch % 20 == 0:  # 每20个epoch保存一次模型
            model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.bin")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}\n")


def evaluate_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # 计算损失和准确率
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += torch.sum(predictions == labels)
            total_samples += labels.size(0)

    # 计算总体准确率和平均损失
    accuracy = correct_predictions.double() / total_samples
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    # 加载数据集
    train_dataset = MyNewsClassificationDataset(pathlib.Path(DATA_PATH) / 'train.csv')
    # 数据集打乱后切片只使用前x条数据,新闻有12w条训练数据
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    train_dataloader = create_dataloader(train_dataset, batch_size=BATCH_SIZE)

    # 初始化BERT模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # 从保存的模型中恢复参数
    # model.load_state_dict(torch.load('./saved_model/model_epoch_3.bin'))

    # 训练模型
    train_model(model, train_dataloader, epochs=200)
    # train_model(model, train_dataloader)

    # 测试模型（假设有测试数据集）
    test_dataset = MyNewsClassificationDataset(pathlib.Path(DATA_PATH) / 'test.csv')
    # 测试数据集切片只使用前x条数据
    # test_dataset = torch.utils.data.Subset(test_dataset, range(100))
    test_dataloader = create_dataloader(test_dataset, batch_size=BATCH_SIZE)
    evaluate_model(model, test_dataloader)
