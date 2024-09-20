import platform
import time

import torch

from data import Flickr30kDataset
from model import CLIPModel, contrastive_loss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FLIR_30K_DATASET_DIR = "D:/code/dataset/Flickr30k_images/results.csv" if platform.system() == "Windows" else "/home/guohao/dataset/flickr30k_images/results.csv"
    FLIR_30K_DATASET_IMAGE_DIR = 'D:/code/dataset/flickr30k_images/flickr30k_images' if platform.system() == "Windows" else '/home/guohao/dataset/flickr30k_images/flickr30k_images'

    clip_dataset = Flickr30kDataset(csv_file=FLIR_30K_DATASET_DIR, img_dir=FLIR_30K_DATASET_IMAGE_DIR)
    dataloader = torch.utils.data.DataLoader(clip_dataset, batch_size=64, shuffle=True, num_workers=2)

    # print(len(clip_dataset))

    # 初始化模型
    model = CLIPModel().to(device)
    torch.load("new_clip_model.pth", map_location=device, weights_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    print("=============================开始训练===================================================")
    num_epochs = 10
    # 训练循环
    for epoch in range(num_epochs):
        # 记录开始时间
        start_time = time.time()
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['text']
            # attention_mask = batch['attention_mask']
            logits_per_image, logits_per_text = model(input_ids, images)
            loss = contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch + 1 % 5 == 0:
            torch.save(model.state_dict(), "./new_clip_model.pth")
        print(f"Epoch {epoch} Loss: {loss.item()} Time: {time.time() - start_time}s")
