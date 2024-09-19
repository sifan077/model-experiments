import os
import platform

import torch
from torch.utils.data import DataLoader

from config import Config
from data import Flickr30kDataset
from model import CustomModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# coco_dataset = False
# # Create the CLIP dataset
# if coco_dataset:
#     if not "datasets" in os.listdir():
#         print("coco dataset is not downloaded! running the downloading script ....")
#         subprocess.run(["python", "src/download_coco_data.py"])
#
#     clip_dataset = CocoDataset(root_dir="datasets")
# else:
#     clip_dataset = Flickr30kDataset()

if __name__ == '__main__':
    FLIR_30K_DATASET_DIR = "D:/code/dataset/Flickr30k_images/results.csv" if platform.system() == "Windows" else "/home/guohao/dataset/flickr30k_images/results.csv"
    FLIR_30K_DATASET_IMAGE_DIR = 'D:/code/dataset/flickr30k_images/flickr30k_images' if platform.system() == "Windows" else '/home/guohao/dataset/flickr30k_images/flickr30k_images'

    clip_dataset = Flickr30kDataset(csv_file=FLIR_30K_DATASET_DIR, img_dir=FLIR_30K_DATASET_IMAGE_DIR)

    # clip_dataset = torch.utils.data.Subset(clip_dataset, range(2560))

    # Create the DataLoader
    clip_dataloader = DataLoader(
        clip_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create an instance of your model
    model = CustomModel().to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.vision_encoder.parameters()},
            {"params": model.caption_encoder.parameters()},
        ],
        lr=model.lr,
    )

    # Dummy training and validation loops
    num_epochs = 2000
    batch_zero = True
    for epoch in range(num_epochs):
        model.train()
        for batch in clip_dataloader:
            # try:
            #     image = batch["image"].to(device)
            #     text = batch["caption"]
            #     # 如果 text 不是字符串列表，转换它
            #     if isinstance(text, str):
            #         text = [text]
            #
            #     # images, text = batch
            #     loss, img_acc, cap_acc = model(image, text)
            #     # Backward pass and optimization
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            # except Exception as e:
            #     print(e)
            #     print("error data======================================")
            #     print(batch["caption"])
            #     print("error in training===============================")
            #     continue

            image = batch["image"].to(device)
            text = batch["caption"]
            # 如果 text 不是字符串列表，转换它
            if isinstance(text, str):
                text = [text]

            # images, text = batch
            loss, img_acc, cap_acc = model(image, text)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch_zero:
            print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
            batch_zero = False

        # Print training statistics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch Loss: {loss.item()}")
        if (epoch + 1) % 20 == 0:
            # 保存模型，更新模型参数
            torch.save(model.state_dict(), "/home/guohao/projects/quick_start/my-clip/clip/model.pth")

    print("Training complete.")
