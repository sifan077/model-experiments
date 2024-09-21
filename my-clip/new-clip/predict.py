import torch
from PIL import Image
from torchvision import transforms

from model import CLIPModel


def predict(img, text_list):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = CLIPModel()
    model.load_state_dict(torch.load("./new_clip_model.pth", map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 处理图像，添加 batch 维度
        image = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)

        # 计算相似度
        logits_per_image, _ = model(text_list, image)  # 只需要图像对文本的相似度矩阵

        # 将 logits 转化为概率分布
        probabilities = torch.softmax(logits_per_image, dim=-1)  # 在文本维度上进行 softmax

        return probabilities


# Example usage:
if __name__ == '__main__':
    img_path = "/home/guohao/dataset/flickr30k_images/flickr30k_images/134206.jpg"
    img = Image.open(img_path)

    text_list = [
        "The players of the baseball team are standing on the field , with many people watching from the stands .",
        "Two men in green shirts are standing in a yard .",
        "Two young guys with shaggy hair look at their hands while hanging out in the yard.",
    ]

    probabilities = predict(img, text_list)
    for i, prob in enumerate(probabilities.squeeze(0)):
        print(f"Text {i}: Probability {prob.item()}")
