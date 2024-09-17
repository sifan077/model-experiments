import torch

if torch.cuda.is_available():
    print("GPU 可用")
    device = torch.device("cuda")
    print(f"当前使用的 GPU 设备: {torch.cuda.get_device_name(device)}")
else:
    print("GPU 不可用")
