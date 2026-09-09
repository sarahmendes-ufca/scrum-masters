import torch

print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))