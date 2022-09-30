import torch

D_TYPE = torch.float32
DEVICE = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu")
