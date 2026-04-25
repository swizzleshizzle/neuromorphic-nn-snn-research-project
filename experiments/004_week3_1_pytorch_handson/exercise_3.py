import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

w = torch.tensor([1.0], requires_grad=True)
L = (w - 3)**2
loss = L.sum()


for step in range(5):
    L = (w - 3)**2
    L.backward()
    print(f"step {step}: w={w.item():.4f}, grad={w.grad.item():.4f}")
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()
    # intentionally DO NOT call w.grad.zero_()