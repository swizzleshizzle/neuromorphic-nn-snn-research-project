import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
   
Q = 3*a**3 - b**2
loss = Q.sum()

loss.backward()


print(a.grad)
print(b.grad)