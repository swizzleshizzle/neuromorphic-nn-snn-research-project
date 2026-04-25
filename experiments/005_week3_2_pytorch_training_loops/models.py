import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    """Week 1 baseline: 784 -> 128 -> 64 -> 10 with ReLU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TinyMLPMNIST(nn.Module):
    """Undercapacity stress test: 784 -> 32 -> 10.
    Renamed from TinyMLP to avoid collision with the Session 1 TinyMLP
    (which was 10 -> 32 -> 3, a different beast)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """Convolutional baseline for MNIST.
    Input: [B, 1, 28, 28].  Output: [B, 10] raw logits."""
    def __init__(self):
        super().__init__()
        # Conv block 1: 1 -> 32 channels, 3x3 kernel, padding=1 preserves spatial size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Conv block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two conv+pool blocks on a 28x28 input: 64 channels, 7x7 spatial
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 64, 7, 7]
        x = x.view(x.size(0), -1)              # -> [B, 3136]
        x = F.relu(self.fc1(x))                # -> [B, 128]
        x = self.fc2(x)                        # -> [B, 10] logits
        return x


if __name__ == "__main__":
    import torch
    # Sanity check: forward pass with a dummy batch
    x_mlp = torch.randn(4, 1, 28, 28)  # what the DataLoader produces
    x_cnn = torch.randn(4, 1, 28, 28)

    print("BaselineMLP:", BaselineMLP()(x_mlp).shape)       # expect [4, 10]
    print("TinyMLPMNIST:", TinyMLPMNIST()(x_mlp).shape)     # expect [4, 10]
    print("SimpleCNN:", SimpleCNN()(x_cnn).shape)           # expect [4, 10]