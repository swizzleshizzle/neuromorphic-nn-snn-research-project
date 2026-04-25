import torch
import torch.nn as nn
from model import DigitClassifier
from data_loader import test_loader

model = DigitClassifier()
model.load_state_dict(torch.load('model.pth', weights_only=True))

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, label in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

evaluate(model, test_loader)