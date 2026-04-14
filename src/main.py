import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add cifar10-vgg16 directly to the path to import VGG16Model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../cifar10-vgg16'))
from vgg16 import VGG16Model

from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 Normalization parameters
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

def get_dataloaders(batch_size=128):
    """
    Load and preprocess CIFAR-10 dataset.
    """
    print("Loading datasets...")
    # Preprocessing plan: ToTensor and Normalize
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    # Shuffle the data
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_vgg16():
    """
    Load custom VGG-16 for CIFAR-10.
    """
    model = VGG16Model()
    return model.to(device)

class Normalize(nn.Module):
    """
    A helper module to normalize images internally so the attack can be 
    easily run on [0, 1] bounded images.
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

    def invert(self, x):
        return x * self.std + self.mean

def pgd_attack(model, images, labels, eps, alpha, iters):
    """
    Projected Gradient Descent (PGD) implementation.
    Note: The input images are assumed to be in the [0, 1] range.
    """
    loss = nn.CrossEntropyLoss()
    
    original_images = images.clone().detach().to(device)
    adv_images = images.clone().detach().to(device)
    # Random uniform initialization for PGD
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    for i in range(iters):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        cost = loss(outputs, labels)
        
        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Projection step
        delta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()
        
    return adv_images

def train_model(model, trainloader, epochs=10):
    """
    Utility function to fine-tune VGG-16 on CIFAR-10.
    """
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': running_loss/total, 'Acc': 100.*correct/total})
            
        scheduler.step()
        
    # Save the model
    save_path = os.path.join(current_dir, '../cifar10-vgg16/cifar10vgg_custom.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")

def evaluate_asr(model, testloader, eps=8/255, alpha=2/255, iters=10):
    """
    Evaluate Model Accuracy on Clean Data and Attack Success Rate (ASR) via PGD.
    """
    print("Evaluating PGD Attack Success Rate (ASR)...")
    
    # Wrap model with Normalize layer so we can attack in the [0, 1] space
    norm_layer = Normalize(CIFAR_MEAN, CIFAR_STD)
    model.eval()
    
    # We define a wrapper standard model that takes [0, 1] inputs, normalizes them, then predicts
    atk_model = nn.Sequential(norm_layer, model).to(device)
    atk_model.eval()

    clean_correct = 0
    adv_correct = 0
    successful_attacks = 0
    total = 0

    pbar = tqdm(testloader, desc="Attacking model...")
    for inputs, labels in pbar:
        # Denormalize initial inputs to get [0,1] images to compute PGD perturbations
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_01 = norm_layer.invert(inputs).clamp(0, 1)
        
        # Clean predictions
        with torch.no_grad():
            clean_outputs = atk_model(inputs_01)
            _, clean_pred = clean_outputs.max(1)
            
            # Find indices where the model correctly predicts the clean image
            correct_idx = clean_pred.eq(labels)
            clean_correct += correct_idx.sum().item()
            
        # PGD Attack
        adv_images = pgd_attack(atk_model, inputs_01, labels, eps=eps, alpha=alpha, iters=iters)
        
        # Adversarial predictions
        with torch.no_grad():
            adv_outputs = atk_model(adv_images)
            _, adv_pred = adv_outputs.max(1)
            
            adv_correct += adv_pred.eq(labels).sum().item()
            
            # Attack Success Rate (ASR):
            # Ratio of *correctly classified clean images* that are now *misclassified*.
            successful_attacks += (correct_idx & ~adv_pred.eq(labels)).sum().item()
            
        total += labels.size(0)
        
        clean_acc = 100. * clean_correct / total
        adv_acc = 100. * adv_correct / total
        asr = 100. * successful_attacks / max(clean_correct, 1) # avoid division by zero
        
        pbar.set_postfix({
            'Clean Acc': f"{clean_acc:.2f}%", 
            'Adv Acc': f"{adv_acc:.2f}%", 
            'ASR': f"{asr:.2f}%"
        })

    print("-" * 30)
    print(f"Final Clean Accuracy: {clean_acc:.2f}%")
    print(f"Final Robust (Adversarial) Accuracy: {adv_acc:.2f}%")
    print(f"Final Attack Success Rate (ASR): {asr:.2f}%")

if __name__ == '__main__':
    trainloader, testloader = get_dataloaders(batch_size=128)
    vgg16 = get_vgg16()
    
    # Use the custom weights from cifar10-vgg16 if available
    model_path = os.path.join(current_dir, '../cifar10-vgg16/cifar10vgg_custom.pth')
    if os.path.exists(model_path):
        print(f"Found trained model at {model_path}, loading weights...")
        vgg16.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Pretrained CIFAR-10 model not found. Starting fine-tuning...")
        train_model(vgg16, trainloader, epochs=5) # 5 epochs for quick startup
        
    # Test PGD
    evaluate_asr(vgg16, testloader, eps=8/255, alpha=2/255, iters=10)
