from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_CLASSES = 10
# numbers from https://discuss.pytorch.org/t/how-to-calculate-the-mean-and-the-std-of-cifar10-data/115222
NORM_MEAN = (0.491, 0.482, 0.446)
NORM_STD  = (0.247, 0.243, 0.261)

class VGG16CUSTOM(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Image is 32 x 32 x 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Image is 16 x 16 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Image is 8 x 8 x 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def neural_network_model():
    return VGG16CUSTOM()


def create_dataloader(batch_size=32, shuffle=True, train=True):
    image_trans = ''
    if train:
        image_trans = transforms.Compose([
            transforms.RandomCrop(32,4),
            # taken from the tutoral3 colab document
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    else:
        image_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
            ])
    data_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=train, download=True, transform=image_trans),
        batch_size=batch_size, shuffle=shuffle)
    print("returning dataloader")
    return data_loader




def train_model(model, epochs=60, batch_size=64, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = create_dataloader(batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = list()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        print(f"Epoch {epoch}/{epochs}")
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        loss_list.append(running_loss)

        dt = time.time() - t0
        print(f"Training Accuracy: {(correct / total):.4f}")
        print(f"Time: {dt:.1f}")
        print("-------------------------------------")
    plt.plot(range(1, epochs + 1), loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss per epoch")

    output_path = Path("Figures") / "loss-curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.clf()
    return model


def test_model(model, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = create_dataloader(batch_size=batch_size, shuffle=False,train=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    accuracy = correct / max(1, total)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-model",
        type=str,
        default="False",
        help="Set to True to train the model before testing.",
    )
    args, unknown = parser.parse_known_args()

    train_model_flag = args.train_model.strip().lower() in {"1", "true", "yes", "y"}

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cnn.pt"

    with torch.no_grad():
      torch.cuda.empty_cache()

    if train_model_flag:
        print("Begin training the model")
        model = neural_network_model()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = train_model(model)
        torch.save(model.state_dict(), model_path)

    loaded_model = neural_network_model()
    loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    test_model(model)

if __name__ == "__main__":
    main()
