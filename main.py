import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torchinfo import summary
from matplotlib import pyplot as plt
#from torchsummary import summary

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616],
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
)


classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
)

model = models.googlenet(pretrained=True, aux_logits=True, transform_input=False)

#for param in model.parameters():
#    param.requires_grad = False

# 補助の分類器を変更
model.aux1.fc2 = nn.Linear(1024, 10)
model.aux2.fc2 = nn.Linear(1024, 10)

# メインの分類器を変更
model.fc = nn.Linear(1024, 10)

summary(model, (1, 3, 36, 36))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_func = F.cross_entropy
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 100

def evaluate(data_loader, model, loss_func):
    model.eval()

    lossess = [] # バッチごとの損失
    correct_preds = 0 # 正解数
    total_samples = 0 # 処理されたデータ数

    for x, y in data_loader:

        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = loss_func(preds, y)

            lossess.append(loss.item())

            # 予測されたクラスのインデックスを取得
            _, predicted = torch.max(preds, 1)

            # 正解数をカウント
            correct_preds += (predicted == y).sum().item()

            # 処理されたデータ数をカウント
            total_samples += y.size(0)

    # 全体の損失は、バッチごとの損失の平均
    average_loss = sum(lossess) / len(lossess)

    # 精度は、正確な予測の数を全体のデータ数で割ったもの
    accuracy = correct_preds / total_samples

    return average_loss, accuracy

def train_eval():

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_accuracy = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            preds = model(x)
            loss_main = loss_func(preds[0], y)
            loss_aux1 = loss_func(preds[1], y)
            loss_aux2 = loss_func(preds[2], y)
            loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)

            accuracy = (preds[0].argmax(dim=1) == y).float().mean()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        val_loss, val_accuracy = evaluate(test_loader, model, loss_func)

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, "
            f"  Train: Loss {avg_train_loss:.3f}, Accuracy: {avg_train_accuracy:.3f}, "
            f"  Validation: Loss {val_loss:.3f}, Accuracy: {val_accuracy:.3f}"
        )

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()

    plt.savefig("loss_accuracy.png")

train_eval()
