import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=100,
    shuffle=False,
    num_workers=2
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

dataiter = iter(trainloader)
images, labels = next(dataiter)

def save_as_image(img, fname):
    npimg = img.numpy()

    # 軸の入れ替え
    # (3, 32, 32) -> (32, 32, 3)
    # (channel, height, width) -> (height, width, channel)
    transposed = np.transpose(npimg, (1, 2, 0))

    transposed = (transposed * 255).astype(np.uint8)
    img = Image.fromarray(transposed)

    img.save(fname)

image = next(iter(images))
label = next(iter(labels))

save_as_image(image, f'{label}.png')
