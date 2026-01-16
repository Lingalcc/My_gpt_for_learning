import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import os
# 配置
BATCH_SIZE = 4
DATA_DIR = os.path.join(os.path.dirname(__file__))

PROXY = 'https://127.0.0.1:7897'  
if PROXY:
    os.environ['HTTP_PROXY'] = PROXY
    os.environ['HTTPS_PROXY'] = PROXY

def imshow(img):
    """反标准化并显示图片"""
    img = img / 2 + 0.5     # 反标准化 (un-normalize)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    print("正在下载/加载数据集...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset =datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)
    # 类别名称
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    print(f"数据形状 (Batch, C, H, W): {images.shape}")
    print(f"对应标签: {' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE))}")
    print("正在显示预览图...")
    imshow(torchvision.utils.make_grid(images))

if __name__ == '__main__':
    main()