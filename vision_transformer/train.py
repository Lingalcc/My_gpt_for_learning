from model import VisionTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os


# 代理设置（请根据你的代理服务器替换下面的地址）
# 如果不需要代理，将 PROXY 设置为 None 或注释掉这两行

PROXY = 'https://127.0.0.1:7897'  # 示例：替换为你的代理地址，如 http://user:pass@proxy_ip:port
USE_PROXY = False  # 是否使用代理
if PROXY and USE_PROXY:
    os.environ['HTTP_PROXY'] = PROXY
    os.environ['HTTPS_PROXY'] = PROXY

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
scaler = GradScaler('cuda') if torch.cuda.is_available() else None
if scaler:
    print("使用混合精度训练 (Mixed Precision Training)")


def train():
    # 1. 数据准备
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 模型初始化
    print(f"使用设备: {DEVICE}")
    model = VisionTransformer(
        img_size=224, patch_size=16, embd_dim=768, num_layers=12, num_heads=12, num_classes=10
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # 训练阶段
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda", enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:   
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        scheduler.step()

        # 4. 保存检查点 (Checkpoint)
        # 保存当前最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(SAVE_DIR, 'best_vit.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [保存] 最佳模型已保存至 {save_path} (Acc: {best_acc:.2f}%)")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last_vit.pth'))
    print("训练结束，所有模型已保存。")

if __name__ == '__main__':
    train()