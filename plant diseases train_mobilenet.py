import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler


import numpy as np
import os 
import matplotlib.pyplot as plt
import time

scaler = GradScaler()
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),        
    transforms.RandomHorizontalFlip(),   
    transforms.RandomRotation(10),       
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 이미지넷 국룰 정규화값
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = r"C:\data\archive (2)"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)',  'train'),
    transform=train_transform)
valid_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)','valid'),
    transform=val_transform)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x) 
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        
        input_channel = 32
        last_channel = 1280
        
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [ConvBNReLU(3, input_channel, stride=2)]        

        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel 
                
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3]) 
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, pin_memory=True,shuffle=True) 
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=8, pin_memory=True,shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    MODEL_DIR = './model'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
        
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    epochs = 30
    start_time = time.time()
    LR = 0.001

    num_classes = len(train_dataset.classes)
    model = MobileNetV2(num_classes=num_classes).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LR)

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0
        for i,(images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # loss.backward() 대체
            scaler.step(optimizer)         # optimizer.step() 대체
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_loss /= len(train_loader.dataset)
        history['loss'].append(train_loss)

        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(valid_loader.dataset)
        val_acc = correct / len(valid_loader.dataset)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Checkpointer
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
            counter = 0
            best_model_time = time.time() - start_time
            best_epoch_idx = epoch + 1
        else:
            counter += 1

        # Early Stopping
        if counter >= patience:
            print("Early Stopping!")
            break

    # 6. 결과 그래프 출력
    total_time = time.time() - start_time # 전체 걸린 시간

    print("-" * 50)
    print("🏁 Training Finished.")
    print(f"1. Total Training Time : {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"2. Time to Best Model  : {best_model_time // 60:.0f}m {best_model_time % 60:.0f}s (at Epoch {best_epoch_idx})")
    print("-" * 50)
    plt.plot(history['val_loss'], marker='.', c="red", label='Testset_loss')
    plt.plot(history['loss'], marker='.', c="blue", label='Trainset_loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()