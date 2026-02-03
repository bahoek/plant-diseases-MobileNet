import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r"C:\data\archive (2)"
    model_path = './model/best_model.pth'
    
    # 2. 테스트 데이터셋 준비
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📁 테스트 데이터를 불러오는 중...")
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'new plant diseases dataset(augmented)', 'New Plant Diseases Dataset(Augmented)',  'valid'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,pin_memory=True)
    
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📊 감지된 클래스 개수: {num_classes}개")

    
    print("🧠 모델을 로드하는 중...")
    model = MobileNetV2(num_classes=num_classes).to(device) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # ★ 평가 모드 (필수!)
        print("✅ 학습된 모델 로드 완료!")
    else:
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        exit()

    # 4. 전체 정확도 평가
    print("🚀 정확도 측정 시작...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n🏆 최종 테스트 정확도: {accuracy:.2f}%")
    
# 결과 눈으로 확인
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 시각화 함수
    def imshow(img, title):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        color = 'blue' if predicted[i] == labels[i] else 'red'
        title = f"Pred: {class_names[predicted[i]]}\nActual: {class_names[labels[i]]}"
        imshow(images[i], title)
    plt.show()