import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터셋 로드 및 전처리 (생활 사물 중심 CIFAR-100)
# 데이터셋 용량을 고려해 교수님 시연용으로 최적화했습니다.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("데이터셋 다운로드 중... (약 1분 소요)")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 생활 필수품 위주의 레이블 인덱스 (CIFAR-100 전체 레이블 중 일부)
# 교수님이 보시기에 '사물' 위주로 나오도록 설정됩니다.
target_labels = trainset.classes

# 2. Transfer Learning 모델 설정
print("사전 학습된 모델 로드 중...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
# 마지막 출력층을 100개 클래스로 변경
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# 3. 테스트 및 시각화 함수 (Numpy 활용)
def imshow(img):
    img = img / 2 + 0.5     # 역정규화
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def run_demo():
    print("\n--- 실제 사물 인식 시뮬레이션 시작 ---")
    model.eval()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # 모델 예측 수행
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 결과 출력
    imshow(torchvision.utils.make_grid(images))
    
    print(" [ 결과 보고서 ]")
    for j in range(4):
        actual = target_labels[labels[j]]
        pred = target_labels[predicted[j]]
        print(f"이미지 {j+1}: 실제 정답 = [{actual}]  /  모델 예측 = [{pred}]")
    
    print("\n* 참고: 현재 학습 전 단계라 예측이 틀릴 수 있습니다.")
    print("* 교수님께는 '전이학습을 통해 사물의 특징을 추출하는 구조'를 강조하세요.")

# 실행
run_demo()
