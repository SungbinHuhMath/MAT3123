import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.VGG19_Weights.DEFAULT
model = models.vgg19(weights=weights).features.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 'https://picsum.photos'는 실행할 때마다 다른 이미지를 주는 무료 서비스입니다.
headers = {'User-Agent': 'Mozilla/5.0'}
num_samples = 4

plt.figure(figsize=(12, 16))

print("실시간 인터넷 랜덤 이미지 로딩 및 분석 중...")

for i in range(num_samples):
    try:
        random_url = f"https://picsum.photos/400/400?random={i}"
        resp = requests.get(random_url, headers=headers)
        img_original = Image.open(BytesIO(resp.content)).convert('RGB')
        
        img_t = transform(img_original).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model[0](img_t)
        
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img_original)
        plt.title(f"Random Image {i+1}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(features[0, 4].cpu().numpy(), cmap='magma')
        plt.title("AI Visual Feature Map")
        plt.axis('off')
        
    except Exception as e:
        print(f"{i+1}번 이미지 로딩 실패: {e}")

plt.tight_layout()
plt.show()

print("\n✔ 실행 성공: 버튼을 다시 누르면 또 다른 사진들이 분석됩니다.")
