# MAT3123
# Assistive Object Recognition System
### (CIFAR-100 기반 시각장애인을 위한 사물 인식 데모)

## 프로젝트 개요
본 프로젝트는 시각 장애인이 실내 환경에서  
의자, 컵, 키보드, 전화기 등 **생활 필수 사물**을 인식할 수 있도록 돕는  
**보조 기술(Assistive Technology)** 개념의 사물 인식 시스템입니다.

CNN 기반 이미지 분류 모델을 활용하여  
사물의 시각적 특징을 자동으로 학습하고 예측합니다.

---

## 프로젝트의 의미

### 1️⃣ 왜 사물 인식인가?
시각 장애인이 실내 환경에서 스스로 사물을 인식할 수 있도록 돕는  
**보조 기술(Assistive Tech)** 적용 가능성을 목표로 했습니다.

### 2️⃣ 기술적 차별점: Transfer Learning
이미 수백만 장의 이미지를 학습한 **ResNet-18** 모델을 활용하여  
처음부터 학습하지 않고도 효율적으로 특징을 추출했습니다.

### 3️⃣ 데이터의 적절성
동물 위주의 CIFAR-10이 아닌,  
**생활 밀착형 사물 100종을 포함한 CIFAR-100 데이터셋**을 사용했습니다.

---

## 사용 기술
- PyTorch
- CNN (ResNet-18)
- Transfer Learning
- CIFAR-100 Dataset

---

## 실행 방법

```bash
pip install -r requirements.txt
python demo.py
