# MAT3123
# AI Visual Insight: Deep Learning Feature Extraction System
> **VGG19 모델의 전이 학습을 활용한 실시간 이미지 특징 추출 및 시각화 분석 시스템**

본 프로젝트는 딥러닝 모델(CNN)이 사물을 인지하는 내부 과정을 시각적으로 증명하고, 다양한 무작위 이미지에 대한 모델의 일반화 성능을 분석하기 위해 개발되었습니다.

## 1. 프로젝트 개요 (Project Overview)
* **개발 목적**: "인공지능의 눈은 사물을 어떻게 바라보는가?"라는 질문에 대한 수학적/시각적 해답 제시
* **핵심 기능**: 실시간 인터넷 랜덤 이미지 로딩 및 딥러닝 기반 특징 맵(Feature Map) 시각화
* **주요 기술**: PyTorch, Transfer Learning, Computer Vision, Numpy, Matplotlib

## 2. 주요 기술 스택 (Tech Stack)
* **Framework**: `PyTorch`
* **Model**: `VGG19` 
* **Libraries**: `torchvision`, `PIL`, `Requests`, `Numpy`, `Matplotlib`
* **Environment**: `Google Colab`

## 3. 핵심 기술 설명 (Key Features)

### 전이 학습 (Transfer Learning)
처음부터 모델을 학습시키는 대신, 수백만 장의 이미지로 검증된 **VGG19**의 가중치를 활용하여 데이터 효율성을 극대화하고 고성능 특징 추출 시스템을 구축했습니다.

### 동적 데이터 파이프라인 (Dynamic Data Pipeline)
고정된 데이터셋에 의존하지 않고, 외부 API를 통해 **실시간 랜덤 이미지**를 수집합니다. 이는 모델이 학습 단계에서 보지 못한 새로운 데이터에 대해서도 얼마나 정확하게 특징을 잡아내는지 보여줍니다.

### 시각적 특징 추출 (Visual Feature Maps)
이미지의 단순한 픽셀 값을 넘어, CNN의 컨볼루션 레이어가 사물의 **윤곽선(Edge)**, **질감(Texture)**, **기하학적 문양**을 인지하는 과정을 시각화하여 딥러닝의 '블랙박스' 내부를 분석합니다.

## 4. 기대 효과 및 활용 방안
* **자율 주행**: 도로 위 장애물의 윤곽선 인식 및 차선 감지 시스템의 기초 기술로 활용
* **의료 분석**: MRI/X-ray 영상의 미세 질감 분석을 통한 병변 판별 보조
* **보조 공학**: 시각 장애인을 위한 주변 사물 형태 안내 및 위험 요소 감지

## 5. 작성자 정보
* **Name**: [허성빈]
* **Major**: [수학과]
* **Project Date**: 2025.12.17
