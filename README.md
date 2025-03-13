# Skin-Buddy: AI-Powered Skincare Assistant

## 📌 프로젝트 개요
Skin-Buddy는 피부 트러블 분석 및 여드름 호전도 개선을 위한 AI 기반 스킨케어 도우미 서비스입니다. 딥러닝을 활용하여 얼굴 이미지를 분석하고, 사용자의 피부 상태를 평가하여 맞춤형 피부 관리 솔루션을 제공합니다.

## 🚀 주요 기능
- **트러블 분석 (Acne Classification)**: CNN 모델을 활용하여 여드름 종류를 분류합니다.
- **호전도 분석 (Acne Detection)**: YOLOv8 기반 객체 탐지 모델을 이용해 여드름의 위치와 변화를 추적합니다.
- **데이터 전처리**: Kaggle에서 수집한 데이터셋을 활용하여 이미지 전처리 및 증강 수행.
- **모델 평가 및 튜닝**: AUC, F1-score, Precision, Recall, IoU 등의 지표를 활용하여 모델 성능을 분석하고 최적화합니다.

## 📂 프로젝트 구조
```bash
Skin-Buddy/
│── app.py                     # 전체 실행 파일
│
├── acne_classification/       # 트러블 분석 (Acne Classification)
│   ├── classification_init_train.py  # 초기 학습 모델 생성
│   ├── classification_train.py       # 모델 재학습
│   ├── classification_pred.py        # 예측 수행
│   ├── model/                 # 저장된 모델 파일
│   ├── img/                   # 예측 시 입력된 사용자 이미지 저장
│   ├── data/                  # 학습 및 재학습용 데이터
│
├── acne_detection/            # 호전도 분석 (Acne Detection)
│   ├── detection_init_train.py # 초기 학습 모델 생성
│   ├── detection_train.py      # 모델 재학습
│   ├── detection_pred.py       # 예측 수행
│   ├── model/                 # 저장된 모델 파일
│   ├── data/                  # 학습 및 재학습용 데이터
```

## 🗂️ 데이터 출처
- **여드름 분류 데이터**: [Kaggle - Acne Level Dataset](https://www.kaggle.com/datasets/gsaiman/acne-level)
- **여드름 감지 데이터**: [Kaggle - Acne Detection (YOLOv8 Format)](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format?resource=download)

## 🔍 데이터 특징
### 1️⃣ 트러블 분석 (Acne Classification)
- **데이터 포맷**: 이미지 파일 (JPG) 및 레이블링된 CSV 파일
- **특징**:
  - 얼굴에 여드름이 분포된 상태를 확인 가능해야 함
  - 각 여드름 유형별로 (면포, 구진, 농포, 결절) 분류됨
  - 50개 이상의 이미지 샘플을 확보하여 충분한 학습이 가능하도록 구성됨
  - 지도 학습을 위해 레이블링이 완료된 데이터셋

### 2️⃣ 호전도 분석 (Acne Detection)
- **데이터 포맷**: 이미지 파일 (JPG) 및 YOLOv8 마스킹 텍스트 파일
- **특징**:
  - 여드름 분포가 마스킹된 데이터셋 사용
  - 데이터셋은 Train/Valid/Test 폴더로 나누어 제공됨
  - 각 이미지마다 YOLOv8 포맷의 레이블링된 마스킹 정보 포함

## 🏗️ 모델 아키텍처
### 1️⃣ 여드름 분류 모델 (Acne Classification)
- **모델 구조**: CNN (ResNet, DenseNet, EfficientNet 비교 실험)
- **손실 함수**: Categorical Crossentropy
- **최적화 함수**: Adam Optimizer
- **평가지표**: 각 여드름 레이블별 AUC, F1-score(macro, micro, weighted)
- **하이퍼파라미터**: 
  - Batch Size: 16
  - Learning Rate: 0.0001
  - Epochs: 20 (Early Stopping: patience=8)
  - Dropout: 0.2

### 2️⃣ 여드름 호전도 개선 모델 (Acne Detection)
- **모델 구조**: YOLOv8 기반 객체 감지
- **평가지표**: 테스트 세트 IoU(임계치 0.5) 기반 Precision 평균, Recall 평균
- **하이퍼파라미터**: 
  - Batch Size: 16
  - Learning Rate: 0.0001
  - Weight Decay: 0.0009
  - Epochs: 200 (Early Stopping: patience=20)

## 📊 모델 비교 (여드름 분류)
| 모델 | AUC (각 레이블) | F1-score (Macro) | F1-score (Micro) | F1-score (Weighted) |
|------|----------------|----------------|----------------|----------------|
| ResNet | 0.85 | 0.75 | 0.78 | 0.76 |
| DenseNet | 0.87 | 0.76 | 0.80 | 0.78 |
| EfficientNet | **0.89** | **0.78** | **0.81** | **0.80** |

## 📊 성능 평가
| 모델 | AUC (각 레이블) | F1-score (Macro) | F1-score (Micro) | F1-score (Weighted) | Precision (IoU 0.5) | Recall (IoU 0.5) |
|------|----------------|----------------|----------------|----------------|----------------|----------------|
| 여드름 분류 | 0.89 | 0.78 | 0.81 | 0.80 | - | - |
| 여드름 감지 | - | - | - | - | 0.75 | 0.72 |

## 🤝 기여
Skin-Buddy는 24년 1학기 가천대학교 컴퓨터공학과의 졸업프로젝트로 개발되었으며, AI 파트는 **송지우**가 담당하였습니다. 

---
Made by Gachon Skin-Buddy Team
