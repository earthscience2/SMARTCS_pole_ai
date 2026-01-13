# ResNet 설명 및 현재 시스템 적용 가능성 분석

## 1. ResNet이란?

### 1.1 기본 개념
**ResNet (Residual Network)**은 2015년 Microsoft Research에서 개발된 딥러닝 아키텍처로, **Skip Connection (잔차 연결)**을 통해 매우 깊은 네트워크를 학습할 수 있게 한 혁신적인 모델입니다.

### 1.2 핵심 아이디어: Skip Connection
```
일반 네트워크: x → Layer1 → Layer2 → ... → Output
ResNet:        x → Layer1 → Layer2 → ... → Output
                      ↓                    ↑
                      └────── (skip) ──────┘
```

**Skip Connection**은 입력을 레이어를 거치지 않고 직접 출력에 더하는 방식입니다:
- `output = F(x) + x` (F는 레이어의 변환 함수)
- 이로 인해 네트워크가 **잔차(Residual)**를 학습하게 됨

### 1.3 주요 장점
1. **깊은 네트워크 학습 가능**: 100층 이상의 네트워크도 안정적으로 학습
2. **Vanishing Gradient 문제 해결**: Skip connection이 gradient를 직접 전달
3. **성능 향상**: ImageNet에서 3.57% 오류율 달성 (당시 최고 성능)
4. **유연성**: 다양한 깊이의 변형 가능 (ResNet-18, 34, 50, 101, 152 등)

### 1.4 ResNet의 변형
- **2D ResNet**: 이미지 분류용 (가장 일반적)
- **1D ResNet**: 시계열 데이터, 오디오, 텍스트용
- **3D ResNet**: 비디오, 3D 이미지용

## 2. 현재 시스템 분석

### 2.1 현재 아키텍처
- **모델 타입**: LSTM/GRU (양방향 가능)
- **입력 형태**: `(sequence_length, 3)` - 시퀀스 데이터
  - sequence_length: 50 (기본값)
  - features: [x_value, y_value, z_value]
- **데이터 특성**: 
  - 높이/각도 기준으로 정렬된 시퀀스
  - 공간적 순서가 있는 데이터 (전주 측정 데이터)

### 2.2 현재 모델 구조 (LSTM)
```python
Input(sequence_length, 3)
  ↓
Bidirectional LSTM(128) → BatchNorm → Dropout
  ↓
Bidirectional LSTM(64) → BatchNorm → Dropout
  ↓
Dense(64) → BatchNorm → Dropout
  ↓
Dense(32) → Dropout
  ↓
Dense(1, sigmoid)  # Binary classification
```

## 3. ResNet 적용 가능성

### 3.1 적용 가능: 1D ResNet 사용

**1D ResNet**은 시계열/시퀀스 데이터에 적용 가능하며, 현재 시스템에 적용할 수 있습니다.

#### 장점:
1. **깊은 네트워크 학습**: 더 복잡한 패턴 학습 가능
2. **Skip Connection**: Gradient flow 개선
3. **공간적 패턴 학습**: CNN 기반이므로 지역적 패턴에 강함
4. **병렬 처리**: LSTM보다 빠른 학습 가능

#### 1D ResNet 구조 예시:
```python
Input(sequence_length, 3)
  ↓
Conv1D(64, kernel=7) → BatchNorm → ReLU
  ↓
MaxPooling1D
  ↓
Residual Block 1 (64 filters)
  ↓
Residual Block 2 (128 filters)
  ↓
Residual Block 3 (256 filters)
  ↓
GlobalAveragePooling1D
  ↓
Dense(128) → Dropout
  ↓
Dense(1, sigmoid)
```

### 3.2 LSTM vs ResNet 비교

| 특성 | LSTM | 1D ResNet |
|------|------|-----------|
| **시계열 의존성** | 강함 (메모리 셀) | 약함 (CNN 기반) |
| **공간적 패턴** | 약함 | 강함 (컨볼루션) |
| **학습 속도** | 느림 (순차 처리) | 빠름 (병렬 처리) |
| **메모리 사용** | 높음 | 낮음 |
| **장기 의존성** | 우수 | 제한적 |
| **지역 패턴** | 제한적 | 우수 |

### 3.3 현재 데이터에 ResNet 적용 시 고려사항

#### ✅ 적용 가능한 이유:
1. **시퀀스 길이가 고정됨**: sequence_length=50으로 고정
2. **공간적 패턴 존재**: 높이/각도 순서로 정렬된 데이터
3. **지역적 특징 중요**: 파단 패턴이 특정 구간에 집중

#### ⚠️ 주의사항:
1. **순서 의존성**: LSTM은 순서를 명시적으로 모델링하지만, ResNet은 암묵적으로만 처리
2. **장기 의존성**: 50개 시퀀스 전체를 고려하는 능력이 LSTM보다 약할 수 있음
3. **데이터 특성**: 현재는 높이/각도 기준 정렬이지만, 실제로는 공간적 측정 데이터

## 4. ResNet 적용 방법

### 4.1 1D ResNet 모델 구현 예시

```python
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, Add,
    MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
)

def residual_block(x, filters, kernel_size=3, stride=1):
    """Residual Block for 1D"""
    shortcut = x
    
    # Main path
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet_1d(sequence_length=50, n_features=3):
    """1D ResNet 모델 구성"""
    inputs = Input(shape=(sequence_length, n_features))
    
    # Initial convolution
    x = Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
```

### 4.2 하이브리드 접근법 (LSTM + ResNet)

두 모델의 장점을 결합할 수 있습니다:

```python
def build_hybrid_model(sequence_length=50, n_features=3):
    """LSTM + ResNet 하이브리드 모델"""
    inputs = Input(shape=(sequence_length, n_features))
    
    # ResNet branch (공간적 패턴)
    resnet_branch = Conv1D(64, 7, padding='same')(inputs)
    resnet_branch = BatchNormalization()(resnet_branch)
    resnet_branch = Activation('relu')(resnet_branch)
    resnet_branch = residual_block(resnet_branch, 64)
    resnet_branch = GlobalAveragePooling1D()(resnet_branch)
    
    # LSTM branch (시계열 패턴)
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_branch = Bidirectional(LSTM(32, return_sequences=False))(lstm_branch)
    
    # Concatenate
    x = Concatenate()([resnet_branch, lstm_branch])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
```

## 5. 권장사항

### 5.1 현재 상황에서의 선택

1. **LSTM 유지** (현재 방식)
   - ✅ 시계열 의존성이 중요한 경우
   - ✅ 순서 정보가 핵심인 경우
   - ✅ 현재 성능이 만족스러운 경우

2. **1D ResNet 시도**
   - ✅ 공간적 패턴이 더 중요한 경우
   - ✅ 학습 속도가 중요한 경우
   - ✅ 더 깊은 네트워크가 필요한 경우

3. **하이브리드 모델** (권장)
   - ✅ 두 모델의 장점 결합
   - ✅ 더 강력한 특징 추출
   - ✅ 성능 향상 가능성 높음

### 5.2 실험 순서

1. **1D ResNet 구현 및 학습**
   - 현재 LSTM과 동일한 데이터로 학습
   - 성능 비교 (정확도, 재현율, 정밀도, F1)

2. **하이브리드 모델 시도**
   - LSTM + ResNet 결합
   - 앙상블 효과 기대

3. **하이퍼파라미터 튜닝**
   - ResNet 깊이, 필터 수 조정
   - 학습률, 배치 크기 최적화

## 6. 결론

**ResNet은 현재 시스템에 적용 가능합니다!**

- **1D ResNet**을 사용하면 시퀀스 데이터에 적용 가능
- LSTM과 다른 접근 방식으로 **공간적 패턴**에 강점
- **하이브리드 모델**로 두 모델의 장점을 결합하는 것이 가장 효과적일 수 있음
- 실제 성능은 데이터에 따라 다르므로 **실험을 통해 검증** 필요

현재 LSTM 모델의 성능이 만족스럽지 않다면, ResNet 또는 하이브리드 모델을 시도해볼 가치가 있습니다.
