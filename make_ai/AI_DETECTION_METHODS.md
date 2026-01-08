# 전주 파단 검출 AI 방법 추천 가이드

## 📊 현재 상황 분석

### 데이터 특성
- **데이터 타입**: 시계열 데이터 (시간에 따른 센서 측정값)
- **채널**: 8개 채널 (ch1~ch8)
- **축**: X, Y, Z 3축
- **데이터 종류**: IN (내부), OUT (외부)
- **레이블**: 정상(Normal) vs 파단(Break)
- **추가 정보**: 파단 위치 (높이, 각도)

### 현재 사용 중인 방법
- **모델**: Conv1D + Bidirectional LSTM 하이브리드
- **시퀀스 길이**: 30
- **데이터 증강**: SMOTE
- **평가 지표**: Accuracy, ROC-AUC, F-Score

---

## 🎯 추천 검출 방법 (우선순위별)

### 1. **현재 방법 개선 (Conv1D + LSTM) - ⭐⭐⭐⭐⭐**

**추천 이유**: 이미 구현되어 있고 시계열 데이터에 적합

**개선 방안**:

#### A. Attention Mechanism 추가
```python
from tensorflow.keras.layers import Attention, MultiHeadAttention

model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(30, features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=True)),
    # Attention 추가
    MultiHeadAttention(num_heads=8, key_dim=64),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**장점**:
- 파단 구간에 더 집중 가능
- 해석 가능성 향상 (어느 부분이 중요한지 시각화 가능)

#### B. 파단 위치 정보 활용
```python
# 파단 위치(높이, 각도)를 추가 피처로 사용
break_height = ...  # 정규화된 파단 높이
break_degree = ...  # 정규화된 파단 각도

# 시퀀스 데이터와 결합
combined_features = np.concatenate([
    sequence_data,
    np.tile([break_height, break_degree], (sequence_length, 1))
], axis=-1)
```

**장점**:
- 도메인 지식 활용
- 정확도 향상 가능

#### C. Multi-scale Feature Extraction
```python
# 다양한 커널 크기로 특징 추출
conv1 = Conv1D(64, kernel_size=3, activation='relu')(input)
conv2 = Conv1D(64, kernel_size=5, activation='relu')(input)
conv3 = Conv1D(64, kernel_size=7, activation='relu')(input)
merged = Concatenate()([conv1, conv2, conv3])
```

**장점**:
- 다양한 시간 스케일의 패턴 포착
- 단기/장기 패턴 모두 학습

---

### 2. **Transformer 기반 모델 - ⭐⭐⭐⭐**

**추천 이유**: 시계열 데이터에서 최신 성능을 보이는 아키텍처

**구현 예시**:
```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

# 모델 구성
inputs = Input(shape=(sequence_length, features))
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128)
x = GlobalAveragePooling1D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)
```

**장점**:
- 장거리 의존성 학습에 우수
- Attention으로 중요한 구간 식별 가능
- 최신 기술 트렌드

**단점**:
- 학습 시간이 길 수 있음
- 데이터가 적으면 과적합 위험

---

### 3. **Ensemble 방법 - ⭐⭐⭐⭐**

**추천 이유**: 단일 모델보다 안정적이고 정확도 향상

**구현 방법**:

#### A. Voting Ensemble
```python
# 여러 모델 조합
models = [
    conv1d_lstm_model,  # 현재 모델
    transformer_model,   # Transformer 모델
    cnn_model           # 순수 CNN 모델
]

# 예측 결합
predictions = []
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

# 평균 또는 가중 평균
final_pred = np.mean(predictions, axis=0)
# 또는
weights = [0.4, 0.4, 0.2]  # 모델별 가중치
final_pred = np.average(predictions, axis=0, weights=weights)
```

#### B. Stacking
```python
# 1단계: 여러 모델로 예측
base_predictions = []
for model in base_models:
    pred = model.predict(X_train)
    base_predictions.append(pred)

# 2단계: 메타 모델 학습
meta_X = np.column_stack(base_predictions)
meta_model = LogisticRegression()
meta_model.fit(meta_X, y_train)
```

**장점**:
- 단일 모델의 한계 보완
- 더 안정적인 예측
- 정확도 향상 가능

---

### 4. **이상 탐지 (Anomaly Detection) 접근법 - ⭐⭐⭐**

**추천 이유**: 파단이 드물게 발생하는 경우에 유용

**방법**:

#### A. Autoencoder 기반
```python
# 정상 데이터만으로 학습
encoder = Sequential([
    Dense(128, activation='relu', input_shape=(features,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu')  # 압축 표현
])

decoder = Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dense(128, activation='relu'),
    Dense(features, activation='sigmoid')
])

autoencoder = Model(encoder.input, decoder(encoder(encoder.input)))
autoencoder.compile(optimizer='adam', loss='mse')

# 정상 데이터만으로 학습
autoencoder.fit(normal_data, normal_data, epochs=50)

# 재구성 오차가 높으면 파단
reconstruction_error = np.mean((data - autoencoder.predict(data))**2, axis=1)
is_break = reconstruction_error > threshold
```

#### B. Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# 정상 데이터로 학습
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(normal_data)

# 예측 (-1: 이상, 1: 정상)
predictions = iso_forest.predict(test_data)
```

**장점**:
- 정상 데이터만으로 학습 가능
- 새로운 패턴에도 대응 가능

**단점**:
- 파단 위치 정보 활용 어려움
- 임계값 설정이 중요

---

### 5. **특징 기반 방법 - ⭐⭐⭐**

**추천 이유**: 해석 가능성과 빠른 학습

**추출할 특징**:
```python
def extract_features(data):
    features = []
    
    # 통계적 특징
    features.append(np.mean(data, axis=0))      # 평균
    features.append(np.std(data, axis=0))       # 표준편차
    features.append(np.max(data, axis=0))        # 최대값
    features.append(np.min(data, axis=0))        # 최소값
    features.append(np.median(data, axis=0))     # 중앙값
    
    # 주파수 도메인 특징
    fft = np.fft.fft(data, axis=0)
    features.append(np.abs(fft[:10]))           # 주파수 성분
    
    # 파단 위치 관련 특징
    # 파단 위치 근처의 특징 강조
    break_region_features = extract_break_region_features(data, break_position)
    features.append(break_region_features)
    
    return np.concatenate(features)

# XGBoost 또는 LightGBM으로 학습
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(extract_features(train_data), train_labels)
```

**장점**:
- 빠른 학습
- 해석 가능
- 도메인 지식 반영 용이

**단점**:
- 특징 엔지니어링 필요
- 시계열 정보 손실 가능

---

### 6. **시계열 전용 모델 - ⭐⭐⭐⭐**

#### A. Temporal Convolutional Network (TCN)
```python
from tcn import TCN

model = Sequential([
    TCN(nb_filters=64, kernel_size=3, nb_stacks=2, 
        dilations=[1, 2, 4, 8], padding='causal', use_skip_connections=True),
    Dense(1, activation='sigmoid')
])
```

**장점**:
- 시계열에 특화
- 병렬 처리 가능 (LSTM보다 빠름)
- 장거리 의존성 학습

#### B. WaveNet
```python
# Dilated Causal Convolutions 사용
# 시계열 생성 모델의 원리를 분류에 적용
```

---

## 🎯 상황별 추천

### 데이터가 충분한 경우 (10,000개 이상)
1. **Transformer 기반 모델** + Ensemble
2. **개선된 Conv1D+LSTM** (Attention 추가)
3. **TCN** 모델

### 데이터가 적은 경우 (1,000개 미만)
1. **특징 기반 방법** (XGBoost/LightGBM)
2. **Transfer Learning** (사전 학습된 모델 활용)
3. **Data Augmentation 강화**

### 해석 가능성이 중요한 경우
1. **Attention 기반 모델** (중요 구간 시각화)
2. **특징 기반 방법** (SHAP 값으로 설명)
3. **Grad-CAM** (CNN 레이어 활성화 시각화)

### 실시간 검출이 필요한 경우
1. **경량화된 CNN 모델**
2. **특징 기반 방법** (빠른 추론)
3. **모델 양자화/최적화**

---

## 🔧 실용적 개선 제안

### 1. 데이터 전처리 개선
```python
# 파단 위치 중심으로 윈도우 조정
def create_break_centered_sequences(data, break_position, window_size=30):
    # 파단 위치를 중심으로 시퀀스 추출
    start = max(0, break_position - window_size // 2)
    end = min(len(data), break_position + window_size // 2)
    return data[start:end]
```

### 2. 클래스 불균형 처리
```python
# 현재: SMOTE
# 추가 고려:
# - Focal Loss (파단 클래스에 더 집중)
# - Class weights 조정
# - Hard negative mining
```

### 3. 평가 지표 개선
```python
# 현재: Accuracy, ROC-AUC, F-Score
# 추가 고려:
# - Precision@K (상위 K개 중 파단 비율)
# - Recall@K (실제 파단 중 상위 K개에 포함된 비율)
# - Confusion Matrix 분석
```

### 4. 앙상블 전략
```python
# 모델별 특화
# - Conv1D+LSTM: 전체 패턴 학습
# - Transformer: 장거리 의존성
# - 특징 기반: 빠른 예측
```

---

## 📈 성능 향상을 위한 체크리스트

- [ ] 파단 위치 정보를 피처로 추가
- [ ] Attention 메커니즘 도입
- [ ] 다양한 시퀀스 길이 실험 (20, 30, 50, 100)
- [ ] 데이터 증강 강화 (Time Warping, Scaling)
- [ ] 교차 검증으로 모델 안정성 확인
- [ ] 앙상블 모델 구축
- [ ] 하이퍼파라미터 최적화 (Optuna, Ray Tune)
- [ ] 모델 해석 도구 활용 (SHAP, LIME)

---

## 🚀 즉시 적용 가능한 개선안

### 1단계: 현재 모델 개선 (1-2주)
- Attention 레이어 추가
- 파단 위치 정보 피처 추가
- 하이퍼파라미터 튜닝

### 2단계: 앙상블 구축 (2-3주)
- Transformer 모델 추가
- Voting/Stacking 구현
- 성능 비교

### 3단계: 고급 기법 도입 (1-2개월)
- TCN 모델 실험
- Autoencoder 기반 이상 탐지
- 실시간 추론 최적화

---

## 📚 참고 자료

- **Transformer for Time Series**: "Temporal Fusion Transformers"
- **TCN**: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks"
- **Attention**: "Attention Is All You Need"
- **Ensemble**: "Ensemble Methods in Machine Learning"

---

## 💡 최종 추천

**현재 상황에 가장 적합한 방법**:

1. **단기 (1-2개월)**: 현재 Conv1D+LSTM 모델에 **Attention + 파단 위치 피처** 추가
2. **중기 (2-3개월)**: **Transformer 모델** 추가 후 **Ensemble** 구축
3. **장기 (3-6개월)**: **TCN**, **특징 기반 모델** 등 다양한 방법 실험 및 최적 조합 찾기

**핵심 포인트**:
- 파단 위치 정보 활용 필수
- Attention으로 해석 가능성 확보
- Ensemble으로 안정성 확보

