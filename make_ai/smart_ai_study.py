import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

# 변수 설정
NORMAL_DATA_PATH = 'raw_data_ai_csv_check/normal/'
BREAK_DATA_PATH = 'raw_data_ai_csv_check/break/'
SEQUENCE_LENGTH = 30
EPOCHS = 50  # 에폭 수
BATCH_SIZE = 32  # 배치 크기 조정

# CSV 파일 목록 가져오기
def get_csv_files(path):
    return glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)

# 데이터 로드 및 병합 함수 정의
def load_and_merge_data(files, label):
    data_list = []
    for file in files:
        try:
            filename = os.path.basename(file)
            poleid = filename.split('_')[0]
            data = pd.read_csv(file)
            data['label'] = label
            data['poleid'] = poleid
            data_list.append(data)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file}, 오류: {e}")
            continue
    return data_list

# 데이터 로드
normal_files = get_csv_files(NORMAL_DATA_PATH)
break_files = get_csv_files(BREAK_DATA_PATH)

normal_data_list = load_and_merge_data(normal_files, label=0)
break_data_list = load_and_merge_data(break_files, label=1)

# 데이터프레임 결합 및 전처리
data_df = pd.concat(normal_data_list + break_data_list, ignore_index=True).fillna(0)
labels = data_df.pop('label').values
poleids = data_df.pop('poleid').values

# 피처 스케일링
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_df)

# 시퀀스 생성 함수 정의
def create_sequences(X_scaled, y, sequence_length):
    X_sequence, y_sequence = [], []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_sequence.append(X_scaled[i:i + sequence_length])
        y_sequence.append(y[i + sequence_length - 1])
    return np.array(X_sequence), np.array(y_sequence)

# 시퀀스 생성
X_sequence, y_sequence = create_sequences(data_scaled, labels, SEQUENCE_LENGTH)

# 데이터 분할
X_train_sequence, X_test_sequence, y_train_sequence, y_test_sequence = train_test_split(
    X_sequence, y_sequence, test_size=0.2, random_state=42)

# 데이터 증강 (SMOTE 적용)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_sequence.reshape(len(X_train_sequence), -1), y_train_sequence)

# 다시 시퀀스 형태로 변환
X_train_sequence = X_resampled.reshape(len(X_resampled), SEQUENCE_LENGTH, X_train_sequence.shape[2])
y_train_sequence = y_resampled

# 학습률 스케줄러 (동적 조정)
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.7  # 학습률 감소율 변경

lr_scheduler = LearningRateScheduler(scheduler)

# Gradient Clipping 설정
optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)

# 모델 정의 (복잡성 증가 + Batch Normalization 추가)
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train_sequence.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),  # Dropout 비율 조정
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),  # 추가 Dense 레이어
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping 적용
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(
    X_train_sequence, y_train_sequence,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# 모델 평가
loss, accuracy = model.evaluate(X_test_sequence, y_test_sequence)
print(f'테스트 정확도: {accuracy}')

# 예측 및 평가
y_pred_probs = model.predict(X_test_sequence).ravel()

# ROC 커브 계산
fpr, tpr, thresholds = roc_curve(y_test_sequence, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve로 최적의 임계값 찾기
precision, recall, thresholds = precision_recall_curve(y_test_sequence, y_pred_probs)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
best_thresh = thresholds[ix]
print(f'Best Threshold={best_thresh}, F-Score={fscore[ix]}')

# 최적의 임계값으로 예측 클래스 생성
y_pred_classes = (y_pred_probs > best_thresh).astype(int)

# 분류 보고서 출력
class_report = classification_report(y_test_sequence, y_pred_classes, target_names=['Normal', 'Break'])
print("\n분류 보고서:")
print(class_report)

# 모델 저장
model.save('result_ai/model_67.keras')
print("모델 저장 완료")
