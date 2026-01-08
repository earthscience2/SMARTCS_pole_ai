import poledb as PDB
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def pole_Info(poleid):
    pole_sgn_count = PDB.get_meas_result_count(poleid)
    re_out = PDB.get_meas_result(poleid, 'OUT')
    num_sig_out = re_out.shape[0]
    out_num = num_sig_out
    return out_num, re_out

# 서버 및 그룹 정보
server = "main"
group_name = "강원강릉-202306"
PDB.poledb_init(server)
Pole_list = PDB.get_pole_list(group_name)

# 시퀀스 길이와 특성 크기 설정 (학습 시 사용한 값과 동일하게 설정)
sequence_length = 30

model_name = "model_38"

# 모델 생성 및 가중치 로드 (compile=False 옵션 추가)
model = load_model('result_ai/'+model_name+'.keras', compile=False)
#model = load_model('result_ai/'+model_name+'.h5', compile=False)
#model.load_weights('result_ai/'+model_name+'.weights.h5')
#model.summary()
break_pole_list = []

# 각 폴(poleid)에 대해 데이터 로드 및 예측 수행
for poleid in Pole_list['poleid']:
    out_num, re_out = pole_Info(poleid)
    for kk in range(out_num):
        stype = 'OUT'
        num = int(re_out['measno'][kk])
        
        # 데이터를 PDB에서 가져오기
        out_x = PDB.get_meas_data(poleid, num, stype, 'x')
        out_y = PDB.get_meas_data(poleid, num, stype, 'y')
        out_z = PDB.get_meas_data(poleid, num, stype, 'z')
        out_x.columns = [f'x_{col}' for col in out_x.columns]
        out_y.columns = [f'y_{col}' for col in out_y.columns]
        out_z.columns = [f'z_{col}' for col in out_z.columns]
        
        # 데이터 병합 및 전처리
        combined_data = pd.concat([out_x, out_y, out_z], axis=1)
        combined_data = combined_data.drop(columns=[
            'x_idx', 'x_measno', 'x_poleid', 'x_groupname', 'x_devicetype', 'x_axis',
            'y_idx', 'y_measno', 'y_poleid', 'y_groupname', 'y_devicetype', 'y_axis',
            'z_idx', 'z_measno', 'z_poleid', 'z_groupname', 'z_devicetype', 'z_axis'
        ])
        # 데이터를 시퀀스 단위로 분할
        X_sequence = []
        for i in range(len(combined_data) - sequence_length + 1):
            X_sequence.append(combined_data.iloc[i:i+sequence_length].values)
        
        X_sequence = np.array(X_sequence)

        if len(X_sequence) > 40:
        # 예측 수행
            predictions = model.predict(X_sequence)
        
        # 파단 확률 계산 및 출력
        max_prediction = np.max(predictions)  # 최대 확률 값
        #print(f"폴 {poleid}, 측정 번호 {num}, 파단 확률: {max_prediction:.4f}")
        
        # 결과 저장
        break_pole_list.append([poleid, kk, max_prediction])
        print([poleid, kk, max_prediction])

# 결과를 파단 확률(max_prob)을 기준으로 내림차순으로 정렬
sorted_break_pole_list = sorted(break_pole_list, key=lambda x: x[2], reverse=True)
def remove_duplicates_and_keep_highest(break_pole_list):
    pole_dict = {}
    
    # 각 전주 ID에 대해 가장 높은 확률 값을 찾아 저장
    for pole in break_pole_list:
        pole_id = pole[0]
        probability = pole[2]
        
        # 이미 해당 전주 ID가 저장되어 있으면 더 높은 확률로 업데이트
        if pole_id in pole_dict:
            if probability > pole_dict[pole_id][2]:
                pole_dict[pole_id] = pole  # 더 높은 확률로 대체
        else:
            pole_dict[pole_id] = pole  # 처음 등장하는 전주 ID 저장

    # 중복 제거된 리스트 반환
    return list(pole_dict.values())
sorted_break_pole_list = remove_duplicates_and_keep_highest(sorted_break_pole_list)
print(sorted_break_pole_list[:20])

# 전주 ID를 기준으로 정렬된 리스트에서 해당 전주의 위치를 찾는 함수
def find_pole_position(pole_list, pole_id):
    positions = []
    
    # 전주 ID가 여러 번 등장하는 경우, 그 값과 함께 위치 저장
    for index, pole in enumerate(pole_list):
        if pole[0] == pole_id:  # pole[0]이 전주 ID라고 가정
            positions.append((index + 1, pole[2]))  # 인덱스와 해당 값 (가중치)

    if positions:
        # 가장 높은 값을 가진 항목의 인덱스 반환
        highest_position = max(positions, key=lambda x: x[1])  # 값이 가장 큰 항목 선택
        return highest_position[0]  # 인덱스 반환
    else:
        return -1  # 전주를 찾지 못한 경우 -1 반환

pole_id = '90281312'
position = find_pole_position(sorted_break_pole_list, pole_id)

if position != -1:
    print(f"전주 '{pole_id}'는 정렬된 리스트에서 {position}번째에 위치합니다. 전체 {len(sorted_break_pole_list)} 개")
else:
    print(f"전주 '{pole_id}'는 리스트에 없습니다.")

pole_id = '9243E511'
position = find_pole_position(sorted_break_pole_list, pole_id)

if position != -1:
    print(f"전주 '{pole_id}'는 정렬된 리스트에서 {position}번째에 위치합니다. 전체 {len(sorted_break_pole_list)} 개")
else:
    print(f"전주 '{pole_id}'는 리스트에 없습니다.")

# 전주 ID '9322B933'의 위치 찾기
pole_id = '9243G972'
position = find_pole_position(sorted_break_pole_list, pole_id)

if position != -1:
    print(f"전주 '{pole_id}'는 정렬된 리스트에서 {position}번째에 위치합니다. 전체 {len(sorted_break_pole_list)} 개")
else:
    print(f"전주 '{pole_id}'는 리스트에 없습니다.")



