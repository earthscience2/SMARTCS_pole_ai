import glob
import json
import poledb as PDB
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def check_csv():
    server_list = ["main", "is", "kh"]
    user_input = None  

    def on_key(event):
        """키보드 입력 핸들러 함수"""
        nonlocal user_input
        if event.key == 'y':
            user_input = 'y'
            plt.close()  # 그래프 창 닫기
        elif event.key == 'n':
            user_input = 'n'
            plt.close()  # 그래프 창 닫기

    for server in server_list:
        PDB.poledb_init(server)
        groupname_list = PDB.groupname_info()

        # 파단 데이터 수집
        print(server + " 서버 // 추가 정상 데이터 찾는 중....")
        for group_name in tqdm(groupname_list, desc="Processing poles"):
            x_files = glob.glob(f"raw_data_ai_csv/normal/{group_name}/*_H_x.csv")
            y_files = glob.glob(f"raw_data_ai_csv/normal/{group_name}/*_H_y.csv")
            z_files = glob.glob(f"raw_data_ai_csv/normal/{group_name}/*_H_z.csv")
            
            json_file_path = (f"raw_data_ai_json/normal/{group_name}.json")
            
            with open(json_file_path, 'r', encoding='utf-8') as file:
                normal_data = json.load(file)
            
            try:
                for x_file, y_file, z_file in zip(x_files, y_files, z_files):
                    file_name = ((x_file.split("\\"))[-1])
                    file_name_list = file_name.split("_")
                    pole_name = file_name_list[0]
                    file_name = file_name[:-8]

                    for i in normal_data:
                        if not (os.path.exists("raw_data_ai_csv_check/break/" + file_name + ".csv") or os.path.exists("raw_data_ai_csv_check/normal/" + file_name + ".csv") or os.path.exists("raw_data_ai_csv_check/other/" + file_name + ".csv")):
                            if i == pole_name:
                                x_data = pd.read_csv(x_file)
                                y_data = pd.read_csv(y_file)
                                z_data = pd.read_csv(z_file)
                                
                                # 칼럼 이름 재설정 (중복 방지)
                                x_data.columns = [f'x_{col}' for col in x_data.columns]
                                y_data.columns = [f'y_{col}' for col in y_data.columns]
                                z_data.columns = [f'z_{col}' for col in z_data.columns]

                                # 피규어 생성 및 서브플롯 설정
                                fig, axs = plt.subplots(3, 1, figsize=(8, 8))
                                
                                # 첫 번째 서브플롯: x_data의 모든 열을 플롯
                                for col in ['x_ch1','x_ch2','x_ch3','x_ch4','x_ch5','x_ch6','x_ch7','x_ch8']:
                                    axs[0].plot(x_data.index, x_data[col])
                                axs[0].set_title('X Data (All Columns)')
                                axs[0].set_ylabel('Values')

                                # 두 번째 서브플롯: y_data의 모든 열을 플롯
                                for col in ['y_ch1','y_ch2','y_ch3','y_ch4','y_ch5','y_ch6','y_ch7','y_ch8']:
                                    axs[1].plot(y_data.index, y_data[col])
                                axs[1].set_title('Y Data (All Columns)')
                                axs[1].set_ylabel('Values')

                                # 세 번째 서브플롯: z_data의 모든 열을 플롯
                                for col in ['z_ch1','z_ch2','z_ch3','z_ch4','z_ch5','z_ch6','z_ch7','z_ch8']:
                                    axs[2].plot(z_data.index, z_data[col])
                                axs[2].set_title('Z Data (All Columns)')
                                axs[2].set_ylabel('Values')
                                axs[2].set_xlabel('Index')

                                # 그래프 간의 여백을 조정
                                plt.tight_layout()

                                # 그래프 이벤트 핸들러 등록n
                                fig.canvas.mpl_connect('key_press_event', on_key)

                                # 그래프 보여주기 (비블로킹 모드)
                                plt.show(block=False)

                                # 사용자 입력 대기 (y 또는 n 입력)
                                while user_input is None:
                                    plt.pause(0.1)

                                # 입력에 따른 데이터 처리
                                if user_input == 'y':
                                    if not (x_data.empty or y_data.empty or z_data.empty):
                                        combined_data = pd.concat([x_data, y_data, z_data], axis=1)
                                        combined_data = combined_data.drop(columns=['x_idx','x_measno','x_poleid','x_groupname','x_devicetype','x_axis','y_idx','y_measno','y_poleid','y_groupname','y_devicetype','y_axis','z_idx', 'z_measno','z_poleid','z_groupname','z_devicetype','z_axis'])
                                        combined_data['label'] = 0  # 정상 데이터 레이블
                                        combined_data.to_csv(os.path.join("raw_data_ai_csv_check/normal", file_name + ".csv"), index=False)
                                elif user_input == 'n':
                                    if not (x_data.empty or y_data.empty or z_data.empty):
                                        combined_data = pd.concat([x_data, y_data, z_data], axis=1)
                                        combined_data = combined_data.drop(columns=['x_idx','x_measno','x_poleid','x_groupname','x_devicetype','x_axis','y_idx','y_measno','y_poleid','y_groupname','y_devicetype','y_axis','z_idx', 'z_measno','z_poleid','z_groupname','z_devicetype','z_axis'])
                                        combined_data['label'] = 2  # 비정상 데이터 레이블
                                        combined_data.to_csv(os.path.join("raw_data_ai_csv_check/other", file_name + ".csv"), index=False)

                                # 입력 후 변수 초기화
                                user_input = None

            except Exception as e:
                print("오류 발생: ", e)

check_csv()