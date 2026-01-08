#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전주 데이터 폴더 경로를 받아서 해당 전주의 원본 데이터 그래프를 그리는 스크립트

사용법:
    python plot_pole_data.py "make_ai/raw_pole_data/break/강릉지사-2506/8732F191"
    또는
    python plot_pole_data.py
    (인터랙티브 모드로 경로 입력)
"""

import sys
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정 (macOS)
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

def load_break_info(pole_dir):
    """
    파단 정보 JSON 파일을 읽기
    
    Args:
        pole_dir: 전주 데이터 폴더 경로
    
    Returns:
        dict: 파단 정보 또는 None
    """
    break_info_file = os.path.join(pole_dir, "*_break_info.json")
    json_files = glob.glob(break_info_file)
    
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def plot_pole_data(pole_dir_path):
    """
    전주 데이터 폴더의 CSV 파일들을 읽어서 그래프로 그리기
    
    Args:
        pole_dir_path: 전주 데이터 폴더 경로
    """
    # 절대 경로로 변환
    if not os.path.isabs(pole_dir_path):
        # 현재 스크립트 위치 기준으로 경로 조정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pole_dir_path = os.path.join(current_dir, pole_dir_path)
    
    pole_dir = os.path.abspath(pole_dir_path)
    
    if not os.path.exists(pole_dir):
        print(f"오류: 폴더를 찾을 수 없습니다: {pole_dir}")
        return
    
    if not os.path.isdir(pole_dir):
        print(f"오류: 경로가 폴더가 아닙니다: {pole_dir}")
        return
    
    print(f"전주 데이터 폴더: {pole_dir}")
    
    # CSV 파일 목록 가져오기
    csv_files = glob.glob(os.path.join(pole_dir, "*.csv"))
    
    if not csv_files:
        print(f"오류: CSV 파일을 찾을 수 없습니다.")
        return
    
    # 파일을 타입별로 분류 (IN_x, OUT_x, OUT_y, OUT_z)
    file_groups = {
        'IN_x': [],
        'OUT_x': [],
        'OUT_y': [],
        'OUT_z': []
    }
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if '_IN_x.csv' in filename:
            file_groups['IN_x'].append(csv_file)
        elif '_OUT_x.csv' in filename:
            file_groups['OUT_x'].append(csv_file)
        elif '_OUT_y.csv' in filename:
            file_groups['OUT_y'].append(csv_file)
        elif '_OUT_z.csv' in filename:
            file_groups['OUT_z'].append(csv_file)
    
    # 파일명으로 정렬
    for key in file_groups:
        file_groups[key].sort()
    
    # 파단 정보 읽기
    break_info = load_break_info(pole_dir)
    
    # 그래프 생성
    total_plots = sum(len(files) for files in file_groups.values())
    
    if total_plots == 0:
        print("그릴 데이터가 없습니다.")
        return
    
    # 각 파일 그룹별로 그래프 생성
    plot_idx = 0
    
    for data_type, files in file_groups.items():
        if not files:
            continue
        
        for file_path in files:
            plot_idx += 1
            filename = os.path.basename(file_path)
            
            try:
                # CSV 파일 읽기
                df = pd.read_csv(file_path)
                
                # ch1-ch8 컬럼 확인
                ch_columns = [col for col in df.columns if col.startswith('ch') and col[2:].isdigit()]
                ch_columns = sorted(ch_columns, key=lambda x: int(x[2:]))
                
                if not ch_columns:
                    print(f"경고: {filename}에 채널 데이터가 없습니다.")
                    continue
                
                # 그래프 생성
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                # 각 채널 플롯
                for ch in ch_columns:
                    ax.plot(df.index, df[ch], label=ch, linewidth=1.5)
                
                # 제목 설정
                title = f"{filename}"
                if break_info:
                    title += f"\n파단 정보 - 높이: {break_info.get('breakheight')}, 각도: {break_info.get('breakdegree')}"
                
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('Index', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show(block=False)
                
                print(f"[{plot_idx}/{total_plots}] {filename} 그래프 표시 중...")
            
            except Exception as e:
                print(f"오류: {filename} 처리 중 오류 발생: {e}")
                continue
    
    # 전체 데이터를 한 화면에 표시 (선택적)
    if total_plots > 1:
        print(f"\n총 {total_plots}개의 그래프가 표시되었습니다.")
        print("그래프를 닫으려면 창을 닫으세요.")
        plt.show(block=True)  # 마지막 그래프는 대기

def plot_all_data_combined(pole_dir_path, save_to_file=True):
    """
    전주 데이터의 모든 파일을 하나의 큰 그래프에 표시 (x, y, z를 서브플롯으로)
    
    Args:
        pole_dir_path: 전주 데이터 폴더 경로
        save_to_file: True면 파일로 저장, False면 화면에 표시
    """
    # 절대 경로로 변환
    if not os.path.isabs(pole_dir_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pole_dir_path = os.path.join(current_dir, pole_dir_path)
    
    pole_dir = os.path.abspath(pole_dir_path)
    
    # 경로에서 정보 추출 (break/프로젝트명/전주ID)
    path_parts = pole_dir.split(os.sep)
    
    # raw_pole_data 이후의 경로 구조 파싱
    try:
        raw_pole_data_idx = path_parts.index('raw_pole_data')
        if raw_pole_data_idx + 3 <= len(path_parts):
            category = path_parts[raw_pole_data_idx + 1]  # break 또는 normal
            project_name = path_parts[raw_pole_data_idx + 2]  # 강릉지사-2506
            poleid = path_parts[raw_pole_data_idx + 3]  # 8732F191
        else:
            category = None
            project_name = None
            poleid = os.path.basename(pole_dir)
    except ValueError:
        # raw_pole_data가 경로에 없는 경우
        category = None
        project_name = None
        poleid = os.path.basename(pole_dir)
    
    # 경로 정보 변수
    pole_info = {
        'category': category,  # 'break' 또는 'normal'
        'project_name': project_name,  # '강릉지사-2506'
        'poleid': poleid,  # '8732F191'
        'full_path': pole_dir
    }
    
    print(f"전주 정보:")
    print(f"  카테고리: {pole_info['category']}")
    print(f"  프로젝트명: {pole_info['project_name']}")
    print(f"  전주ID: {pole_info['poleid']}")
    print(f"  전체 경로: {pole_info['full_path']}")
    
    if not os.path.exists(pole_dir):
        print(f"오류: 폴더를 찾을 수 없습니다: {pole_dir}")
        return
    
    # OUT_x, OUT_y, OUT_z 파일 찾기 (첫 번째 파일만)
    out_x_files = sorted(glob.glob(os.path.join(pole_dir, "*_OUT_x.csv")))
    out_y_files = sorted(glob.glob(os.path.join(pole_dir, "*_OUT_y.csv")))
    out_z_files = sorted(glob.glob(os.path.join(pole_dir, "*_OUT_z.csv")))
    
    # IN_x 파일 찾기
    in_x_files = sorted(glob.glob(os.path.join(pole_dir, "*_IN_x.csv")))
    
    # 파단 정보 읽기
    break_info = load_break_info(pole_dir)
    
    # 그래프 생성 (IN + OUT 데이터를 함께 표시)
    # IN_x가 있으면 4개 서브플롯, 없으면 3개 서브플롯
    has_in_data = len(in_x_files) > 0
    has_out_data = len(out_x_files) > 0 or len(out_y_files) > 0 or len(out_z_files) > 0
    
    if has_in_data or has_out_data:
        num_subplots = 4 if has_in_data and has_out_data else (1 if has_in_data else 3)
        fig, axs = plt.subplots(num_subplots, 1, figsize=(14, 12 if has_in_data else 10))
        
        # axs를 리스트로 변환 (서브플롯이 1개인 경우)
        if num_subplots == 1:
            axs = [axs]
        
        plot_idx = 0
        
        # IN X 데이터 (첫 번째 서브플롯)
        if in_x_files:
            try:
                df_in_x = pd.read_csv(in_x_files[0])
                ch_columns = [col for col in df_in_x.columns if col.startswith('ch') and col[2:].isdigit()]
                ch_columns = sorted(ch_columns, key=lambda x: int(x[2:]))
                
                for ch in ch_columns:
                    axs[plot_idx].plot(df_in_x.index, df_in_x[ch], label=ch, linewidth=1.5)
                axs[plot_idx].set_title(f'IN X Data - {os.path.basename(in_x_files[0])}', fontsize=11)
                axs[plot_idx].set_ylabel('Value', fontsize=10)
                axs[plot_idx].legend(loc='upper right', fontsize=8, ncol=4)
                axs[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            except Exception as e:
                axs[plot_idx].text(0.5, 0.5, f'IN X 오류: {e}', transform=axs[plot_idx].transAxes, ha='center')
                plot_idx += 1
        
        # OUT X 데이터
        if out_x_files:
            try:
                df_x = pd.read_csv(out_x_files[0])
                ch_columns = [col for col in df_x.columns if col.startswith('ch') and col[2:].isdigit()]
                ch_columns = sorted(ch_columns, key=lambda x: int(x[2:]))
                
                for ch in ch_columns:
                    axs[plot_idx].plot(df_x.index, df_x[ch], label=ch, linewidth=1.5)
                axs[plot_idx].set_title(f'OUT X Data - {os.path.basename(out_x_files[0])}', fontsize=11)
                axs[plot_idx].set_ylabel('Value', fontsize=10)
                axs[plot_idx].legend(loc='upper right', fontsize=8, ncol=4)
                axs[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            except Exception as e:
                axs[plot_idx].text(0.5, 0.5, f'OUT X 오류: {e}', transform=axs[plot_idx].transAxes, ha='center')
                plot_idx += 1
        
        # OUT Y 데이터
        if out_y_files:
            try:
                df_y = pd.read_csv(out_y_files[0])
                ch_columns = [col for col in df_y.columns if col.startswith('ch') and col[2:].isdigit()]
                ch_columns = sorted(ch_columns, key=lambda x: int(x[2:]))
                
                for ch in ch_columns:
                    axs[plot_idx].plot(df_y.index, df_y[ch], label=ch, linewidth=1.5)
                axs[plot_idx].set_title(f'OUT Y Data - {os.path.basename(out_y_files[0])}', fontsize=11)
                axs[plot_idx].set_ylabel('Value', fontsize=10)
                axs[plot_idx].legend(loc='upper right', fontsize=8, ncol=4)
                axs[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            except Exception as e:
                axs[plot_idx].text(0.5, 0.5, f'OUT Y 오류: {e}', transform=axs[plot_idx].transAxes, ha='center')
                plot_idx += 1
        
        # OUT Z 데이터 (마지막 서브플롯)
        if out_z_files:
            try:
                df_z = pd.read_csv(out_z_files[0])
                ch_columns = [col for col in df_z.columns if col.startswith('ch') and col[2:].isdigit()]
                ch_columns = sorted(ch_columns, key=lambda x: int(x[2:]))
                
                for ch in ch_columns:
                    axs[plot_idx].plot(df_z.index, df_z[ch], label=ch, linewidth=1.5)
                axs[plot_idx].set_title(f'OUT Z Data - {os.path.basename(out_z_files[0])}', fontsize=11)
                axs[plot_idx].set_xlabel('Index', fontsize=10)
                axs[plot_idx].set_ylabel('Value', fontsize=10)
                axs[plot_idx].legend(loc='upper right', fontsize=8, ncol=4)
                axs[plot_idx].grid(True, alpha=0.3)
            except Exception as e:
                axs[plot_idx].text(0.5, 0.5, f'OUT Z 오류: {e}', transform=axs[plot_idx].transAxes, ha='center')
        
        # 전체 제목
        main_title = f"전주 데이터: {poleid}"
        if project_name:
            main_title += f" ({project_name})"
        if category:
            main_title += f" [{category}]"
        if break_info:
            main_title += f" | 파단 높이: {break_info.get('breakheight')}, 각도: {break_info.get('breakdegree')}"
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_to_file:
            # 저장 디렉토리 생성
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "pole_plot_list")
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성
            filename_parts = [poleid]
            if project_name:
                filename_parts.append(project_name)
            if category:
                filename_parts.append(category)
            filename = "_".join(filename_parts) + ".png"
            
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n그래프 저장 완료: {output_path}")
            plt.close()
        else:
            plt.show(block=True)

# ============================================================================
# 설정: 여기에 전주 데이터 폴더 경로를 입력하세요
# ============================================================================
POLE_DIR_PATH = "raw_pole_data/break/강릉지사-2506/8732F191"
# 예시:
# POLE_DIR_PATH = "raw_pole_data/break/강릉지사-2506/8732F191"
# POLE_DIR_PATH = "raw_pole_data/normal/아산지사2-2511/1234A567"
# POLE_DIR_PATH = "/Users/heegulee/Desktop/SMARTCS/make_ai/raw_pole_data/break/강릉지사-2506/8732F191"
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 명령줄 인자로 경로 제공 (우선순위 높음)
        pole_dir_path = sys.argv[1]
    else:
        # 코드 내부에서 설정한 경로 사용
        pole_dir_path = POLE_DIR_PATH
    
    if not pole_dir_path:
        print("경로가 설정되지 않았습니다.")
        print("코드 상단의 POLE_DIR_PATH 변수를 설정하거나 명령줄 인자로 경로를 제공하세요.")
        sys.exit(1)
    
    # 통합 그래프 모드 (x, y, z를 하나의 그림에)
    print("\n통합 그래프 모드로 생성합니다...")
    plot_all_data_combined(pole_dir_path, save_to_file=True)
    
    # 개별 그래프 모드 (각 파일별로)
    # plot_pole_data(pole_dir_path)

