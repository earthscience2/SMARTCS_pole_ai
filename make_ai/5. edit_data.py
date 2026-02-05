#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""이미지를 보면서 ROI(Region of Interest) 영역을 설정하는 GUI 프로그램"""

import os
import json
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, messagebox

# 한글 폰트 설정 (Windows)
import platform
if platform.system() == 'Windows':
    try:
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    except:
        try:
            matplotlib.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕
        except:
            pass  # 폰트 설정 실패 시 기본 폰트 사용
elif platform.system() == 'Darwin':  # macOS
    matplotlib.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    matplotlib.rcParams['font.family'] = 'NanumGothic'

matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 현재 스크립트 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))

# 변경된 사항(저장 데이터) 저장 경로: make_ai\5. edit_data
EDIT_DATA_SAVE_DIR = Path(current_dir) / "5. edit_data"

# plot_processed_csv_2d 모듈 임포트 (같은 디렉토리)
sys.path.insert(0, current_dir)
try:
    from plot_processed_csv_2d import plot_csv_2d as regenerate_image
except ImportError:
    regenerate_image = None
    print("경고: plot_processed_csv_2d 모듈을 찾을 수 없습니다. 이미지 재생성 기능이 비활성화됩니다.")


def load_break_info(break_info_path: str) -> dict:
    """break_info.json 파일 읽기"""
    if not os.path.exists(break_info_path):
        return {}
    
    try:
        with open(break_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"오류: break_info.json 읽기 실패 - {e}")
        return {}


def save_break_info(break_info_path: str, data: dict):
    """break_info.json 파일 저장"""
    try:
        with open(break_info_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"오류: break_info.json 저장 실패 - {e}")
        return False


def roi_info_has_saved_roi(info: dict) -> bool:
    """ROI 정보가 하나라도 있으면 True. 동시 모드(roi_degree_min/roi_regions) 또는 개별 모드(roi_0/1/2_degree_min/roi_0/1/2_regions 중 하나)"""
    return (
        info.get('roi_degree_min') is not None
        or (info.get('roi_regions') and len(info.get('roi_regions')) > 0)
        or info.get('roi_0_degree_min') is not None
        or info.get('roi_1_degree_min') is not None
        or info.get('roi_2_degree_min') is not None
        or (info.get('roi_0_regions') and len(info.get('roi_0_regions')) > 0)
        or (info.get('roi_1_regions') and len(info.get('roi_1_regions')) > 0)
        or (info.get('roi_2_regions') and len(info.get('roi_2_regions')) > 0)
    )


class ROIEditorGUI:
    """ROI 영역 설정 GUI"""
    
    def __init__(self, root: tk.Tk, pole_dir: Path, all_pole_dirs: List[Path] = None, current_pole_idx: int = 0, initial_image_idx: Optional[int] = None):
        self.root = root
        self.pole_dir = pole_dir
        self.poleid = pole_dir.name
        
        # 전주 목록 관리 (전주 간 이동용)
        self.all_pole_dirs = all_pole_dirs if all_pole_dirs else [pole_dir]
        self.current_pole_idx = current_pole_idx
        self._initial_image_idx = initial_image_idx
        
        # 프로젝트별 전주 목록 구성
        self.project_pole_map = {}
        for pole_dir_path in self.all_pole_dirs:
            project_name = pole_dir_path.parent.name
            if project_name not in self.project_pole_map:
                self.project_pole_map[project_name] = []
            self.project_pole_map[project_name].append(pole_dir_path)
        
        # 5. edit_data 디렉토리 경로 설정 (make_ai\5. edit_data)
        project_name = pole_dir.parent.name
        crop_data_dir = EDIT_DATA_SAVE_DIR / "break"
        self.crop_data_pole_dir = crop_data_dir / project_name / self.poleid
        os.makedirs(self.crop_data_pole_dir, exist_ok=True)
        
        # 이미지 파일 목록 (확정되지 않은 이미지만)
        all_images = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
        self.break_info_files = {}
        self.image_files = []
        # 모든 이미지 목록 (확정된 것 포함, 목록 표시용)
        self.all_image_list = []
        for img_file in all_images:
            # 이미지 파일명: {poleid}_{measno}_OUT_processed_2d_plot.png
            # 파단 정보 파일명: {poleid}_{measno}_OUT_processed_break_info.json
            break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
            
            # 삭제 여부는 5. edit_data에서 확인 (confirmed 미사용)
            is_deleted = False
            crop_data_roi_info_file = None
            if self.crop_data_pole_dir.exists():
                crop_data_roi_info_file = self.crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
                if crop_data_roi_info_file.exists():
                    info = load_break_info(str(crop_data_roi_info_file))
                    if info.get('deleted') is True:
                        is_deleted = True
            
            # 4. merge_data의 break_info 파일도 유지 (작업용)
            if break_info_file.exists():
                self.break_info_files[img_file] = break_info_file
            
            # 모든 이미지 목록에 추가
            self.all_image_list.append({
                'file': img_file,
                'break_info_file': break_info_file if break_info_file.exists() else None
            })
            # 모든 이미지를 self.image_files에 추가 (삭제된 이미지 포함, 다음/이전 이동 시 모두 표시)
            if break_info_file.exists():
                self.break_info_files[img_file] = break_info_file
            self.image_files.append(img_file)
        self.current_image_idx = 0
        if self._initial_image_idx is not None and self.image_files:
            self.current_image_idx = min(max(0, self._initial_image_idx), len(self.image_files) - 1)
        
        # 이미지 파일이 없으면 생성
        if not self.image_files:
            self.generate_image()
            # generate_image 함수 내부에서 이미 확정되지 않은 이미지만 필터링하여 목록을 다시 불러옴
            # generate_image 후에도 이미지가 없으면, 모든 이미지가 확정된 경우일 수 있음
            if not self.image_files:
                # 확정되지 않은 이미지가 없는 경우 확인
                all_images_check = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
                # 이미지 파일이 있지만 모두 확정된 경우 또는 이미지 파일이 없는 경우 모두 조용히 처리
        
        # 현재 ROI 정보 파일 경로 (첫 번째 이미지 기준)
        self.current_roi_info_file = None
        if self.image_files:
            img_file = self.image_files[0]
            self.current_roi_info_file = self.crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
        
        # ROI 정보 읽기 (첫 번째 이미지 기준)
        if self.current_roi_info_file and self.current_roi_info_file.exists():
            self.roi_info = load_break_info(str(self.current_roi_info_file))
        else:
            self.roi_info = {}
        
        # 현재 ROI 영역 정보 (degree와 height 범위) - 여러 개 지원
        # 동시 모드: ROI 영역 리스트 (각 영역은 dict)
        self.roi_regions = []  # [{'degree_min': ..., 'degree_max': ..., 'height_min': ..., 'height_max': ...}, ...]
        
        # 개별 모드용: X(0), Y(1), Z(2) 축별 ROI 영역 리스트
        self.roi_subplots = [[], [], []]  # 각 축마다 여러 영역을 리스트로 관리
        self.current_editing_subplot_idx = 0
        
        # 현재 드래그 중인 영역 임시 저장
        self.temp_roi_region = None
        
        # 기존 단일 영역 형식과의 호환성 유지 (불러오기용)
        if self.roi_info.get('roi_degree_min') is not None:
            self.roi_regions = [{
                'degree_min': self.roi_info.get('roi_degree_min'),
                'degree_max': self.roi_info.get('roi_degree_max'),
                'height_min': self.roi_info.get('roi_height_min'),
                'height_max': self.roi_info.get('roi_height_max')
            }]
        elif self.roi_info.get('roi_regions'):
            self.roi_regions = self.roi_info.get('roi_regions', [])
        
        # 개별 모드 기존 데이터 불러오기
        for i in range(3):
            if self.roi_info.get(f'roi_{i}_degree_min') is not None:
                self.roi_subplots[i] = [{
                    'degree_min': self.roi_info.get(f'roi_{i}_degree_min'),
                    'degree_max': self.roi_info.get(f'roi_{i}_degree_max'),
                    'height_min': self.roi_info.get(f'roi_{i}_height_min'),
                    'height_max': self.roi_info.get(f'roi_{i}_height_max')
                }]
            elif self.roi_info.get(f'roi_{i}_regions'):
                self.roi_subplots[i] = self.roi_info.get(f'roi_{i}_regions', [])
        
        # 현재 CSV 데이터 범위 저장 (마우스 드래그 좌표 변환용)
        self.current_csv_data = None  # (heights, degrees) 튜플 저장
        
        # ROI 사각형 표시용 (여러 개의 사각형을 저장)
        self.roi_rectangles = []  # 리스트로 변경하여 모든 사각형 추적
        self.roi_region_outline_patches = []  # ROI 가능 영역 테두리 (X,Y,Z 세 칸)
        self.roi_delete_buttons = []  # 각 영역의 삭제 버튼 (텍스트 객체)
        self.rectangle_selector = None
        # 저장 형식: 'sync' | 'individual' 중 가장 최근에 저장한 방식 유지
        self._last_save_mode = None
        
        # GUI 초기화
        self.setup_gui()
        
        # 첫 번째 이미지 로드 (이미지가 있으면 무조건 로드, initial_image_idx 있으면 해당 인덱스부터)
        if self.image_files:
            self.load_image(self.current_image_idx)
        elif all_images:
            # 이미지 파일은 있지만 모두 삭제된 경우
            self.ax.clear()
            self.ax.text(0.5, 0.5, '이 전주의 모든 이미지가 삭제되었습니다.', 
                        ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
            self.canvas.draw()
        else:
            # 이미지 파일 자체가 없는 경우 - 다음 전주로 자동 이동
            if len(self.all_pole_dirs) > 1:
                # 다음 전주가 있으면 자동으로 이동
                if self.current_pole_idx < len(self.all_pole_dirs) - 1:
                    self.ax.clear()
                    self.ax.text(0.5, 0.5, f'이 전주에 이미지 파일이 없습니다.\n다음 전주로 이동합니다...', 
                                ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                    self.canvas.draw()
                    # 다음 전주로 자동 이동
                    self.current_pole_idx += 1
                    self.load_pole(self.all_pole_dirs[self.current_pole_idx])
                    return
                else:
                    # 마지막 전주인 경우
                    self.ax.clear()
                    self.ax.text(0.5, 0.5, '이 전주에 이미지 파일이 없습니다.\n(마지막 전주입니다.)', 
                                ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                    self.canvas.draw()
            else:
                # 단일 전주 모드인 경우
                self.ax.clear()
                self.ax.text(0.5, 0.5, '이 전주에 이미지 파일이 없습니다.', 
                            ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                self.canvas.draw()
    
    def generate_image(self):
        """CSV 파일에서 이미지 생성"""
        csv_files = sorted(list(self.pole_dir.glob("*_OUT_processed.csv")))
        if csv_files:
            for csv_file in csv_files:
                try:
                    # CSV 파일명: {poleid}_{measno}_OUT_processed.csv
                    # 파단 정보 파일명: {poleid}_{measno}_OUT_processed_break_info.json
                    break_info_file = csv_file.parent / csv_file.name.replace(".csv", "_break_info.json")
                    
                    # 파단 정보 파일이 없으면 생성
                    if not break_info_file.exists():
                        # 기본 파단 정보 생성
                        poleid = self.poleid
                        project_name = self.pole_dir.parent.name
                        default_info = {
                            'poleid': poleid,
                            'project_name': project_name,
                            'breakstate': 'B',
                            'breakheight': 0.0,
                            'breakdegree': 0.0
                        }
                        save_break_info(str(break_info_file), default_info)
                    
                    if regenerate_image is not None:
                        regenerate_image(
                            str(csv_file),
                            None,
                            str(break_info_file)
                        )
                except Exception as e:
                    print(f"  경고: 이미지 생성 중 오류 발생 - {e}")
            # 이미지 목록 다시 불러오기 (삭제된 이미지 포함하여 모두 표시)
            all_images = sorted(list(self.pole_dir.glob("*_OUT_processed_2d_plot.png")))
            self.break_info_files = {}
            self.image_files = []
            for img_file in all_images:
                break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
                if break_info_file.exists():
                    self.break_info_files[img_file] = break_info_file
                self.image_files.append(img_file)
            
            # current_image_idx가 범위를 벗어나면 조정
            if self.image_files and self.current_image_idx >= len(self.image_files):
                self.current_image_idx = 0
    
    def setup_gui(self):
        """GUI 구성 요소 설정"""
        self.root.title(f"ROI 영역 설정 - {self.poleid}")
        # ROI 적용 모드: 'sync'=동시(전주 전체), 'individual'=개별(현재 이미지만)
        self.roi_apply_mode = tk.StringVar(value='individual')
        self.root.geometry("1600x900")
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 정보 프레임 (전체 너비)
        self.info_frame = ttk.LabelFrame(main_frame, text="전주 정보", padding="10")
        self.info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.poleid_label = ttk.Label(self.info_frame, text=f"전주ID: {self.poleid}", font=("Arial", 12, "bold"))
        self.poleid_label.grid(row=0, column=0, sticky=tk.W)
        
        self.image_count_label = ttk.Label(self.info_frame, text=f"이미지 수: {len(self.image_files)}개", font=("Arial", 10))
        self.image_count_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        if len(self.all_pole_dirs) > 1:
            self.pole_count_label = ttk.Label(self.info_frame, text=f"전주: {self.current_pole_idx + 1}/{len(self.all_pole_dirs)}", font=("Arial", 10))
            self.pole_count_label.grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # 전체 대비 저장 통계 (상단 두 번째 행)
        self.stats_label = ttk.Label(self.info_frame, text="", font=("Arial", 10))
        self.stats_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))
        self.update_global_stats_label()
        
        # 이미지 프레임
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 삭제 버튼 클릭 이벤트 연결
        self.canvas.mpl_connect('pick_event', self.on_delete_button_click)
        
        # ROI 영역 선택을 위한 RectangleSelector (이미지 로드 후 설정)
        self.rectangle_selector = None
        
        # 컨트롤 프레임
        control_frame = ttk.LabelFrame(main_frame, text="ROI 영역 정보", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 개별 영역 정보 (축별 X, Y, Z)
        axis_names = ['X', 'Y', 'Z']
        self._axis_degree_min_vars = [tk.StringVar(value="-") for _ in range(3)]
        self._axis_degree_max_vars = [tk.StringVar(value="-") for _ in range(3)]
        self._axis_height_min_vars = [tk.StringVar(value="-") for _ in range(3)]
        self._axis_height_max_vars = [tk.StringVar(value="-") for _ in range(3)]
        for i in range(3):
            ttk.Label(control_frame, text=f"{axis_names[i]}축", width=4).grid(row=i, column=0, sticky=tk.W, padx=(0, 8))
            ttk.Label(control_frame, text="각도(°):", width=6).grid(row=i, column=1, sticky=tk.W, padx=(0, 2))
            ttk.Label(control_frame, textvariable=self._axis_degree_min_vars[i], width=8).grid(row=i, column=2, sticky=tk.W, padx=(0, 2))
            ttk.Label(control_frame, text="~").grid(row=i, column=3, sticky=tk.W, padx=0)
            ttk.Label(control_frame, textvariable=self._axis_degree_max_vars[i], width=8).grid(row=i, column=4, sticky=tk.W, padx=(0, 12))
            ttk.Label(control_frame, text="높이(m):", width=7).grid(row=i, column=5, sticky=tk.W, padx=(0, 2))
            ttk.Label(control_frame, textvariable=self._axis_height_min_vars[i], width=10).grid(row=i, column=6, sticky=tk.W, padx=(0, 2))
            ttk.Label(control_frame, text="~").grid(row=i, column=7, sticky=tk.W, padx=0)
            ttk.Label(control_frame, textvariable=self._axis_height_max_vars[i], width=10).grid(row=i, column=8, sticky=tk.W, padx=(0, 5))
        
        # ROI 적용 모드 선택 (동시 모드 = X,Y,Z 같은 영역 / 개별 모드 = X,Y,Z 각각 영역)
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=3, column=0, columnspan=9, sticky=tk.W, pady=(8, 0))
        ttk.Label(mode_frame, text="ROI 적용 모드:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(mode_frame, text="동시 모드 (X,Y,Z 같은 영역)", variable=self.roi_apply_mode, value='sync', command=self._on_roi_mode_changed).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(mode_frame, text="개별 모드 (X,Y,Z 각각 영역 설정)", variable=self.roi_apply_mode, value='individual', command=self._on_roi_mode_changed).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(mode_frame, text="  축:").pack(side=tk.LEFT, padx=(8, 4))
        self.subplot_btn_frame = ttk.Frame(mode_frame)
        self.subplot_btn_frame.pack(side=tk.LEFT)
        self.btn_x = ttk.Button(self.subplot_btn_frame, text="X", width=3, command=lambda: self._select_subplot(0))
        self.btn_x.pack(side=tk.LEFT, padx=2)
        self.btn_y = ttk.Button(self.subplot_btn_frame, text="Y", width=3, command=lambda: self._select_subplot(1))
        self.btn_y.pack(side=tk.LEFT, padx=2)
        self.btn_z = ttk.Button(self.subplot_btn_frame, text="Z", width=3, command=lambda: self._select_subplot(2))
        self.btn_z.pack(side=tk.LEFT, padx=2)
        # 초기 모드가 개별(individual)이면 X,Y,Z 활성화, 동시(sync)이면 비활성화
        for _b in (self.btn_x, self.btn_y, self.btn_z):
            _b.config(state=tk.NORMAL if self.roi_apply_mode.get() == 'individual' else tk.DISABLED)
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 이미지 이동 버튼 (전주까지 고려)
        has_prev = self.has_previous_image()
        has_next = self.has_next_image()
        
        self.prev_btn = ttk.Button(button_frame, text="◀ 이전 이미지", command=self.prev_image, state=tk.NORMAL if has_prev else tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.next_btn = ttk.Button(button_frame, text="다음 이미지 ▶", command=self.next_image, state=tk.NORMAL if has_next else tk.DISABLED)
        self.next_btn.grid(row=0, column=1, padx=5)
        
        action_button_start_col = 2
        
        # 영역 초기화 버튼 (모든 ROI 영역 취소)
        self.reset_roi_btn = ttk.Button(button_frame, text="🔄 전체 초기화", command=self.reset_roi_area)
        self.reset_roi_btn.grid(row=0, column=action_button_start_col, padx=5)
        
        # 삭제/되돌리기 버튼 (삭제된 이미지일 때는 되돌리기로 전환)
        self.delete_btn = ttk.Button(button_frame, text="🗑️ 삭제", command=self.delete_pole, style="Danger.TButton")
        self.delete_btn.grid(row=0, column=action_button_start_col + 1, padx=5)
        
        # 닫기 버튼
        ttk.Button(button_frame, text="✖ 닫기", command=self.close_window).grid(row=0, column=action_button_start_col + 2, padx=5)
        
        # 오른쪽 이미지 목록 프레임
        list_frame = ttk.LabelFrame(main_frame, text="이미지 목록", padding="10")
        list_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # 이미지 목록 표 (Treeview)
        list_tree_frame = ttk.Frame(list_frame)
        list_tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_tree_frame.columnconfigure(0, weight=1)
        list_tree_frame.rowconfigure(0, weight=1)
        
        # 스크롤바
        list_scrollbar = ttk.Scrollbar(list_tree_frame)
        list_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Treeview (이미지 목록 표)
        self.image_list_tree = ttk.Treeview(list_tree_frame, columns=("상태", "각도범위", "높이범위"), show="tree headings", height=30, yscrollcommand=list_scrollbar.set)
        self.image_list_tree.heading("#0", text="파일명")
        self.image_list_tree.heading("상태", text="상태")
        self.image_list_tree.heading("각도범위", text="각도 범위(°)")
        self.image_list_tree.heading("높이범위", text="높이 범위(m)")
        self.image_list_tree.column("#0", width=200)
        self.image_list_tree.column("상태", width=60)
        self.image_list_tree.column("각도범위", width=120)
        self.image_list_tree.column("높이범위", width=120)
        self.image_list_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_scrollbar.config(command=self.image_list_tree.yview)
        
        # 목록 클릭 이벤트
        self.image_list_tree.bind('<ButtonRelease-1>', self.on_list_select)
        self.image_list_tree.bind('<Double-Button-1>', self.on_list_double_click)
        
        # 목록 업데이트
        self.update_image_list()
        
        # 상태 변수
        self.saved = False
        self.deleted = False
        
        # 키보드 단축키 바인딩
        self.root.bind('<Key>', self.on_key_press)
        # 포커스를 받을 수 있도록 설정
        self.root.focus_set()
        # 축별 ROI 개별 영역 정보 초기 표시
        self._refresh_subplot_display()
    
    def has_previous_image(self) -> bool:
        """이전 이미지가 있는지 확인 (이전 전주 포함)"""
        # 현재 전주의 이전 이미지가 있으면 True
        if self.current_image_idx > 0:
            return True
        # 현재 전주가 첫 번째 이미지인 경우, 이전 전주가 있는지 확인
        if self.current_pole_idx > 0:
            return True
        return False
    
    def has_next_image(self) -> bool:
        """다음 이미지가 있는지 확인 (다음 전주 포함)"""
        # 현재 전주의 다음 이미지가 있으면 True
        if self.current_image_idx < len(self.image_files) - 1:
            return True
        # 현재 전주가 마지막 이미지인 경우, 다음 전주가 있는지 확인
        if self.current_pole_idx < len(self.all_pole_dirs) - 1:
            return True
        return False
    
    def get_global_saved_stats(self):
        """
        전체 전주 대상으로 전체 파일 수·저장 파일 수·미저장 파일 수·삭제 파일 수, 전체 프로젝트 수·저장 완료 프로젝트 수 계산.
        저장 = roi 정보(roi_degree_min 또는 roi_0/1/2_degree_min 중 하나)가 있는 경우.
        """
        total_files = 0
        saved_files = 0
        unsaved_files = 0
        deleted_files = 0
        project_pole_map = {}
        for pole_dir in self.all_pole_dirs:
            project_name = pole_dir.parent.name
            if project_name not in project_pole_map:
                project_pole_map[project_name] = []
            project_pole_map[project_name].append(pole_dir)
        
        project_saved = {}
        for project_name, pole_dirs in project_pole_map.items():
            project_saved[project_name] = True
            for pole_dir in pole_dirs:
                crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / pole_dir.name
                all_images = list(pole_dir.glob("*_OUT_processed_2d_plot.png"))
                for img_file in all_images:
                    total_files += 1
                    roi_file = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
                    if roi_file.exists():
                        try:
                            info = load_break_info(str(roi_file))
                            is_deleted = info.get('deleted', False)
                            if is_deleted:
                                deleted_files += 1
                            elif roi_info_has_saved_roi(info):
                                saved_files += 1
                            else:
                                unsaved_files += 1
                                project_saved[project_name] = False
                        except Exception:
                            unsaved_files += 1
                            project_saved[project_name] = False
                    else:
                        unsaved_files += 1
                        project_saved[project_name] = False
        
        total_projects = len(project_pole_map)
        saved_projects = sum(1 for c in project_saved.values() if c)
        return total_files, saved_files, unsaved_files, deleted_files, total_projects, saved_projects
    
    def update_global_stats_label(self):
        """상단 '전체 대비 저장/미저장/삭제' 통계 라벨 갱신."""
        if not hasattr(self, 'stats_label'):
            return
        try:
            total_files, saved_files, unsaved_files, deleted_files, total_projects, saved_projects = self.get_global_saved_stats()
            self.stats_label.config(
                text=f"전체 파일: {total_files}개 (저장: {saved_files}, 미저장: {unsaved_files}, 삭제: {deleted_files})  |  전체 프로젝트: {saved_projects}/{total_projects} 저장 완료"
            )
        except Exception:
            self.stats_label.config(text="")
    
    def update_image_list(self):
        """이미지 목록 표 업데이트 (프로젝트 > 전주 > 이미지 계층 구조)"""
        if not hasattr(self, 'image_list_tree'):
            return
        
        # 기존 항목 삭제
        for item in self.image_list_tree.get_children():
            self.image_list_tree.delete(item)
        
        # 프로젝트별로 그룹화하여 표시
        project_items = {}  # 프로젝트 아이템 ID 저장
        pole_items = {}  # 전주 아이템 ID 저장 (key: (project_name, poleid))
        
        # 프로젝트명으로 정렬
        sorted_projects = sorted(self.project_pole_map.keys())
        
        # 현재 프로젝트와 전주 확인
        current_project = self.pole_dir.parent.name
        current_poleid = self.poleid
        
        for project_name in sorted_projects:
            # 해당 프로젝트의 전주 목록
            pole_dirs = sorted(self.project_pole_map[project_name], key=lambda x: x.name)
            
            # 프로젝트 내 모든 전주가 저장 완료인지 확인 (한 번만)
            project_all_saved = True
            pole_saved_map = {}  # 전주별 저장 완료 여부 캐싱
            
            for pole_dir_path in pole_dirs:
                poleid = pole_dir_path.name
                all_images = sorted(list(pole_dir_path.glob("*_OUT_processed_2d_plot.png")))
                
                if not all_images:
                    pole_saved_map[poleid] = False
                    project_all_saved = False
                    continue
                
                # 전주의 모든 이미지가 저장되었는지 확인 (ROI 정보 유무)
                pole_all_saved = True
                crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / poleid
                
                for img_file in all_images:
                    crop_data_roi_info_file = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
                    if crop_data_roi_info_file.exists():
                        try:
                            info = load_break_info(str(crop_data_roi_info_file))
                            is_deleted = info.get('deleted', False)
                            if not is_deleted and not roi_info_has_saved_roi(info):
                                pole_all_saved = False
                                break
                        except:
                            pole_all_saved = False
                            break
                    else:
                        pole_all_saved = False
                        break
                
                pole_saved_map[poleid] = pole_all_saved
                if not pole_all_saved:
                    project_all_saved = False
            
            # 프로젝트 아이템 추가 (저장 완료 여부 표시)
            project_text = f"📁 {project_name}"
            if project_all_saved and len(pole_dirs) > 0:
                project_text += " ✅"
            project_id = self.image_list_tree.insert("", "end", text=project_text, values=("프로젝트", "", ""))
            project_items[project_name] = project_id
            
            # 현재 프로젝트인지 확인
            if project_name == current_project:
                # 현재 프로젝트는 확장
                self.image_list_tree.item(project_id, open=True)
            
            for pole_dir_path in pole_dirs:
                poleid = pole_dir_path.name
                
                # 해당 전주의 이미지 목록
                all_images = sorted(list(pole_dir_path.glob("*_OUT_processed_2d_plot.png")))
                
                # 캐시된 저장 완료 여부 사용
                pole_all_saved = pole_saved_map.get(poleid, False)
                
                # 전주 아이템 추가 (저장 완료 여부 표시)
                pole_text = f"  📂 {poleid}"
                if pole_all_saved and all_images:
                    pole_text += " ✅"
                pole_id = self.image_list_tree.insert(project_id, "end", text=pole_text, values=("전주", "", ""))
                pole_items[(project_name, poleid)] = pole_id
                
                # 현재 전주인지 확인
                is_current_pole = (pole_dir_path == self.pole_dir)
                if is_current_pole:
                    # 현재 전주는 확장
                    self.image_list_tree.item(pole_id, open=True)
                
                # 현재 전주인 경우에만 이미지 목록 표시 (성능 최적화: 확장 시 동적 로드)
                if is_current_pole:
                    crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / poleid
                    
                    for img_file in all_images:
                        # 삭제 여부·저장 여부는 5. edit_data에서 확인 (ROI 정보 유무로 저장 판단)
                        is_deleted = False
                        is_saved = False
                        roi_degree_range = ""
                        roi_height_range = ""
                        
                        if crop_data_pole_dir.exists():
                            crop_data_roi_info_file = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
                            if crop_data_roi_info_file.exists():
                                try:
                                    info = load_break_info(str(crop_data_roi_info_file))
                                    is_deleted = info.get('deleted', False)
                                    is_saved = roi_info_has_saved_roi(info)
                                    
                                    # 여러 영역 지원: roi_regions 또는 roi_0/1/2_regions
                                    if info.get('roi_regions'):
                                        regions = info.get('roi_regions')
                                        if len(regions) == 1:
                                            roi_degree_range = f"{regions[0]['degree_min']:.1f}~{regions[0]['degree_max']:.1f}°"
                                            roi_height_range = f"{regions[0]['height_min']:.3f}~{regions[0]['height_max']:.3f}m"
                                        else:
                                            roi_degree_range = f"{len(regions)}개 영역"
                                            roi_height_range = ""
                                    elif info.get('roi_0_regions') or info.get('roi_1_regions') or info.get('roi_2_regions'):
                                        # 개별 모드의 여러 영역
                                        total_regions = sum([
                                            len(info.get('roi_0_regions', [])),
                                            len(info.get('roi_1_regions', [])),
                                            len(info.get('roi_2_regions', []))
                                        ])
                                        roi_degree_range = f"개별 {total_regions}개"
                                        roi_height_range = ""
                                    else:
                                        # 기존 단일 영역 형식
                                        roi_degree_min = info.get('roi_degree_min') or info.get('roi_0_degree_min') or info.get('roi_1_degree_min') or info.get('roi_2_degree_min')
                                        roi_degree_max = info.get('roi_degree_max') or info.get('roi_0_degree_max') or info.get('roi_1_degree_max') or info.get('roi_2_degree_max')
                                        roi_height_min = info.get('roi_height_min') or info.get('roi_0_height_min') or info.get('roi_1_height_min') or info.get('roi_2_height_min')
                                        roi_height_max = info.get('roi_height_max') or info.get('roi_0_height_max') or info.get('roi_1_height_max') or info.get('roi_2_height_max')
                                        if roi_degree_min is not None and roi_degree_max is not None:
                                            roi_degree_range = f"{roi_degree_min:.1f}~{roi_degree_max:.1f}°"
                                        if roi_height_min is not None and roi_height_max is not None:
                                            roi_height_range = f"{roi_height_min:.3f}~{roi_height_max:.3f}m"
                                except:
                                    pass
                        
                        # 상태 결정
                        if is_deleted:
                            status = "삭제됨"
                        elif is_saved:
                            status = "저장됨"
                        else:
                            status = "미저장"
                        
                        file_name = img_file.name
                        img_item_id = self.image_list_tree.insert(pole_id, "end", text=f"    📄 {file_name}", values=(status, roi_degree_range, roi_height_range))
                        
                        # 현재 선택된 이미지는 강조 표시
                        if self.image_files and self.current_image_idx < len(self.image_files):
                            current_img_file = self.image_files[self.current_image_idx]
                            if img_file == current_img_file and is_current_pole:
                                self.image_list_tree.selection_set(img_item_id)
                                self.image_list_tree.see(img_item_id)
        
        # 아이템 ID 저장 (선택 이벤트에서 사용)
        self.project_items = project_items
        self.pole_items = pole_items
    
    def _load_pole_images(self, pole_item):
        """전주 확장 시 이미지 목록 동적 로드"""
        # 이미 이미지가 로드되어 있는지 확인
        if self.image_list_tree.get_children(pole_item):
            return
        
        pole_text = self.image_list_tree.item(pole_item, "text")
        poleid = pole_text.replace("  📂 ", "").replace(" ✅", "").strip()
        
        # 프로젝트명 찾기
        parent_item = self.image_list_tree.parent(pole_item)
        if not parent_item:
            return
        
        project_text = self.image_list_tree.item(parent_item, "text")
        project_name = project_text.replace("📁 ", "").replace(" ✅", "").strip()
        
        # 전주 디렉토리 찾기
        target_pole_dir = None
        for pole_dir_path in self.all_pole_dirs:
            if pole_dir_path.name == poleid and pole_dir_path.parent.name == project_name:
                target_pole_dir = pole_dir_path
                break
        
        if not target_pole_dir:
            return
        
        all_images = sorted(list(target_pole_dir.glob("*_OUT_processed_2d_plot.png")))
        crop_data_dir = EDIT_DATA_SAVE_DIR / "break"
        crop_data_pole_dir = crop_data_dir / project_name / poleid
        is_current_pole = (target_pole_dir == self.pole_dir)
        
        for img_file in all_images:
            is_deleted = False
            is_saved = False
            roi_degree_range = ""
            roi_height_range = ""
            
            crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / poleid
            if crop_data_pole_dir.exists():
                crop_data_roi_info_file = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
                if crop_data_roi_info_file.exists():
                    try:
                        info = load_break_info(str(crop_data_roi_info_file))
                        is_deleted = info.get('deleted', False)
                        is_saved = roi_info_has_saved_roi(info)
                        
                        # 여러 영역 지원: roi_regions 또는 roi_0/1/2_regions
                        if info.get('roi_regions'):
                            regions = info.get('roi_regions')
                            if len(regions) == 1:
                                roi_degree_range = f"{regions[0]['degree_min']:.1f}~{regions[0]['degree_max']:.1f}°"
                                roi_height_range = f"{regions[0]['height_min']:.3f}~{regions[0]['height_max']:.3f}m"
                            else:
                                roi_degree_range = f"{len(regions)}개 영역"
                                roi_height_range = ""
                        elif info.get('roi_0_regions') or info.get('roi_1_regions') or info.get('roi_2_regions'):
                            # 개별 모드의 여러 영역
                            total_regions = sum([
                                len(info.get('roi_0_regions', [])),
                                len(info.get('roi_1_regions', [])),
                                len(info.get('roi_2_regions', []))
                            ])
                            roi_degree_range = f"개별 {total_regions}개"
                            roi_height_range = ""
                        else:
                            # 기존 단일 영역 형식
                            roi_degree_min = info.get('roi_degree_min') or info.get('roi_0_degree_min') or info.get('roi_1_degree_min') or info.get('roi_2_degree_min')
                            roi_degree_max = info.get('roi_degree_max') or info.get('roi_0_degree_max') or info.get('roi_1_degree_max') or info.get('roi_2_degree_max')
                            roi_height_min = info.get('roi_height_min') or info.get('roi_0_height_min') or info.get('roi_1_height_min') or info.get('roi_2_height_min')
                            roi_height_max = info.get('roi_height_max') or info.get('roi_0_height_max') or info.get('roi_1_height_max') or info.get('roi_2_height_max')
                            if roi_degree_min is not None and roi_degree_max is not None:
                                roi_degree_range = f"{roi_degree_min:.1f}~{roi_degree_max:.1f}°"
                            if roi_height_min is not None and roi_height_max is not None:
                                roi_height_range = f"{roi_height_min:.3f}~{roi_height_max:.3f}m"
                    except:
                        pass
            
            if is_deleted:
                status = "삭제됨"
            elif is_saved:
                status = "저장됨"
            else:
                status = "미저장"
            
            file_name = img_file.name
            img_item_id = self.image_list_tree.insert(pole_item, "end", text=f"    📄 {file_name}", values=(status, roi_degree_range, roi_height_range))
            
            if self.image_files and self.current_image_idx < len(self.image_files):
                current_img_file = self.image_files[self.current_image_idx]
                if img_file == current_img_file and is_current_pole:
                    self.image_list_tree.selection_set(img_item_id)
                    self.image_list_tree.see(img_item_id)
    
    def _load_project_images(self, project_item):
        """프로젝트 확장 시 전주 이미지 목록 동적 로드"""
        project_text = self.image_list_tree.item(project_item, "text")
        project_name = project_text.replace("📁 ", "").replace(" ✅", "").strip()
        
        # 해당 프로젝트의 전주 목록 가져오기
        if project_name not in self.project_pole_map:
            return
        
        pole_dirs = sorted(self.project_pole_map[project_name], key=lambda x: x.name)
        for pole_dir_path in pole_dirs:
            poleid = pole_dir_path.name
            pole_key = (project_name, poleid)
            
            # 이미 로드된 전주는 건너뛰기
            if pole_key not in self.pole_items:
                continue
            
            pole_id = self.pole_items[pole_key]
            
            # 이미 이미지가 로드되어 있거나 전주가 확장되지 않은 경우 건너뛰기
            if self.image_list_tree.get_children(pole_id) or not self.image_list_tree.item(pole_id, "open"):
                continue
            
            # 전주 이미지 로드
            self._load_pole_images(pole_id)
    
    def on_list_select(self, event):
        """목록에서 항목 선택 시 호출 (프로젝트/전주/이미지 모두 처리)"""
        selection = self.image_list_tree.selection()
        if not selection:
            return
        
        selected_item = selection[0]
        item_text = self.image_list_tree.item(selected_item, "text")
        values = self.image_list_tree.item(selected_item, "values")
        
        # 프로젝트 선택 시
        if values and values[0] == "프로젝트":
            # 프로젝트 확장/축소만 처리
            is_open = self.image_list_tree.item(selected_item, "open")
            if is_open:
                self.image_list_tree.item(selected_item, open=False)
            else:
                self.image_list_tree.item(selected_item, open=True)
                # 확장 시 해당 프로젝트의 전주 이미지 목록 동적 로드
                self._load_project_images(selected_item)
            return
        
        # 전주 선택 시
        if values and values[0] == "전주":
            # 전주 확장/축소 처리
            is_open = self.image_list_tree.item(selected_item, "open")
            if is_open:
                self.image_list_tree.item(selected_item, open=False)
            else:
                self.image_list_tree.item(selected_item, open=True)
                # 확장 시 이미지 목록 동적 로드
                self._load_pole_images(selected_item)
            
            # 전주명 추출 (📂, ✅ 제거)
            poleid = item_text.replace("  📂 ", "").replace(" ✅", "").strip()
            
            # 프로젝트명 찾기
            parent_item = self.image_list_tree.parent(selected_item)
            if parent_item:
                project_text = self.image_list_tree.item(parent_item, "text")
                project_name = project_text.replace("📁 ", "").replace(" ✅", "").strip()
                
                # 해당 전주로 이동
                target_pole_dir = None
                for pole_dir_path in self.all_pole_dirs:
                    if pole_dir_path.name == poleid and pole_dir_path.parent.name == project_name:
                        target_pole_dir = pole_dir_path
                        break
                
                if target_pole_dir:
                    # 저장 확인
                    if not self.check_save_before_switch():
                        return
                    
                    # 전주 인덱스 찾기
                    if target_pole_dir in self.all_pole_dirs:
                        self.current_pole_idx = self.all_pole_dirs.index(target_pole_dir)
                        self.load_pole(target_pole_dir)
            return
        
        # 이미지 선택 시
        # 파일명 추출 (📄 제거)
        file_name = item_text.replace("    📄 ", "").strip()
        
        # 부모 전주 찾기
        parent_item = self.image_list_tree.parent(selected_item)
        if not parent_item:
            return
        
        pole_text = self.image_list_tree.item(parent_item, "text")
        poleid = pole_text.replace("  📂 ", "").replace(" ✅", "").strip()
        
        # 프로젝트 찾기
        project_item = self.image_list_tree.parent(parent_item)
        if not project_item:
            return
        
        project_text = self.image_list_tree.item(project_item, "text")
        project_name = project_text.replace("📁 ", "").replace(" ✅", "").strip()
        
        # 해당 전주로 이동 (다른 전주인 경우)
        target_pole_dir = None
        for pole_dir_path in self.all_pole_dirs:
            if pole_dir_path.name == poleid and pole_dir_path.parent.name == project_name:
                target_pole_dir = pole_dir_path
                break
        
        if not target_pole_dir:
            return
        
        # 전주가 다르면 전주 먼저 로드
        if target_pole_dir != self.pole_dir:
            # 저장 확인
            if not self.check_save_before_switch():
                return
            
            # 전주 인덱스 찾기
            if target_pole_dir in self.all_pole_dirs:
                self.current_pole_idx = self.all_pole_dirs.index(target_pole_dir)
                self.load_pole(target_pole_dir)
        
        # 이미지 파일 찾기
        img_file = target_pole_dir / file_name
        if not img_file.exists():
            return
        
        # 같은 전주인데 선택한 이미지가 목록에 없으면 (예: 이번 세션에서 확정해서 제거된 경우) 전주 새로고침
        if target_pole_dir == self.pole_dir and img_file not in self.image_files:
            self.load_pole(target_pole_dir)
        
        # self.image_files에서 해당 이미지의 인덱스 찾기 (파일명 기준)
        idx = next((i for i, f in enumerate(self.image_files) if f.name == file_name), None)
        if idx is not None:
            self.load_image(idx)
    
    def on_list_double_click(self, event):
        """목록에서 항목 더블 클릭 시 호출 (on_list_select와 동일)"""
        self.on_list_select(event)
    
    def load_image(self, idx: int):
        """이미지 로드 및 표시"""
        if not self.image_files or idx < 0 or idx >= len(self.image_files):
            return
        
        self.current_image_idx = idx
        img_file = self.image_files[idx]
        
        # 현재 이미지에 대응하는 ROI 정보 파일 로드
        self.current_roi_info_file = self.crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
        
        # ROI 정보 로드 (여러 영역 지원)
        if self.current_roi_info_file.exists():
            self.roi_info = load_break_info(str(self.current_roi_info_file))
            self.temp_roi_region = None  # 임시 영역 초기화
            
            # 여러 영역 형식 또는 개별 모드 확인
            has_individual = ('roi_0_degree_min' in self.roi_info or 'roi_0_regions' in self.roi_info or
                            'roi_1_degree_min' in self.roi_info or 'roi_1_regions' in self.roi_info or
                            'roi_2_degree_min' in self.roi_info or 'roi_2_regions' in self.roi_info)
            
            if has_individual:
                # 개별 모드 데이터 로드
                self._last_save_mode = 'individual'
                self.roi_subplots = [[], [], []]
                
                for i in range(3):
                    # 여러 영역 형식
                    if self.roi_info.get(f'roi_{i}_regions'):
                        self.roi_subplots[i] = [dict(r) for r in self.roi_info.get(f'roi_{i}_regions')]
                    # 기존 단일 영역 형식 호환
                    elif self.roi_info.get(f'roi_{i}_degree_min') is not None:
                        self.roi_subplots[i] = [{
                            'degree_min': self.roi_info.get(f'roi_{i}_degree_min'),
                            'degree_max': self.roi_info.get(f'roi_{i}_degree_max'),
                            'height_min': self.roi_info.get(f'roi_{i}_height_min'),
                            'height_max': self.roi_info.get(f'roi_{i}_height_max')
                        }]
                    else:
                        self.roi_subplots[i] = []
                
                self.roi_regions = []
                self.roi_apply_mode.set('individual')
                self._on_roi_mode_changed()
            else:
                # 동시 모드 데이터 로드
                self._last_save_mode = 'sync'
                
                # 여러 영역 형식
                if self.roi_info.get('roi_regions'):
                    self.roi_regions = [dict(r) for r in self.roi_info.get('roi_regions')]
                # 기존 단일 영역 형식 호환
                elif self.roi_info.get('roi_degree_min') is not None:
                    self.roi_regions = [{
                        'degree_min': self.roi_info.get('roi_degree_min'),
                        'degree_max': self.roi_info.get('roi_degree_max'),
                        'height_min': self.roi_info.get('roi_height_min'),
                        'height_max': self.roi_info.get('roi_height_max')
                    }]
                else:
                    self.roi_regions = []
                
                self.roi_subplots = [[], [], []]
                self.roi_apply_mode.set('sync')
                self._on_roi_mode_changed()
        else:
            self.roi_info = {}
            self.roi_regions = []
            self.roi_subplots = [[], [], []]
            self.temp_roi_region = None
            self.roi_apply_mode.set('individual')  # 기본 모드: 개별
            if hasattr(self, '_on_roi_mode_changed'):
                self._on_roi_mode_changed()
        
        # 입력 필드 업데이트 (축별 개별 영역 정보)
        self._refresh_subplot_display()
        
        try:
            img = mpimg.imread(str(img_file))
            self.ax.clear()
            # 기존 ROI 사각형 모두 제거
            self.clear_roi_rectangles()
            
            # 기존 RectangleSelector 제거
            if self.rectangle_selector:
                try:
                    self.rectangle_selector.set_active(False)
                    self.rectangle_selector = None
                except:
                    pass
            
            self.ax.imshow(img)
            self.roi_region_outline_patches = []
            
            # CSV 파일 읽어서 데이터 범위 파악 (마우스 드래그 좌표 변환용)
            csv_filename = img_file.name.replace("_2d_plot.png", ".csv")
            csv_file = self.pole_dir / csv_filename
            if csv_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    if not df.empty and 'height' in df.columns and 'degree' in df.columns:
                        heights = sorted(df['height'].unique())
                        degrees = sorted(df['degree'].unique())
                        self.current_csv_data = {
                            'heights': heights,
                            'degrees': degrees,
                            'height_min': min(heights),
                            'height_max': max(heights),
                            'degree_min': min(degrees),
                            'degree_max': max(degrees),
                            'img_shape': img.shape  # (height, width, channels)
                        }
                    else:
                        self.current_csv_data = None
                except Exception:
                    self.current_csv_data = None
            else:
                self.current_csv_data = None
            
            # ROI 가능 영역 테두리 표시 (드래그가 유효한 구간을 사용자에게 보여줌)
            if self.current_csv_data:
                self._draw_roi_region_outline()
            
            # ROI 영역 사각형 표시 (여러 영역 지원)
            has_roi = self.current_csv_data and (
                self.roi_regions or any(self.roi_subplots)
            )
            if has_roi:
                self.draw_roi_rectangle()
            
            # RectangleSelector 추가 (드래그로 ROI 선택)
            if self.current_csv_data:
                # 이미지 픽셀 범위 계산
                img_height = self.current_csv_data['img_shape'][0]
                img_width = self.current_csv_data['img_shape'][1]
                
                # RectangleSelector의 선택 범위를 이미지 픽셀 범위로 제한
                # imshow는 기본적으로 x: -0.5 ~ (width-0.5), y: (height-0.5) ~ -0.5
                self.rectangle_selector = RectangleSelector(
                    self.ax, self.on_roi_select,
                    useblit=True, button=[1], minspanx=5, minspany=5,
                    spancoords='pixels', interactive=True,
                    props=dict(facecolor='none', edgecolor='cyan', alpha=0.3, linestyle='--', linewidth=2)
                )
            
            self.ax.axis('off')
            title_suffix = "\n[삭제됨]" if self.roi_info.get('deleted') else "\n(마우스로 드래그하여 ROI 영역 선택)"
            self.ax.set_title(f"{self.poleid} - 이미지 {idx + 1}/{len(self.image_files)}\n파일: {img_file.name}{title_suffix}", 
                            fontsize=12, fontweight='bold', pad=10)
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 이미지 목록 업데이트
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            
            # 이전/다음 버튼 상태 업데이트 (전주 포함)
            self.prev_btn.config(state=tk.NORMAL if self.has_previous_image() else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.has_next_image() else tk.DISABLED)
            
            # 저장 버튼 상태 갱신 (변경사항 없으면 비활성화)
            self.update_save_button()
            
            # 삭제/되돌리기 버튼 갱신 (현재 이미지가 삭제됨이면 되돌리기, 아니면 삭제)
            self._update_delete_restore_button()
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
    
    def update_break_marker(self, break_height, break_degree):
        """파단 위치 마커를 즉시 업데이트"""
        if not self.current_csv_data:
            return
        
        try:
            # 데이터 범위
            height_min = self.current_csv_data['height_min']
            height_max = self.current_csv_data['height_max']
            degree_min = self.current_csv_data['degree_min']
            degree_max = self.current_csv_data['degree_max']
            img_height = self.current_csv_data['img_shape'][0]
            img_width = self.current_csv_data['img_shape'][1]
            
            # 데이터 값을 이미지 좌표로 변환
            plot_y = (break_height - height_min) / (height_max - height_min) if (height_max - height_min) > 0 else 0.5
            plot_x_in_subplot = (break_degree - degree_min) / (degree_max - degree_min) if (degree_max - degree_min) > 0 else 0.5
            plot_y = max(0.0, min(1.0, plot_y))
            plot_x_in_subplot = max(0.0, min(1.0, plot_x_in_subplot))
            
            # 플롯 영역 비율 -> 정규화된 이미지 좌표 (0~1)
            plot_margin_top = 0.05
            plot_margin_bottom = 0.09
            subplot_margin_left = 0.09
            subplot_margin_right = 0.03
            
            norm_y = plot_margin_bottom + plot_y * (1 - plot_margin_top - plot_margin_bottom)
            
            # 각 서브플롯에 마커 표시 (3개 서브플롯 모두)
            marker_x_coords = []
            marker_y_coords = []
            for subplot_idx in range(3):
                # 서브플롯 내 정규화된 x 좌표
                local_x_in_subplot = subplot_margin_left + plot_x_in_subplot * (1 - subplot_margin_left - subplot_margin_right)
                # 전체 이미지에서의 정규화된 x 좌표
                norm_x = (subplot_idx + local_x_in_subplot) / 3.0
                
                # 픽셀 좌표로 변환
                pixel_x = (norm_x * img_width) - 0.5
                pixel_y = (img_height - 0.5) - (norm_y * img_height)
                
                marker_x_coords.append(pixel_x)
                marker_y_coords.append(pixel_y)
            
            # 기존 마커 제거 (모든 마커 라인 제거)
            if hasattr(self, 'break_marker_lines') and self.break_marker_lines:
                try:
                    if isinstance(self.break_marker_lines, list):
                        for line in self.break_marker_lines:
                            try:
                                line.remove()
                            except:
                                pass
                    else:
                        try:
                            self.break_marker_lines.remove()
                        except:
                            pass
                except:
                    pass
                self.break_marker_lines = None
            
            # 혹시 모를 중복 마커 제거 (라벨이 '파단 위치'인 모든 라인 제거)
            for line in list(self.ax.lines):
                try:
                    if line.get_label() == '파단 위치':
                        line.remove()
                except:
                    pass
            
            # 새 마커 추가 (빨간색 별 마커)
            # 3개 서브플롯에 각각 별을 표시 (하나의 plot 호출로 3개 점 표시)
            marker_line = self.ax.plot(marker_x_coords, marker_y_coords, '*', 
                                      color='red', markersize=20, markeredgewidth=2, 
                                      label='파단 위치', zorder=10, linestyle='None')
            # plot은 리스트를 반환하므로 첫 번째 요소만 저장 (하나의 Line2D 객체)
            self.break_marker_lines = marker_line[0] if marker_line and len(marker_line) > 0 else None
            
            # 캔버스 즉시 업데이트
            self.canvas.draw_idle()
        except Exception as e:
            # 마커 업데이트 실패해도 계속 진행
            pass
    
    def on_roi_select(self, eclick, erelease):
        """ROI 영역 선택 이벤트 핸들러 (RectangleSelector 콜백)"""
        if not self.current_csv_data:
            return
        
        # 선택된 영역의 픽셀 좌표
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        
        # 이미지 픽셀 범위 정보
        img_height = self.current_csv_data['img_shape'][0]
        img_width = self.current_csv_data['img_shape'][1]
        
        # 좌표 정규화 (min, max) - 이미지 전체 범위 허용
        # imshow 좌표계: x는 -0.5 ~ (width-0.5), y는 (height-0.5) ~ -0.5
        # 하지만 실제 표시 영역은 더 넓을 수 있으므로 여유를 둠
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        
        # 유효한 범위인지 확인 (최소 크기 체크)
        if abs(x_max - x_min) < 1.0 or abs(y_max - y_min) < 1.0:
            return
        
        # 픽셀 좌표를 데이터 좌표(degree, height)로 변환
        img_height = self.current_csv_data['img_shape'][0]
        img_width = self.current_csv_data['img_shape'][1]
        height_min = self.current_csv_data['height_min']
        height_max = self.current_csv_data['height_max']
        degree_min = self.current_csv_data['degree_min']
        degree_max = self.current_csv_data['degree_max']
        
        # 픽셀 좌표를 정규화 좌표로 변환
        # imshow 좌표계: x는 -0.5 ~ (width-0.5), y는 (height-0.5) ~ -0.5
        # 정규화 좌표: 0.0 ~ 1.0 (이미지 전체 범위)
        norm_x1 = (x_min + 0.5) / img_width if img_width > 0 else 0.0
        norm_x2 = (x_max + 0.5) / img_width if img_width > 0 else 1.0
        norm_y1 = (img_height - 0.5 - y_max) / img_height if img_height > 0 else 0.0
        norm_y2 = (img_height - 0.5 - y_min) / img_height if img_height > 0 else 1.0
        
        # 정규화 좌표를 0~1 범위로 제한 (오른쪽 영역까지 포함)
        norm_x1 = max(0.0, min(1.0, norm_x1))
        norm_x2 = max(0.0, min(1.0, norm_x2))
        norm_y1 = max(0.0, min(1.0, norm_y1))
        norm_y2 = max(0.0, min(1.0, norm_y2))
        
        # 이미지가 3개의 서브플롯으로 구성되어 있으므로, 중간 서브플롯(y축) 기준으로 변환
        # 각 서브플롯의 너비는 1/3
        subplot_width = 1.0 / 3.0
        
        # 각도 범위 계산 (x축은 중간 서브플롯 기준)
        # 중간 서브플롯의 x 범위: 1/3 ~ 2/3
        # 선택된 영역이 어느 서브플롯에 있는지 확인
        center_norm_x = (norm_x1 + norm_x2) / 2.0
        subplot_idx = int(center_norm_x / subplot_width)
        subplot_idx = max(0, min(2, subplot_idx))
        
        # 서브플롯 내에서의 상대 위치 계산
        # 컬러바가 없으므로 오른쪽 여백을 최소화
        subplot_margin_left = 0.09
        subplot_margin_right = 0.03
        
        # 전체 이미지 좌표를 서브플롯 좌표로 변환
        local_x1 = (norm_x1 - subplot_idx * subplot_width) / subplot_width
        local_x2 = (norm_x2 - subplot_idx * subplot_width) / subplot_width
        
        # 서브플롯 내 플롯 영역으로 변환 (오른쪽 영역까지 포함)
        plot_x1 = (local_x1 - subplot_margin_left) / (1 - subplot_margin_left - subplot_margin_right)
        plot_x2 = (local_x2 - subplot_margin_left) / (1 - subplot_margin_left - subplot_margin_right)
        # 오른쪽 영역까지 포함하도록 범위 확장 (더 넓은 여유 허용)
        plot_x1 = max(-0.2, min(1.2, plot_x1))  # 더 넓은 여유 허용
        plot_x2 = max(-0.2, min(1.2, plot_x2))
        # 최종적으로 0~1 범위로 제한
        plot_x1 = max(0.0, min(1.0, plot_x1))
        plot_x2 = max(0.0, min(1.0, plot_x2))
        
        # 높이 범위 계산 (y축)
        plot_margin_top = 0.05
        plot_margin_bottom = 0.09
        plot_y1 = (norm_y1 - plot_margin_bottom) / (1 - plot_margin_top - plot_margin_bottom)
        plot_y2 = (norm_y2 - plot_margin_bottom) / (1 - plot_margin_top - plot_margin_bottom)
        plot_y1 = max(0.0, min(1.0, plot_y1))
        plot_y2 = max(0.0, min(1.0, plot_y2))
        
        # 데이터 값으로 변환
        degree1 = degree_min + plot_x1 * (degree_max - degree_min)
        degree2 = degree_min + plot_x2 * (degree_max - degree_min)
        height1 = height_min + plot_y1 * (height_max - height_min)
        height2 = height_min + plot_y2 * (height_max - height_min)
        dm, dM = min(degree1, degree2), max(degree1, degree2)
        hm, hM = min(height1, height2), max(height1, height2)
        
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        
        # 드래그한 영역을 바로 추가
        new_region = {
            'degree_min': dm,
            'degree_max': dM,
            'height_min': hm,
            'height_max': hM
        }
        
        if mode == 'individual':
            # 개별 모드: 해당 축에만 추가
            self.roi_subplots[subplot_idx].append(new_region)
        else:
            # 동시 모드: 공통 리스트에 추가
            self.roi_regions.append(new_region)
        
        # 임시 영역 초기화
        self.temp_roi_region = None
        
        # 화면 업데이트
        self._refresh_subplot_display()
        self.clear_roi_rectangles()
        self.draw_roi_rectangle()
        self.canvas.draw_idle()
        
        # 자동 저장
        self.auto_save_roi()
    
    def clear_roi_rectangles(self):
        """모든 ROI 사각형 및 삭제 버튼 제거"""
        if hasattr(self, 'roi_rectangles') and self.roi_rectangles:
            for rect in self.roi_rectangles:
                try:
                    rect.remove()
                except:
                    pass
            self.roi_rectangles.clear()
        
        # 삭제 버튼 제거
        if hasattr(self, 'roi_delete_buttons') and self.roi_delete_buttons:
            for btn in self.roi_delete_buttons:
                try:
                    btn.remove()
                except:
                    pass
            self.roi_delete_buttons.clear()
        
        # 혹시 모를 남아있는 사각형도 제거 (edgecolor가 cyan인 사각형)
        if hasattr(self, 'ax') and self.ax:
            for patch in list(self.ax.patches):
                try:
                    # ROI 사각형인지 확인 (edgecolor가 cyan인 사각형)
                    if isinstance(patch, patches.Rectangle):
                        edgecolor = patch.get_edgecolor()
                        # edgecolor는 다양한 형식일 수 있음 (문자열, 튜플, 리스트 등)
                        if edgecolor == 'cyan' or (isinstance(edgecolor, (tuple, list)) and len(edgecolor) >= 3 and edgecolor[0] == 0.0 and edgecolor[1] == 1.0 and edgecolor[2] == 1.0):
                            patch.remove()
                except:
                    pass
    
    def _draw_roi_region_outline(self):
        """ROI를 드래그할 수 있는 영역(X,Y,Z 각 서브플롯의 플롯 구간)을 화면에 테두리로 표시"""
        if not self.current_csv_data or not hasattr(self, 'ax'):
            return
        try:
            img_height = self.current_csv_data['img_shape'][0]
            img_width = self.current_csv_data['img_shape'][1]
            plot_margin_top = 0.05
            plot_margin_bottom = 0.09
            subplot_margin_left = 0.09
            subplot_margin_right = 0.03
            subplot_width = 1.0 / 3.0
            # 플롯 전체 영역 (0~1, 0~1)에 해당하는 norm/픽셀 계산
            plot_y1, plot_y2 = 0.0, 1.0
            norm_y1 = plot_margin_bottom + plot_y1 * (1 - plot_margin_top - plot_margin_bottom)
            norm_y2 = plot_margin_bottom + plot_y2 * (1 - plot_margin_top - plot_margin_bottom)
            pixel_y1 = (img_height - 0.5) - (norm_y2 * img_height)
            pixel_y2 = (img_height - 0.5) - (norm_y1 * img_height)
            rect_height = pixel_y2 - pixel_y1
            local_x1 = subplot_margin_left
            local_x2 = 1.0 - subplot_margin_right
            for subplot_idx in range(3):
                subplot_start_x = subplot_idx * subplot_width
                norm_x1 = subplot_start_x + local_x1 * subplot_width
                norm_x2 = subplot_start_x + local_x2 * subplot_width
                pixel_x1 = (norm_x1 * img_width) - 0.5
                pixel_x2 = (norm_x2 * img_width) - 0.5
                rect_width = pixel_x2 - pixel_x1
                rect = patches.Rectangle(
                    (pixel_x1, pixel_y1), rect_width, rect_height,
                    linewidth=1.5, edgecolor='orange', facecolor='none', linestyle=':', alpha=0.9
                )
                self.ax.add_patch(rect)
                self.roi_region_outline_patches.append(rect)
            # "ROI 가능 영역" 문구는 제목 옆에 두기보다, 첫 번째 칸 위에 작게 표시
            if self.roi_region_outline_patches:
                # 첫 번째 사각형 왼쪽 위 근처에 텍스트 (데이터 좌표)
                x0 = (0 * subplot_width + local_x1 * subplot_width) * img_width - 0.5
                y0 = (img_height - 0.5) - (norm_y2 * img_height)
                t = self.ax.text(x0, y0 + rect_height + 8, 'ROI 가능 영역', fontsize=9, color='orange', alpha=0.95, clip_on=False)
                self.roi_region_outline_patches.append(t)
        except Exception:
            pass
    
    def draw_roi_rectangle(self):
        """저장된 ROI 영역을 사각형으로 표시 (여러 개 지원, 동시 모드: X,Y,Z 동일 / 개별 모드: 축마다 다른 영역, 각 영역에 삭제 버튼 추가)"""
        if not self.current_csv_data:
            return
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        try:
            self.clear_roi_rectangles()
            img_height = self.current_csv_data['img_shape'][0]
            img_width = self.current_csv_data['img_shape'][1]
            height_min = self.current_csv_data['height_min']
            height_max = self.current_csv_data['height_max']
            degree_min = self.current_csv_data['degree_min']
            degree_max = self.current_csv_data['degree_max']
            plot_margin_top = 0.05
            plot_margin_bottom = 0.09
            subplot_margin_left = 0.09
            subplot_margin_right = 0.03
            subplot_width = 1.0 / 3.0
            
            def draw_one_rect(roi, subplot_idx, color='cyan', linestyle='--', linewidth=2, add_delete_btn=True, axis_idx=None, region_idx=None):
                if roi is None or roi.get('degree_min') is None:
                    return
                plot_x1 = (roi['degree_min'] - degree_min) / (degree_max - degree_min) if (degree_max - degree_min) > 0 else 0.0
                plot_x2 = (roi['degree_max'] - degree_min) / (degree_max - degree_min) if (degree_max - degree_min) > 0 else 1.0
                plot_y1 = (roi['height_min'] - height_min) / (height_max - height_min) if (height_max - height_min) > 0 else 0.0
                plot_y2 = (roi['height_max'] - height_min) / (height_max - height_min) if (height_max - height_min) > 0 else 1.0
                norm_y1 = plot_margin_bottom + plot_y1 * (1 - plot_margin_top - plot_margin_bottom)
                norm_y2 = plot_margin_bottom + plot_y2 * (1 - plot_margin_top - plot_margin_bottom)
                local_x1 = subplot_margin_left + plot_x1 * (1 - subplot_margin_left - subplot_margin_right)
                local_x2 = subplot_margin_left + plot_x2 * (1 - subplot_margin_left - subplot_margin_right)
                pixel_y1 = (img_height - 0.5) - (norm_y2 * img_height)
                pixel_y2 = (img_height - 0.5) - (norm_y1 * img_height)
                rect_height = pixel_y2 - pixel_y1
                subplot_start_x = subplot_idx * subplot_width
                norm_x1 = subplot_start_x + local_x1 * subplot_width
                norm_x2 = subplot_start_x + local_x2 * subplot_width
                pixel_x1 = (norm_x1 * img_width) - 0.5
                pixel_x2 = (norm_x2 * img_width) - 0.5
                rect_width = pixel_x2 - pixel_x1
                rect = patches.Rectangle(
                    (pixel_x1, pixel_y1), rect_width, rect_height,
                    linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle
                )
                self.ax.add_patch(rect)
                self.roi_rectangles.append(rect)
                
                # 삭제 버튼 추가 (각 subplot의 영역 오른쪽 위 모서리에)
                if add_delete_btn and axis_idx is not None and region_idx is not None:
                    btn_text = self.ax.text(
                        pixel_x2 - 5, pixel_y1 + 5,  # 오른쪽 위 모서리
                        'X',  # ✕ 대신 X 사용 (폰트 경고 방지)
                        fontsize=11,
                        color='red',
                        fontweight='bold',
                        fontfamily='sans-serif',  # Arial 같은 기본 sans-serif 폰트 사용
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='red', linewidth=1.5, alpha=0.9),
                        picker=True,
                        ha='center',
                        va='center',
                        clip_on=False
                    )
                    # 버튼에 메타데이터 저장 (어떤 영역인지)
                    btn_text._roi_info = {
                        'mode': mode,
                        'axis_idx': axis_idx,
                        'region_idx': region_idx
                    }
                    self.roi_delete_buttons.append(btn_text)
            
            if mode == 'individual':
                # 개별 모드: 각 축(X,Y,Z)마다 해당 축의 모든 영역 그리기
                for i in range(3):
                    if i < len(self.roi_subplots) and self.roi_subplots[i]:
                        for region_idx, roi in enumerate(self.roi_subplots[i]):
                            draw_one_rect(roi, i, add_delete_btn=True, axis_idx=i, region_idx=region_idx)
            else:
                # 동시 모드: 모든 영역을 X,Y,Z 세 축에 그리기
                if self.roi_regions:
                    for region_idx, roi in enumerate(self.roi_regions):
                        for subplot_idx in range(3):
                            # 모든 축에 삭제 버튼 표시
                            draw_one_rect(roi, subplot_idx, add_delete_btn=True, axis_idx=-1, region_idx=region_idx)
        except Exception:
            pass
    
    def draw_temp_roi_rectangle(self):
        """임시로 드래그한 영역을 반투명하게 표시"""
        if not self.current_csv_data or not self.temp_roi_region:
            return
        
        mode = self.temp_roi_region['mode']
        subplot_idx = self.temp_roi_region['subplot_idx']
        
        try:
            # 기존 ROI 사각형들은 유지하고, 임시 사각형만 추가
            img_height = self.current_csv_data['img_shape'][0]
            img_width = self.current_csv_data['img_shape'][1]
            height_min = self.current_csv_data['height_min']
            height_max = self.current_csv_data['height_max']
            degree_min = self.current_csv_data['degree_min']
            degree_max = self.current_csv_data['degree_max']
            plot_margin_top = 0.05
            plot_margin_bottom = 0.09
            subplot_margin_left = 0.09
            subplot_margin_right = 0.03
            subplot_width = 1.0 / 3.0
            
            roi = self.temp_roi_region
            
            def draw_temp_rect(roi, subplot_idx):
                plot_x1 = (roi['degree_min'] - degree_min) / (degree_max - degree_min) if (degree_max - degree_min) > 0 else 0.0
                plot_x2 = (roi['degree_max'] - degree_min) / (degree_max - degree_min) if (degree_max - degree_min) > 0 else 1.0
                plot_y1 = (roi['height_min'] - height_min) / (height_max - height_min) if (height_max - height_min) > 0 else 0.0
                plot_y2 = (roi['height_max'] - height_min) / (height_max - height_min) if (height_max - height_min) > 0 else 1.0
                norm_y1 = plot_margin_bottom + plot_y1 * (1 - plot_margin_top - plot_margin_bottom)
                norm_y2 = plot_margin_bottom + plot_y2 * (1 - plot_margin_top - plot_margin_bottom)
                local_x1 = subplot_margin_left + plot_x1 * (1 - subplot_margin_left - subplot_margin_right)
                local_x2 = subplot_margin_left + plot_x2 * (1 - subplot_margin_left - subplot_margin_right)
                pixel_y1 = (img_height - 0.5) - (norm_y2 * img_height)
                pixel_y2 = (img_height - 0.5) - (norm_y1 * img_height)
                rect_height = pixel_y2 - pixel_y1
                subplot_start_x = subplot_idx * subplot_width
                norm_x1 = subplot_start_x + local_x1 * subplot_width
                norm_x2 = subplot_start_x + local_x2 * subplot_width
                pixel_x1 = (norm_x1 * img_width) - 0.5
                pixel_x2 = (norm_x2 * img_width) - 0.5
                rect_width = pixel_x2 - pixel_x1
                rect = patches.Rectangle(
                    (pixel_x1, pixel_y1), rect_width, rect_height,
                    linewidth=2, edgecolor='yellow', facecolor='yellow', linestyle='-', alpha=0.3
                )
                self.ax.add_patch(rect)
                self.roi_rectangles.append(rect)
            
            if mode == 'individual':
                # 개별 모드: 해당 축에만 표시
                draw_temp_rect(roi, subplot_idx)
            else:
                # 동시 모드: X,Y,Z 모두에 표시
                for i in range(3):
                    draw_temp_rect(roi, i)
        except Exception:
            pass
    
    def on_image_click(self, event):
        """이미지 클릭 이벤트 핸들러 (사용하지 않음, RectangleSelector 사용)"""
        pass

    def disable_ui(self):
        """UI 비활성화 (로딩 중)"""
        self.root.config(cursor="wait")
        if hasattr(self, 'delete_btn'):
            self.delete_btn.config(state=tk.DISABLED)
        if hasattr(self, 'prev_btn'):
            self.prev_btn.config(state=tk.DISABLED)
        if hasattr(self, 'next_btn'):
            self.next_btn.config(state=tk.DISABLED)
        if hasattr(self, 'save_btn'):
            self.save_btn.config(state=tk.DISABLED)
        # Entry 필드 비활성화
        if hasattr(self, 'height_entry'):
            self.height_entry.config(state=tk.DISABLED)
        if hasattr(self, 'degree_entry'):
            self.degree_entry.config(state=tk.DISABLED)
        self.root.update()
    
    def enable_ui(self):
        """UI 활성화 (로딩 완료)"""
        self.root.config(cursor="")
        if hasattr(self, 'delete_btn'):
            self.delete_btn.config(state=tk.NORMAL)
        if hasattr(self, 'prev_btn'):
            self.prev_btn.config(state=tk.NORMAL if self.has_previous_image() else tk.DISABLED)
        if hasattr(self, 'next_btn'):
            self.next_btn.config(state=tk.NORMAL if self.has_next_image() else tk.DISABLED)
        # 저장 버튼: 변경사항 있을 때만 활성화
        self.update_save_button()
        # Entry 필드 활성화
        if hasattr(self, 'height_entry'):
            self.height_entry.config(state=tk.NORMAL)
        if hasattr(self, 'degree_entry'):
            self.degree_entry.config(state=tk.NORMAL)
        self.root.update()
    
    def _has_roi_changes(self):
        """현재 ROI가 저장된 내용과 다른지 여부 (변경사항 있으면 True, 여러 영역 지원)"""
        if not self.current_roi_info_file:
            return False
        
        # 저장 파일이 없으면: ROI 데이터가 있으면 변경 있음(저장 가능)
        if not self.current_roi_info_file.exists():
            mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
            if mode == 'sync':
                return bool(self.roi_regions) or bool(self.temp_roi_region)
            else:
                # 개별 모드: 각 축에 영역이 하나라도 있으면 True
                has_regions = any(len(regions) > 0 for regions in self.roi_subplots if regions)
                return has_regions or bool(self.temp_roi_region)
        
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        saved = self.roi_info
        tol = 1e-6
        
        # 임시 영역이 있으면 변경사항 있음
        if self.temp_roi_region:
            return True
        
        if mode == 'sync':
            # 동시 모드: roi_regions 비교
            saved_regions = saved.get('roi_regions', [])
            
            # 기존 단일 영역 형식 호환
            if not saved_regions and saved.get('roi_degree_min') is not None:
                saved_regions = [{
                    'degree_min': saved.get('roi_degree_min'),
                    'degree_max': saved.get('roi_degree_max'),
                    'height_min': saved.get('roi_height_min'),
                    'height_max': saved.get('roi_height_max')
                }]
            
            # 영역 개수가 다르면 변경사항 있음
            if len(self.roi_regions) != len(saved_regions):
                return True
            
            # 각 영역 비교
            for cur_r, saved_r in zip(self.roi_regions, saved_regions):
                if (abs(cur_r['degree_min'] - saved_r['degree_min']) > tol or
                    abs(cur_r['degree_max'] - saved_r['degree_max']) > tol or
                    abs(cur_r['height_min'] - saved_r['height_min']) > tol or
                    abs(cur_r['height_max'] - saved_r['height_max']) > tol):
                    return True
            
            # 모든 영역이 같으면 False
            return False
        else:
            # 개별 모드: roi_subplots vs roi_0/1/2_regions
            for i in range(3):
                cur_regions = self.roi_subplots[i] if i < len(self.roi_subplots) else []
                saved_regions = saved.get(f'roi_{i}_regions', [])
                
                # 기존 단일 영역 형식 호환
                if not saved_regions and saved.get(f'roi_{i}_degree_min') is not None:
                    saved_regions = [{
                        'degree_min': saved.get(f'roi_{i}_degree_min'),
                        'degree_max': saved.get(f'roi_{i}_degree_max'),
                        'height_min': saved.get(f'roi_{i}_height_min'),
                        'height_max': saved.get(f'roi_{i}_height_max')
                    }]
                
                # 영역 개수가 다르면 변경사항 있음
                if len(cur_regions) != len(saved_regions):
                    return True
                
                # 각 영역 비교
                for cur_r, saved_r in zip(cur_regions, saved_regions):
                    if (abs(cur_r['degree_min'] - saved_r['degree_min']) > tol or
                        abs(cur_r['degree_max'] - saved_r['degree_max']) > tol or
                        abs(cur_r['height_min'] - saved_r['height_min']) > tol or
                        abs(cur_r['height_max'] - saved_r['height_max']) > tol):
                        return True
            
            # 모든 축에서 변경사항이 없으면 False
            return False
    
    def update_save_button(self):
        """변경사항 있으면 저장 버튼 활성화, 없으면 비활성화"""
        if not hasattr(self, 'save_btn'):
            return
        if self._has_roi_changes():
            self.save_btn.config(state=tk.NORMAL)
        else:
            self.save_btn.config(state=tk.DISABLED)
    
    def _update_delete_restore_button(self):
        """현재 이미지가 삭제됨이면 되돌리기 버튼, 아니면 삭제 버튼으로 표시"""
        if not hasattr(self, 'delete_btn'):
            return
        if self.roi_info.get('deleted'):
            self.delete_btn.config(text="↩ 되돌리기", command=self.restore_pole)
        else:
            self.delete_btn.config(text="🗑️ 삭제", command=self.delete_pole)
    
    def add_roi_region(self):
        """임시로 드래그한 영역을 리스트에 추가"""
        if not self.temp_roi_region:
            messagebox.showwarning("경고", "먼저 마우스로 영역을 드래그해주세요.")
            return
        
        mode = self.temp_roi_region['mode']
        subplot_idx = self.temp_roi_region['subplot_idx']
        
        new_region = {
            'degree_min': self.temp_roi_region['degree_min'],
            'degree_max': self.temp_roi_region['degree_max'],
            'height_min': self.temp_roi_region['height_min'],
            'height_max': self.temp_roi_region['height_max']
        }
        
        if mode == 'individual':
            # 개별 모드: 해당 축에만 추가
            self.roi_subplots[subplot_idx].append(new_region)
            axis_name = ['X', 'Y', 'Z'][subplot_idx]
            messagebox.showinfo("영역 추가됨", f"{axis_name}축에 영역이 추가되었습니다.\n현재 {axis_name}축 영역 개수: {len(self.roi_subplots[subplot_idx])}개")
        else:
            # 동시 모드: 공통 리스트에 추가
            self.roi_regions.append(new_region)
            messagebox.showinfo("영역 추가됨", f"영역이 추가되었습니다. (X,Y,Z 공통)\n현재 영역 개수: {len(self.roi_regions)}개")
        
        # 임시 영역 초기화
        self.temp_roi_region = None
        
        # 화면 업데이트
        self._refresh_subplot_display()
        self.clear_roi_rectangles()
        self.draw_roi_rectangle()
        self.canvas.draw_idle()
        self.update_save_button()
    
    def remove_last_roi_region(self):
        """마지막으로 추가한 영역 삭제"""
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        
        if mode == 'individual':
            # 개별 모드: 현재 선택된 축의 마지막 영역 삭제
            idx = self.current_editing_subplot_idx
            if self.roi_subplots[idx]:
                self.roi_subplots[idx].pop()
                axis_name = ['X', 'Y', 'Z'][idx]
                messagebox.showinfo("영역 삭제됨", f"{axis_name}축의 마지막 영역이 삭제되었습니다.\n남은 {axis_name}축 영역 개수: {len(self.roi_subplots[idx])}개")
            else:
                axis_name = ['X', 'Y', 'Z'][idx]
                messagebox.showwarning("경고", f"{axis_name}축에 삭제할 영역이 없습니다.")
                return
        else:
            # 동시 모드: 공통 리스트의 마지막 영역 삭제
            if self.roi_regions:
                self.roi_regions.pop()
                messagebox.showinfo("영역 삭제됨", f"마지막 영역이 삭제되었습니다.\n남은 영역 개수: {len(self.roi_regions)}개")
            else:
                messagebox.showwarning("경고", "삭제할 영역이 없습니다.")
                return
        
        # 화면 업데이트
        self._refresh_subplot_display()
        self.clear_roi_rectangles()
        self.draw_roi_rectangle()
        self.canvas.draw_idle()
        self.update_save_button()
    
    def on_delete_button_click(self, event):
        """삭제 버튼 클릭 이벤트 핸들러"""
        if not hasattr(event, 'artist'):
            return
        
        artist = event.artist
        
        # 클릭된 객체가 삭제 버튼인지 확인
        if not hasattr(artist, '_roi_info'):
            return
        
        roi_info = artist._roi_info
        mode = roi_info['mode']
        axis_idx = roi_info['axis_idx']
        region_idx = roi_info['region_idx']
        
        # 삭제 확인
        if mode == 'individual':
            axis_name = ['X', 'Y', 'Z'][axis_idx]
            confirm = messagebox.askyesno("삭제 확인", 
                                         f"{axis_name}축의 {region_idx + 1}번째 영역을 삭제하시겠습니까?")
        else:
            confirm = messagebox.askyesno("삭제 확인", 
                                         f"{region_idx + 1}번째 영역을 삭제하시겠습니까? (X,Y,Z 공통)")
        
        if not confirm:
            return
        
        # 영역 삭제
        try:
            if mode == 'individual':
                if axis_idx < len(self.roi_subplots) and region_idx < len(self.roi_subplots[axis_idx]):
                    del self.roi_subplots[axis_idx][region_idx]
                    axis_name = ['X', 'Y', 'Z'][axis_idx]
                    messagebox.showinfo("영역 삭제됨", 
                                       f"{axis_name}축의 영역이 삭제되었습니다.\n남은 {axis_name}축 영역 개수: {len(self.roi_subplots[axis_idx])}개")
            else:
                if region_idx < len(self.roi_regions):
                    del self.roi_regions[region_idx]
                    messagebox.showinfo("영역 삭제됨", 
                                       f"영역이 삭제되었습니다. (X,Y,Z 공통)\n남은 영역 개수: {len(self.roi_regions)}개")
            
            # 화면 업데이트
            self._refresh_subplot_display()
            self.clear_roi_rectangles()
            self.draw_roi_rectangle()
            self.canvas.draw_idle()
            
            # 자동 저장
            self.auto_save_roi()
        except Exception as e:
            messagebox.showerror("오류", f"영역 삭제 중 오류가 발생했습니다: {e}")
    
    def reset_roi_area(self):
        """현재 이미지에서 그린 ROI 영역을 모두 지움 (초기화)"""
        if not self.image_files or self.current_image_idx >= len(self.image_files):
            return
        
        # 모든 영역 초기화
        self.roi_regions = []
        self.roi_subplots = [[], [], []]
        self.temp_roi_region = None
        
        self._refresh_subplot_display()
        self.clear_roi_rectangles()
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw_idle()
        
        # 자동 저장 (빈 영역으로 저장)
        self.auto_save_roi()
    
    def _on_roi_mode_changed(self):
        """동시/개별 모드 전환 시 데이터 정합 및 X,Y,Z 버튼 상태 갱신"""
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        if mode == 'individual':
            # 동시 -> 개별: 기존 영역들을 X,Y,Z 모두에 복사
            if self.roi_regions and all(not s for s in self.roi_subplots):
                for i in range(3):
                    self.roi_subplots[i] = [dict(r) for r in self.roi_regions]  # 깊은 복사
            self.current_editing_subplot_idx = 0
            if hasattr(self, 'btn_x'):
                self.btn_x.config(state=tk.NORMAL)
                self.btn_y.config(state=tk.NORMAL)
                self.btn_z.config(state=tk.NORMAL)
            self._refresh_subplot_display()
        else:
            # 개별 -> 동시: X(0) 값을 공통으로 사용
            if self.roi_subplots[0]:
                self.roi_regions = [dict(r) for r in self.roi_subplots[0]]  # 깊은 복사
                self._refresh_subplot_display()
            if hasattr(self, 'btn_x'):
                self.btn_x.config(state=tk.DISABLED)
                self.btn_y.config(state=tk.DISABLED)
                self.btn_z.config(state=tk.DISABLED)
        if hasattr(self, 'canvas') and self.current_csv_data:
            self.clear_roi_rectangles()
            self.draw_roi_rectangle()
            self.canvas.draw_idle()
        self.update_save_button()
    
    def _select_subplot(self, idx: int):
        """개별 모드에서 편집할 축(X/Y/Z) 선택"""
        if (self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync') != 'individual':
            return
        self.current_editing_subplot_idx = max(0, min(2, idx))
        self._refresh_subplot_display()
    
    def _refresh_subplot_display(self):
        """축별(X,Y,Z) ROI 값을 개별 영역 정보 라벨에 반영 (여러 개 영역 지원)"""
        if not hasattr(self, '_axis_degree_min_vars'):
            return
        mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
        for i in range(3):
            if mode == 'individual' and self.roi_subplots[i]:
                # 개별 모드: 해당 축의 영역 개수 표시
                count = len(self.roi_subplots[i])
                if count == 1:
                    s = self.roi_subplots[i][0]
                    self._axis_degree_min_vars[i].set(f"{s['degree_min']:.1f}")
                    self._axis_degree_max_vars[i].set(f"{s['degree_max']:.1f}")
                    self._axis_height_min_vars[i].set(f"{s['height_min']:.3f}")
                    self._axis_height_max_vars[i].set(f"{s['height_max']:.3f}")
                else:
                    self._axis_degree_min_vars[i].set(f"{count}개 영역")
                    self._axis_degree_max_vars[i].set("")
                    self._axis_height_min_vars[i].set("")
                    self._axis_height_max_vars[i].set("")
            elif mode == 'sync' and self.roi_regions:
                # 동시 모드: 영역 개수 표시
                count = len(self.roi_regions)
                if count == 1:
                    s = self.roi_regions[0]
                    self._axis_degree_min_vars[i].set(f"{s['degree_min']:.1f}")
                    self._axis_degree_max_vars[i].set(f"{s['degree_max']:.1f}")
                    self._axis_height_min_vars[i].set(f"{s['height_min']:.3f}")
                    self._axis_height_max_vars[i].set(f"{s['height_max']:.3f}")
                else:
                    self._axis_degree_min_vars[i].set(f"{count}개 영역")
                    self._axis_degree_max_vars[i].set("")
                    self._axis_height_min_vars[i].set("")
                    self._axis_height_max_vars[i].set("")
            else:
                self._axis_degree_min_vars[i].set("-")
                self._axis_degree_max_vars[i].set("-")
                self._axis_height_min_vars[i].set("-")
                self._axis_height_max_vars[i].set("-")
    
    def on_key_press(self, event):
        """키보드 이벤트 핸들러 (전역)"""
        key = event.keysym
        
        # 's' 키: 저장
        if key == 's' or key == 'S':
            self.save_data()
        # 왼쪽 화살표 또는 위쪽 화살표: 이전 이미지
        elif key == 'Left' or key == 'Up':
            if self.has_previous_image():
                self.prev_image()
        # 오른쪽 화살표 또는 아래쪽 화살표: 다음 이미지
        elif key == 'Right' or key == 'Down':
            if self.has_next_image():
                self.next_image()
    
    def on_entry_key_press(self, event):
        """Entry 위젯에서의 키보드 이벤트 핸들러"""
        key = event.keysym
        
        # 's' 키: 저장 (Entry에서도 작동)
        if key == 's' or key == 'S':
            self.save_data()
            return "break"  # Entry에 's'가 입력되지 않도록
        # 방향키는 Entry의 기본 동작 사용 (커서 이동)
        # 화살표 키는 Entry에서 커서 이동으로 사용되므로 여기서는 처리하지 않음
    
    def prev_image(self):
        """이전 이미지로 이동 (이전 전주까지 고려)"""
        # 현재 전주의 이전 이미지가 있으면 이동
        if self.current_image_idx > 0:
            self.load_image(self.current_image_idx - 1)
        # 현재 전주가 첫 번째 이미지이고, 이전 전주가 있으면 이전 전주의 마지막 이미지로 이동
        elif self.current_pole_idx > 0:
            # 저장 여부 확인
            if not self.check_save_before_switch():
                return
            # 이전 전주로 이동
            self.current_pole_idx -= 1
            self.load_pole(self.all_pole_dirs[self.current_pole_idx])
            # 마지막 이미지 로드
            if self.image_files:
                self.load_image(len(self.image_files) - 1)
    
    def next_image(self):
        """다음 이미지로 이동 (다음 전주까지 고려)"""
        # 현재 전주의 다음 이미지가 있으면 이동
        if self.current_image_idx < len(self.image_files) - 1:
            self.load_image(self.current_image_idx + 1)
        # 현재 전주가 마지막 이미지이고, 다음 전주가 있으면 다음 전주의 첫 번째 이미지로 이동
        elif self.current_pole_idx < len(self.all_pole_dirs) - 1:
            # 저장 여부 확인
            if not self.check_save_before_switch():
                return
            # 다음 전주로 이동
            self.current_pole_idx += 1
            self.load_pole(self.all_pole_dirs[self.current_pole_idx])
            # 첫 번째 이미지 로드
            if self.image_files:
                self.load_image(0)
    
    
    def check_save_before_switch(self) -> bool:
        """이미지 전환 전 저장 확인"""
        # ROI 영역이 설정되었는지 확인
        try:
            if self.roi_degree_min is not None and self.roi_degree_max is not None and \
               self.roi_height_min is not None and self.roi_height_max is not None:
                # ROI 정보가 저장되었는지 확인
                if self.current_roi_info_file and self.current_roi_info_file.exists():
                    saved_info = load_break_info(str(self.current_roi_info_file))
                    saved_degree_min = saved_info.get('roi_degree_min')
                    saved_degree_max = saved_info.get('roi_degree_max')
                    saved_height_min = saved_info.get('roi_height_min')
                    saved_height_max = saved_info.get('roi_height_max')
                    
                    # ROI 정보가 변경되었는지 확인
                    roi_changed = (
                        abs(saved_degree_min - self.roi_degree_min) > 0.01 or
                        abs(saved_degree_max - self.roi_degree_max) > 0.01 or
                        abs(saved_height_min - self.roi_height_min) > 0.001 or
                        abs(saved_height_max - self.roi_height_max) > 0.001
                    )
                else:
                    # 저장된 파일이 없으면 변경된 것으로 간주
                    roi_changed = True
                
                if roi_changed:
                    response = messagebox.askyesnocancel(
                        "저장 확인",
                        "ROI 영역이 설정되었지만 저장되지 않았습니다.\n저장하시겠습니까?\n\n예: 저장 후 이동\n아니오: 저장하지 않고 이동\n취소: 이동 취소"
                    )
                    
                    if response is None:  # 취소
                        return False
                    elif response:  # 예 - 저장
                        self.save_data()
                        return True
                    else:  # 아니오 - 저장하지 않고 이동
                        return True
        except (ValueError, TypeError, AttributeError):
            pass
        
        return True
    
    def load_pole(self, pole_dir: Path):
        """전주 데이터 로드"""
        self.pole_dir = pole_dir
        self.poleid = pole_dir.name
        
        # 5. edit_data 디렉토리 경로 설정 (make_ai\5. edit_data)
        project_name = pole_dir.parent.name
        crop_data_dir = EDIT_DATA_SAVE_DIR / "break"
        self.crop_data_pole_dir = crop_data_dir / project_name / self.poleid
        os.makedirs(self.crop_data_pole_dir, exist_ok=True)
        
        # 전주 인덱스 업데이트
        if pole_dir in self.all_pole_dirs:
            self.current_pole_idx = self.all_pole_dirs.index(pole_dir)
        
        # 이미지 파일 목록 (삭제된 이미지 포함하여 모두 표시)
        all_images = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
        self.break_info_files = {}
        self.image_files = []
        for img_file in all_images:
            break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
            if break_info_file.exists():
                self.break_info_files[img_file] = break_info_file
            self.image_files.append(img_file)
        self.current_image_idx = 0
        
        # 이미지 파일이 없으면 생성
        if not self.image_files:
            self.generate_image()
            # generate_image 함수 내부에서 이미 확정되지 않은 이미지만 필터링하여 목록을 다시 불러옴
            # generate_image 후에도 이미지가 없으면, 모든 이미지가 확정된 경우일 수 있음
            if not self.image_files:
                # 확정되지 않은 이미지가 없는 경우 확인
                all_images_check = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
                # 이미지 파일이 있지만 모두 확정된 경우 또는 이미지 파일이 없는 경우 모두 조용히 처리
        
        # 현재 ROI 정보 파일 경로 (첫 번째 이미지 기준)
        self.current_roi_info_file = None
        if self.image_files:
            img_file = self.image_files[0]
            self.current_roi_info_file = self.crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
        
        # ROI 정보 읽기 (첫 번째 이미지 기준)
        if self.current_roi_info_file and self.current_roi_info_file.exists():
            self.roi_info = load_break_info(str(self.current_roi_info_file))
            self.roi_degree_min = self.roi_info.get('roi_degree_min', None)
            self.roi_degree_max = self.roi_info.get('roi_degree_max', None)
            self.roi_height_min = self.roi_info.get('roi_height_min', None)
            self.roi_height_max = self.roi_info.get('roi_height_max', None)
        else:
            self.roi_info = {}
            self.roi_degree_min = None
            self.roi_degree_max = None
            self.roi_height_min = None
            self.roi_height_max = None
        
        # 입력 필드 업데이트 (축별 개별 영역 정보)
        if hasattr(self, '_axis_degree_min_vars'):
            self._refresh_subplot_display()
        
        # GUI 제목 업데이트
        self.root.title(f"ROI 영역 설정 - {self.poleid}")
        
        # 정보 라벨 업데이트
        if hasattr(self, 'poleid_label'):
            self.poleid_label.config(text=f"전주ID: {self.poleid}")
        if hasattr(self, 'image_count_label'):
            self.image_count_label.config(text=f"이미지 수: {len(self.image_files)}개")
        
        # 이미지 목록 업데이트
        if hasattr(self, 'image_list_tree'):
            self.update_image_list()
        
        # 전체 대비 저장 통계 라벨 갱신
        self.update_global_stats_label()
        
        # 이미지 로드 (이미지가 있으면 무조건 로드, 확정 여부와 관계없이)
        if self.image_files:
            self.load_image(0)
        elif hasattr(self, 'ax') and hasattr(self, 'canvas'):
            # 이미지 파일이 있지만 모두 삭제된 경우
            all_images_check = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
            if all_images_check:
                self.ax.clear()
                self.ax.text(0.5, 0.5, '이 전주의 모든 이미지가 삭제되었습니다.', 
                            ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                self.canvas.draw()
            else:
                # 이미지 파일 자체가 없는 경우 - 다음 전주로 자동 이동
                if hasattr(self, 'all_pole_dirs') and len(self.all_pole_dirs) > 1:
                    # 다음 전주가 있으면 자동으로 이동
                    if self.current_pole_idx < len(self.all_pole_dirs) - 1:
                        self.ax.clear()
                        self.ax.text(0.5, 0.5, f'이 전주에 이미지 파일이 없습니다.\n다음 전주로 이동합니다...', 
                                    ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                        self.canvas.draw()
                        # 다음 전주로 자동 이동
                        self.current_pole_idx += 1
                        self.load_pole(self.all_pole_dirs[self.current_pole_idx])
                        return
                    else:
                        # 마지막 전주인 경우
                        self.ax.clear()
                        self.ax.text(0.5, 0.5, '이 전주에 이미지 파일이 없습니다.\n(마지막 전주입니다.)', 
                                    ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                        self.canvas.draw()
                else:
                    # 단일 전주 모드인 경우
                    self.ax.clear()
                    self.ax.text(0.5, 0.5, '이 전주에 이미지 파일이 없습니다.', 
                                ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
                    self.canvas.draw()
        if hasattr(self, 'pole_count_label'):
            self.pole_count_label.config(text=f"전주: {self.current_pole_idx + 1}/{len(self.all_pole_dirs)}")
        
        # 이전/다음 버튼 상태 업데이트 (전주 포함)
        if hasattr(self, 'prev_btn'):
            self.prev_btn.config(state=tk.NORMAL if self.has_previous_image() else tk.DISABLED)
        if hasattr(self, 'next_btn'):
            self.next_btn.config(state=tk.NORMAL if self.has_next_image() else tk.DISABLED)
        
        # 첫 번째 이미지 로드
        if self.image_files:
            self.load_image(0)
        
        # 상태 변수 초기화
        self.saved = False
        self.deleted = False
    
    def save_and_refresh_image(self):
        """ROI 영역 자동 저장 (사용하지 않음, ROI는 수동 저장)"""
        # ROI 편집기에서는 사용하지 않음
        pass
    
    def auto_save_roi(self):
        """ROI 정보 자동 저장 (메시지 없이 조용히 저장)"""
        try:
            if not self.current_roi_info_file:
                return
            
            # 현재 UI 모드(동시/개별)대로 저장
            mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
            info = {'poleid': self.poleid, 'project_name': self.pole_dir.parent.name}
            
            if mode == 'sync':
                if not self.roi_regions:
                    return
                info['roi_regions'] = self.roi_regions
            else:
                # 개별 모드: 영역을 그린 축만 저장
                any_ok = False
                for i in range(3):
                    if i < len(self.roi_subplots) and self.roi_subplots[i]:
                        info[f'roi_{i}_regions'] = self.roi_subplots[i]
                        any_ok = True
                
                if not any_ok:
                    return
            
            if save_break_info(str(self.current_roi_info_file), info):
                self.saved = True
                self.roi_info = info
                self._last_save_mode = mode
                # 임시 영역 초기화
                self.temp_roi_region = None
                if hasattr(self, 'image_list_tree'):
                    self.update_image_list()
        except Exception as e:
            print(f"자동 저장 중 오류: {e}")
    
    def save_data(self):
        """ROI 정보 저장. 동시/개별 중 가장 최근에 저장한 방식으로 저장 (여러 영역 지원)."""
        self.disable_ui()
        try:
            if not self.current_roi_info_file:
                messagebox.showerror("오류", "ROI 정보 파일을 찾을 수 없습니다.")
                return
            
            # 임시 영역이 있으면 경고
            if self.temp_roi_region:
                if not messagebox.askyesno("확인", "드래그한 영역이 추가되지 않았습니다.\n이 영역을 무시하고 저장하시겠습니까?"):
                    self.enable_ui()
                    return
            
            # 현재 UI 모드(동시/개별)대로 저장 (화면에 보이는 내용이 그대로 저장되도록)
            mode = self.roi_apply_mode.get() if hasattr(self, 'roi_apply_mode') else 'sync'
            info = {'poleid': self.poleid, 'project_name': self.pole_dir.parent.name}
            
            if mode == 'sync':
                if not self.roi_regions:
                    messagebox.showwarning("경고", "ROI 영역을 먼저 선택해주세요.")
                    return
                info['roi_regions'] = self.roi_regions
                msg = f"ROI 영역이 저장되었습니다. (X,Y,Z 동일)\n영역 개수: {len(self.roi_regions)}개"
            else:
                # 개별 모드: 영역을 그린 축만 저장. 그리지 않은 축은 저장하지 않음.
                axis_names = ['X', 'Y', 'Z']
                any_ok = False
                lines = ["개별 저장"]
                for i in range(3):
                    if i < len(self.roi_subplots) and self.roi_subplots[i]:
                        info[f'roi_{i}_regions'] = self.roi_subplots[i]
                        any_ok = True
                        lines.append(f"  {axis_names[i]}: {len(self.roi_subplots[i])}개 영역")
                
                if not any_ok:
                    messagebox.showwarning("경고", "한 개 이상의 축에 ROI 영역을 그려주세요.")
                    return
                msg = "\n".join(lines)
            
            if save_break_info(str(self.current_roi_info_file), info):
                self.saved = True
                self.roi_info = info
                self._last_save_mode = mode  # 이번에 사용한 방식을 최근 저장 방식으로 기록
                # 임시 영역 초기화
                self.temp_roi_region = None
                messagebox.showinfo("저장 완료", msg)
                if hasattr(self, 'image_list_tree'):
                    self.update_image_list()
                # 저장 후 다음 이미지가 있으면 자동으로 이동
                if self.has_next_image():
                    self.next_image()
            else:
                messagebox.showerror("저장 실패", "파일 저장에 실패했습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"저장 중 오류가 발생했습니다: {e}")
        finally:
            self.enable_ui()
    
    def restore_pole(self):
        """삭제된 이미지를 되돌리기 (deleted 플래그 해제)"""
        if not self.image_files or self.current_image_idx >= len(self.image_files):
            messagebox.showwarning("경고", "되돌릴 이미지가 없습니다.")
            return
        current_img_file = self.image_files[self.current_image_idx]
        img_filename = current_img_file.name
        if not self.roi_info.get('deleted'):
            return
        self.disable_ui()
        try:
            break_info_filename = img_filename.replace("_2d_plot.png", "_break_info.json")
            if hasattr(self, 'crop_data_pole_dir') and self.crop_data_pole_dir.exists():
                crop_data_roi_info_file = self.crop_data_pole_dir / break_info_filename.replace("_break_info.json", "_roi_info.json")
                if crop_data_roi_info_file.exists():
                    info = load_break_info(str(crop_data_roi_info_file))
                    info['deleted'] = False
                    save_break_info(str(crop_data_roi_info_file), info)
                    self.roi_info['deleted'] = False
            break_info_file = self.pole_dir / break_info_filename
            if break_info_file.exists():
                info = load_break_info(str(break_info_file))
                info['deleted'] = False
                save_break_info(str(break_info_file), info)
            idx = self.current_image_idx
            self.ax.set_title(f"{self.poleid} - 이미지 {idx + 1}/{len(self.image_files)}\n파일: {img_filename}\n(마우스로 드래그하여 ROI 영역 선택)", 
                              fontsize=12, fontweight='bold', pad=10)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            self.update_global_stats_label()
            self._update_delete_restore_button()
            messagebox.showinfo("알림", "삭제를 되돌렸습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"되돌리기 중 오류가 발생했습니다: {e}")
        finally:
            self.enable_ui()
    
    def delete_pole(self):
        """현재 이미지를 소프트 삭제 (파일은 유지하고 deleted 플래그만 설정)"""
        if not self.image_files or self.current_image_idx >= len(self.image_files):
            messagebox.showwarning("경고", "삭제할 이미지가 없습니다.")
            return
        
        current_img_file = self.image_files[self.current_image_idx]
        img_filename = current_img_file.name
        
        if not messagebox.askyesno("삭제 확인", 
                                   f"현재 이미지를 삭제 처리하시겠습니까?\n\n파일: {img_filename}\n\n(소프트 삭제: 파일은 유지되고 목록에서만 제외됩니다)"):
            return
        
        # UI 비활성화
        self.disable_ui()
        
        try:
            # 이미지 파일명에서 파단 정보 파일명 생성
            break_info_filename = img_filename.replace("_2d_plot.png", "_break_info.json")
            
            # 5. edit_data의 roi_info.json에 deleted 플래그 설정
            if hasattr(self, 'crop_data_pole_dir') and self.crop_data_pole_dir.exists():
                crop_data_roi_info_file = self.crop_data_pole_dir / break_info_filename.replace("_break_info.json", "_roi_info.json")
                
                # roi_info 파일이 있으면 deleted 플래그 추가, 없으면 생성
                if crop_data_roi_info_file.exists():
                    info = load_break_info(str(crop_data_roi_info_file))
                else:
                    info = {
                        'poleid': self.poleid,
                        'project_name': self.pole_dir.parent.name
                    }
                
                info['deleted'] = True
                save_break_info(str(crop_data_roi_info_file), info)
            
            # 4. merge_data의 break_info.json에도 deleted 플래그 설정 (선택사항)
            break_info_file = self.pole_dir / break_info_filename
            if break_info_file.exists():
                info = load_break_info(str(break_info_file))
                info['deleted'] = True
                save_break_info(str(break_info_file), info)
            
            # 목록에서 제거하지 않음. 메모리의 roi_info와 제목만 갱신 후 같은 이미지 유지
            self.deleted = True
            self.roi_info['deleted'] = True
            idx = self.current_image_idx
            title_suffix = "\n[삭제됨]"
            self.ax.set_title(f"{self.poleid} - 이미지 {idx + 1}/{len(self.image_files)}\n파일: {img_filename}{title_suffix}", 
                              fontsize=12, fontweight='bold', pad=10)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            
            # 이미지 목록·통계 갱신
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            self.update_global_stats_label()
            
            # 버튼을 되돌리기로 전환
            self._update_delete_restore_button()
            
            # 삭제 후 다음 이미지가 있으면 자동으로 이동
            if self.has_next_image():
                self.next_image()
            
        except Exception as e:
            messagebox.showerror("삭제 실패", f"삭제 처리 중 오류 발생: {e}")
        finally:
            # UI 활성화
            self.enable_ui()
    
    def close_window(self):
        """창 닫기"""
        if self.saved:
            self.root.destroy()
        elif messagebox.askyesno("확인", "저장하지 않고 종료하시겠습니까?"):
            self.root.destroy()
    
    def get_result(self) -> str:
        """결과 반환: 'saved', 'deleted', 또는 'cancelled'"""
        if self.deleted:
            return 'deleted'
        elif self.saved:
            return 'saved'
        else:
            return 'cancelled'


def edit_single_pole(pole_dir: Path) -> str:
    """단일 전주 폴더의 break_info.json 수정 (GUI)"""
    poleid = pole_dir.name
    
    # 이미지 파일이 있는지 확인 (이미지 파일이 없으면 건너뛰기)
    image_files = list(pole_dir.glob("*_OUT_processed_2d_plot.png"))
    
    if not image_files:
        print(f"  건너뜀: {poleid} 전주에 이미지 파일이 없습니다.")
        return 'skipped'
    
    # GUI 실행
    root = tk.Tk()
    app = ROIEditorGUI(root, pole_dir)
    root.mainloop()
    
    return app.get_result() if hasattr(app, 'get_result') else 'cancelled'


def find_first_unsaved_image(all_pole_dirs: List[Path]) -> Tuple[Path, int]:
    """가장 상단(트리 순서) 미저장 이미지의 (전주, 이미지 인덱스) 반환. 모두 저장이면 (첫 전주, 0)."""
    for pole_dir in all_pole_dirs:
        project_name = pole_dir.parent.name
        poleid = pole_dir.name
        crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / poleid
        all_images = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
        for idx, img_file in enumerate(all_images):
            roi_path = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
            if roi_path.exists():
                try:
                    info = load_break_info(str(roi_path))
                    if info.get("deleted") or roi_info_has_saved_roi(info):
                        continue
                except Exception:
                    pass
            return (pole_dir, idx)
    return (all_pole_dirs[0], 0) if all_pole_dirs else (Path("."), 0)


def edit_all_poles(base_dir: str):
    """모든 전주 폴더의 break_info.json 수정 (GUI - 전주 간 이동 가능)"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다: {base_dir}")
        print(f"      경로를 확인하거나 --input-dir 옵션으로 올바른 경로를 지정하세요.")
        return
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다.")
        print(f"      입력 디렉토리: {base_dir}")
        print(f"      이 디렉토리가 비어있거나 프로젝트 폴더가 없습니다.")
        print(f"      예상 구조: {base_dir}/프로젝트명/전주ID/")
        return
    
    # 모든 전주 디렉토리 수집 (이미지 파일이 있는 전주만 포함)
    all_pole_dirs = []
    for project_dir in sorted(projects):
        pole_dirs = sorted([d for d in project_dir.iterdir() if d.is_dir()])
        for pole_dir in pole_dirs:
            # 이미지 파일이 있는 전주만 포함
            image_files = list(pole_dir.glob("*_OUT_processed_2d_plot.png"))
            if image_files:
                all_pole_dirs.append(pole_dir)
    
    if not all_pole_dirs:
        print(f"경고: 이미지 파일이 있는 전주를 찾을 수 없습니다.")
        return
    
    # 미저장 전주를 먼저 정렬
    def has_unsaved_images(pole_dir: Path) -> bool:
        """전주에 미저장 이미지가 있는지 확인 (ROI 정보 유무로 판단)"""
        project_name = pole_dir.parent.name
        poleid = pole_dir.name
        crop_data_pole_dir = EDIT_DATA_SAVE_DIR / "break" / project_name / poleid
        
        all_images = list(pole_dir.glob("*_OUT_processed_2d_plot.png"))
        for img_file in all_images:
            crop_data_roi_info_file = crop_data_pole_dir / img_file.name.replace("_2d_plot.png", "_roi_info.json")
            if crop_data_roi_info_file.exists():
                try:
                    info = load_break_info(str(crop_data_roi_info_file))
                    is_deleted = info.get('deleted', False)
                    if not is_deleted and not roi_info_has_saved_roi(info):
                        return True
                except:
                    return True
            else:
                return True
        return False
    
    # 미저장 전주와 저장 완료 전주 분리
    unsaved_poles = []
    saved_poles = []
    
    for pole_dir in all_pole_dirs:
        if has_unsaved_images(pole_dir):
            unsaved_poles.append(pole_dir)
        else:
            saved_poles.append(pole_dir)
    
    # 미저장 전주를 먼저, 그 다음 저장 완료 전주를 순서대로 정렬
    all_pole_dirs = unsaved_poles + saved_poles
    
    print(f"입력 디렉토리: {base_dir}")
    print(f"총 전주 수: {len(all_pole_dirs)}개")
    print(f"  - 미저장 전주: {len(unsaved_poles)}개")
    print(f"  - 저장 완료 전주: {len(saved_poles)}개")
    print(f"\nGUI 창에서 전주 간 이동이 가능합니다.\n")
    
    # 가장 상단 미저장 이미지로 GUI 시작 (없으면 첫 전주 첫 이미지)
    start_pole, start_image_idx = find_first_unsaved_image(all_pole_dirs)
    pole_idx = all_pole_dirs.index(start_pole) if start_pole in all_pole_dirs else 0
    root = tk.Tk()
    app = ROIEditorGUI(root, start_pole, all_pole_dirs, pole_idx, initial_image_idx=start_image_idx)
    root.mainloop()
    
    # 결과 통계 (실제로는 GUI 내에서 처리되므로 간단한 메시지만)
    print(f"\n{'='*60}")
    print(f"GUI 창이 닫혔습니다.")
    print(f"  총 전주 수: {len(all_pole_dirs)}개")
    print(f"{'='*60}")


def copy_confirmed_files_to_edit_data(pole_dir: Path, confirmed_break_info: dict, img_filename: str = None):
    """
    확정된 파일을 4. merge_data에서 읽어서 5. edit_data에 저장
    
    Args:
        pole_dir: 4. merge_data의 전주 디렉토리 경로
        confirmed_break_info: 확정된 파단 정보 (breakheight, breakdegree 포함)
        img_filename: 확정할 이미지 파일명 (None이면 전주의 모든 파일 처리)
    """
    poleid = pole_dir.name
    project_name = pole_dir.parent.name
    
    # 입력 디렉토리 (4. merge_data)
    merge_data_dir = Path(current_dir) / "4. merge_data" / "break"
    source_pole_dir = merge_data_dir / project_name / poleid
    
    # 출력 디렉토리 (make_ai\5. edit_data)
    edit_data_dir = EDIT_DATA_SAVE_DIR / "break"
    output_pole_dir = edit_data_dir / project_name / poleid
    output_pole_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_pole_dir.exists():
        return
    
    # 특정 이미지 파일만 처리하는 경우
    if img_filename:
        # 이미지 파일명에서 CSV 파일명 추출
        csv_filename = img_filename.replace("_2d_plot.png", ".csv")
        csv_file = source_pole_dir / csv_filename
        
        if csv_file.exists():
            output_csv = output_pole_dir / csv_file.name
            if not output_csv.exists():
                shutil.copy2(csv_file, output_csv)
            
            # break_info.json 생성 (confirmed 미사용)
            break_info_file = output_pole_dir / csv_file.name.replace(".csv", "_break_info.json")
            break_info = {
                'poleid': poleid,
                'project_name': project_name,
                'breakstate': 'B',
                'breakheight': confirmed_break_info.get('breakheight'),
                'breakdegree': confirmed_break_info.get('breakdegree')
            }
            save_break_info(str(break_info_file), break_info)
            
            # 이미지 파일도 복사 (있는 경우)
            img_file = source_pole_dir / img_filename
            if img_file.exists():
                output_img = output_pole_dir / img_file.name
                if not output_img.exists():
                    shutil.copy2(img_file, output_img)
        return
    
    # img_filename이 None인 경우 (기존 동작: 전주의 모든 파일 처리)
    # 이미 저장된 파일이 있으면 건너뛰기
    if output_pole_dir.exists():
        csv_files = list(output_pole_dir.glob("*_OUT_processed.csv"))
        if len(csv_files) > 0:
            return
    
    # CSV 파일 복사
    csv_files = list(source_pole_dir.glob("*_OUT_processed.csv"))
    for csv_file in csv_files:
        output_csv = output_pole_dir / csv_file.name
        if not output_csv.exists():
            shutil.copy2(csv_file, output_csv)
        
        # break_info.json 생성 (confirmed 미사용)
        break_info_file = output_pole_dir / csv_file.name.replace(".csv", "_break_info.json")
        break_info = {
            'poleid': poleid,
            'project_name': project_name,
            'breakstate': 'B',
            'breakheight': confirmed_break_info.get('breakheight'),
            'breakdegree': confirmed_break_info.get('breakdegree')
        }
        save_break_info(str(break_info_file), break_info)
        
        # 이미지 파일도 복사 (있는 경우)
        img_file = source_pole_dir / csv_file.name.replace(".csv", "_2d_plot.png")
        if img_file.exists():
            output_img = output_pole_dir / img_file.name
            if not output_img.exists():
                shutil.copy2(img_file, output_img)


def main():
    input_dir = os.path.join(current_dir, "4. merge_data/break")
    
    print("=" * 60)
    print("파단 위치 수정 시작")
    print("=" * 60)
    print(f"입력 디렉토리: {input_dir}")
    
    edit_all_poles(input_dir)


if __name__ == "__main__":
    main()
