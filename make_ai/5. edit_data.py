#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지를 보면서 break_info.json의 파단 위치(breakheight, breakdegree)를 수정하는 프로그램
GUI 버전 - 마우스로 수정 가능
"""

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
import matplotlib
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

# plot_processed_csv_2d 모듈 임포트 (같은 디렉토리)
sys.path.insert(0, current_dir)
from plot_processed_csv_2d import plot_csv_2d as regenerate_image


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


class BreakInfoEditorGUI:
    """파단 정보 편집 GUI"""
    
    def __init__(self, root: tk.Tk, pole_dir: Path, all_pole_dirs: List[Path] = None, current_pole_idx: int = 0):
        self.root = root
        self.pole_dir = pole_dir
        self.poleid = pole_dir.name
        
        # 전주 목록 관리 (전주 간 이동용)
        self.all_pole_dirs = all_pole_dirs if all_pole_dirs else [pole_dir]
        self.current_pole_idx = current_pole_idx
        
        # 프로젝트별 전주 목록 구성
        self.project_pole_map = {}
        for pole_dir_path in self.all_pole_dirs:
            project_name = pole_dir_path.parent.name
            if project_name not in self.project_pole_map:
                self.project_pole_map[project_name] = []
            self.project_pole_map[project_name].append(pole_dir_path)
        
        # 5. edit_data 디렉토리 경로 설정
        project_name = pole_dir.parent.name
        edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
        self.edit_data_pole_dir = edit_data_dir / project_name / self.poleid
        
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
            
            # 확정 여부와 삭제 여부는 5. edit_data에서 확인
            is_confirmed = False
            is_deleted = False
            edit_data_break_info_file = None
            if self.edit_data_pole_dir.exists():
                edit_data_break_info_file = self.edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                if edit_data_break_info_file.exists():
                    info = load_break_info(str(edit_data_break_info_file))
                    if info.get('confirmed') is True:
                        is_confirmed = True
                    if info.get('deleted') is True:
                        is_deleted = True
            
            # 4. merge_data의 break_info 파일도 유지 (작업용)
            if break_info_file.exists():
                self.break_info_files[img_file] = break_info_file
            
            # 모든 이미지 목록에 추가 (확정 여부 포함)
            self.all_image_list.append({
                'file': img_file,
                'break_info_file': break_info_file if break_info_file.exists() else None,
                'confirmed': is_confirmed
            })
            # 삭제되지 않은 이미지는 모두 self.image_files에 추가 (확정된 것도 포함)
            if not is_deleted:
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
        
        # 현재 파단 정보 파일 경로 (첫 번째 이미지 기준)
        self.current_break_info_file = None
        if self.image_files:
            self.current_break_info_file = self.break_info_files.get(self.image_files[0])
        
        # break_info 읽기 (첫 번째 이미지 기준)
        if self.current_break_info_file:
            self.break_info = load_break_info(str(self.current_break_info_file))
        else:
            self.break_info = {}
        
        # 현재 파단 위치 정보
        self.current_height = self.break_info.get('breakheight', 0.0)
        self.current_degree = self.break_info.get('breakdegree', 0.0)
        
        # 현재 CSV 데이터 범위 저장 (마우스 클릭 좌표 변환용)
        self.current_csv_data = None  # (heights, degrees) 튜플 저장
        
        # GUI 초기화
        self.setup_gui()
        
        # 첫 번째 이미지 로드 (이미지가 있으면 무조건 로드, 확정 여부와 관계없이)
        if self.image_files:
            self.load_image(0)
        elif all_images:
            # 이미지 파일은 있지만 모두 삭제된 경우
            self.ax.clear()
            self.ax.text(0.5, 0.5, '이 전주의 모든 이미지가 삭제되었습니다.', 
                        ha='center', va='center', fontsize=14, transform=self.ax.transAxes)
            self.canvas.draw()
        else:
            # 이미지 파일 자체가 없는 경우
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
                    
                    regenerate_image(
                        str(csv_file),
                        None,
                        str(break_info_file)
                    )
                except Exception as e:
                    print(f"  경고: 이미지 생성 중 오류 발생 - {e}")
            # 이미지 목록 다시 불러오기 (확정되지 않은 것만)
            all_images = sorted(list(self.pole_dir.glob("*_OUT_processed_2d_plot.png")))
            self.break_info_files = {}
            self.image_files = []
            for img_file in all_images:
                break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
                
                # 확정 여부와 삭제 여부는 5. edit_data에서 확인
                is_confirmed = False
                is_deleted = False
                if hasattr(self, 'edit_data_pole_dir') and self.edit_data_pole_dir.exists():
                    edit_data_break_info_file = self.edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                    if edit_data_break_info_file.exists():
                        info = load_break_info(str(edit_data_break_info_file))
                        if info.get('confirmed') is True:
                            is_confirmed = True
                        if info.get('deleted') is True:
                            is_deleted = True
                
                # 확정되었거나 삭제된 이미지는 건너뜀
                if is_confirmed or is_deleted:
                    continue
                
                # break_info 파일이 있으면 추가
                if break_info_file.exists():
                    self.break_info_files[img_file] = break_info_file
                # break_info 파일이 없으면 확정되지 않은 것으로 간주하여 포함
                self.image_files.append(img_file)
            
            # current_image_idx가 범위를 벗어나면 조정
            if self.image_files and self.current_image_idx >= len(self.image_files):
                self.current_image_idx = 0
    
    def setup_gui(self):
        """GUI 구성 요소 설정"""
        self.root.title(f"파단 위치 수정 - {self.poleid}")
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
        
        # 마우스 클릭 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self.on_image_click)
        
        # 컨트롤 프레임
        control_frame = ttk.LabelFrame(main_frame, text="파단 위치 입력", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 높이 입력
        ttk.Label(control_frame, text="파단 높이 (m):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.height_var = tk.StringVar(value=str(self.current_height))
        self.height_entry = ttk.Entry(control_frame, textvariable=self.height_var, width=15)
        self.height_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        # Entry에서 키 이벤트 바인딩
        self.height_entry.bind('<Key>', self.on_entry_key_press)
        
        # 각도 입력
        ttk.Label(control_frame, text="파단 각도 (°):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.degree_var = tk.StringVar(value=str(self.current_degree))
        self.degree_entry = ttk.Entry(control_frame, textvariable=self.degree_var, width=15)
        self.degree_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        # Entry에서 키 이벤트 바인딩
        self.degree_entry.bind('<Key>', self.on_entry_key_press)
        
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
        
        # 저장 버튼
        self.save_btn = ttk.Button(button_frame, text="💾 저장", command=self.save_data)
        self.save_btn.grid(row=0, column=action_button_start_col, padx=5)
        
        # 확정 버튼 (다음에 이 이미지를 다시 보여주지 않음)
        self.confirm_btn = ttk.Button(button_frame, text="✅ 확정", command=self.confirm_image)
        self.confirm_btn.grid(row=0, column=action_button_start_col + 1, padx=5)
        
        # 삭제 버튼
        self.delete_btn = ttk.Button(button_frame, text="🗑️ 삭제", command=self.delete_pole, style="Danger.TButton")
        self.delete_btn.grid(row=0, column=action_button_start_col + 2, padx=5)
        
        # 닫기 버튼
        ttk.Button(button_frame, text="✖ 닫기", command=self.close_window).grid(row=0, column=action_button_start_col + 3, padx=5)
        
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
        self.image_list_tree = ttk.Treeview(list_tree_frame, columns=("상태", "높이", "각도"), show="tree headings", height=30, yscrollcommand=list_scrollbar.set)
        self.image_list_tree.heading("#0", text="파일명")
        self.image_list_tree.heading("상태", text="상태")
        self.image_list_tree.heading("높이", text="높이(m)")
        self.image_list_tree.heading("각도", text="각도(°)")
        self.image_list_tree.column("#0", width=200)
        self.image_list_tree.column("상태", width=60)
        self.image_list_tree.column("높이", width=80)
        self.image_list_tree.column("각도", width=80)
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
            
            # 프로젝트 내 모든 전주가 확정되었는지 확인 (한 번만)
            project_all_confirmed = True
            pole_confirmed_map = {}  # 전주별 확정 여부 캐싱
            
            edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
            
            for pole_dir_path in pole_dirs:
                poleid = pole_dir_path.name
                all_images = sorted(list(pole_dir_path.glob("*_OUT_processed_2d_plot.png")))
                
                if not all_images:
                    pole_confirmed_map[poleid] = False
                    project_all_confirmed = False
                    continue
                
                # 전주의 모든 이미지가 확정되었는지 확인
                pole_all_confirmed = True
                edit_data_pole_dir = edit_data_dir / project_name / poleid
                
                for img_file in all_images:
                    edit_data_break_info_file = edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                    if edit_data_break_info_file.exists():
                        try:
                            info = load_break_info(str(edit_data_break_info_file))
                            is_confirmed = info.get('confirmed', False)
                            is_deleted = info.get('deleted', False)
                            if not is_confirmed or is_deleted:
                                pole_all_confirmed = False
                                break
                        except:
                            pole_all_confirmed = False
                            break
                    else:
                        pole_all_confirmed = False
                        break
                
                pole_confirmed_map[poleid] = pole_all_confirmed
                if not pole_all_confirmed:
                    project_all_confirmed = False
            
            # 프로젝트 아이템 추가 (확정 여부 표시)
            project_text = f"📁 {project_name}"
            if project_all_confirmed and len(pole_dirs) > 0:
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
                
                # 캐시된 확정 여부 사용
                pole_all_confirmed = pole_confirmed_map.get(poleid, False)
                
                # 전주 아이템 추가 (확정 여부 표시)
                pole_text = f"  📂 {poleid}"
                if pole_all_confirmed and all_images:
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
                    edit_data_pole_dir = edit_data_dir / project_name / poleid
                    
                    for img_file in all_images:
                        # 확정 여부와 삭제 여부는 5. edit_data에서 확인
                        is_confirmed = False
                        is_deleted = False
                        break_height = ""
                        break_degree = ""
                        
                        if edit_data_pole_dir.exists():
                            edit_data_break_info_file = edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                            if edit_data_break_info_file.exists():
                                try:
                                    info = load_break_info(str(edit_data_break_info_file))
                                    is_confirmed = info.get('confirmed', False)
                                    is_deleted = info.get('deleted', False)
                                    break_height = f"{info.get('breakheight', 0.0):.3f}" if info.get('breakheight') is not None else ""
                                    break_degree = f"{info.get('breakdegree', 0.0):.1f}" if info.get('breakdegree') is not None else ""
                                except:
                                    pass
                        
                        # 상태 결정
                        if is_deleted:
                            status = "삭제됨"
                        elif is_confirmed:
                            status = "확정"
                        else:
                            status = "미확정"
                        
                        file_name = img_file.name
                        img_item_id = self.image_list_tree.insert(pole_id, "end", text=f"    📄 {file_name}", values=(status, break_height, break_degree))
                        
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
        edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
        edit_data_pole_dir = edit_data_dir / project_name / poleid
        is_current_pole = (target_pole_dir == self.pole_dir)
        
        for img_file in all_images:
            is_confirmed = False
            is_deleted = False
            break_height = ""
            break_degree = ""
            
            if edit_data_pole_dir.exists():
                edit_data_break_info_file = edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                if edit_data_break_info_file.exists():
                    try:
                        info = load_break_info(str(edit_data_break_info_file))
                        is_confirmed = info.get('confirmed', False)
                        is_deleted = info.get('deleted', False)
                        break_height = f"{info.get('breakheight', 0.0):.3f}" if info.get('breakheight') is not None else ""
                        break_degree = f"{info.get('breakdegree', 0.0):.1f}" if info.get('breakdegree') is not None else ""
                    except:
                        pass
            
            if is_deleted:
                status = "삭제됨"
            elif is_confirmed:
                status = "확정"
            else:
                status = "미확정"
            
            file_name = img_file.name
            img_item_id = self.image_list_tree.insert(pole_item, "end", text=f"    📄 {file_name}", values=(status, break_height, break_degree))
            
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
        edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
        
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
        
        # 확정된 이미지는 선택할 수 없도록 (미확정 이미지만)
        if values and values[0] == "확정":
            return
        
        # self.image_files에서 해당 이미지의 인덱스 찾기
        if img_file in self.image_files:
            idx = self.image_files.index(img_file)
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
        
        # 현재 이미지에 대응하는 파단 정보 파일 로드
        self.current_break_info_file = self.break_info_files.get(img_file)
        
        # 파단 정보 파일이 없으면 생성
        if not self.current_break_info_file:
            # 이미지 파일명에서 파단 정보 파일 경로 생성
            break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
            # 기본 정보로 파일 생성
            poleid = self.poleid
            project_name = self.pole_dir.parent.name
            default_info = {
                'poleid': poleid,
                'project_name': project_name,
                'breakstate': 'B',
                'breakheight': 0.0,
                'breakdegree': 0.0
            }
            if save_break_info(str(break_info_file), default_info):
                self.current_break_info_file = break_info_file
                self.break_info_files[img_file] = break_info_file
        
        # 파단 정보 로드
        if self.current_break_info_file:
            self.break_info = load_break_info(str(self.current_break_info_file))
            self.current_height = self.break_info.get('breakheight', 0.0)
            self.current_degree = self.break_info.get('breakdegree', 0.0)
            
            # 입력 필드 업데이트
            self.height_var.set(str(self.current_height))
            self.degree_var.set(str(self.current_degree))
        
        try:
            img = mpimg.imread(str(img_file))
            self.ax.clear()
            self.ax.imshow(img)
            
            # CSV 파일 읽어서 데이터 범위 파악 (마우스 클릭 좌표 변환용)
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
            
            self.ax.axis('off')
            self.ax.set_title(f"{self.poleid} - 이미지 {idx + 1}/{len(self.image_files)}\n파일: {img_file.name}\n(이미지 클릭으로 파단 위치 설정)", 
                            fontsize=12, fontweight='bold', pad=10)
            self.fig.tight_layout()
            self.canvas.draw()
            
            # 이미지 목록 업데이트
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            
            # 이전/다음 버튼 상태 업데이트 (전주 포함)
            self.prev_btn.config(state=tk.NORMAL if self.has_previous_image() else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.has_next_image() else tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
    
    def on_image_click(self, event):
        """이미지 클릭 이벤트 핸들러"""
        if event.inaxes != self.ax or self.current_csv_data is None:
            return
        
        # 클릭한 위치의 픽셀 좌표 (imshow는 기본적으로 픽셀 좌표 사용)
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        # 이미지 크기
        img_height = self.current_csv_data['img_shape'][0]
        img_width = self.current_csv_data['img_shape'][1]
        
        # 데이터 범위
        height_min = self.current_csv_data['height_min']
        height_max = self.current_csv_data['height_max']
        degree_min = self.current_csv_data['degree_min']
        degree_max = self.current_csv_data['degree_max']
        
        # 이미지 구조: 3개의 서브플롯이 가로로 나열 (subplots(1, 3, figsize=(18, 6)))
        # 각 서브플롯은 동일한 데이터 범위 (degree_min~degree_max, height_min~height_max) 사용
        # tight_layout()과 bbox_inches='tight'로 저장되어 여백이 최소화됨
        
        # imshow의 기본 좌표계: x는 -0.5 ~ (width-0.5), y는 (height-0.5) ~ -0.5
        # 픽셀 좌표를 0~1 범위로 정규화 (이미지 좌상단이 (0,0), 우하단이 (1,1))
        norm_x = (click_x + 0.5) / img_width if img_width > 0 else 0.5
        norm_y = (img_height - 0.5 - click_y) / img_height if img_height > 0 else 0.5
        
        # 이미지가 3개의 서브플롯으로 구성되어 있고, 각 서브플롯이 대략 1/3 너비를 차지
        # 클릭한 위치가 어느 서브플롯인지 판단 (0, 1, 2)
        subplot_idx = int(norm_x * 3)
        subplot_idx = max(0, min(2, subplot_idx))
        
        # 클릭한 서브플롯 내에서의 상대 위치 (0~1)
        local_x_in_subplot = (norm_x * 3) % 1.0
        if subplot_idx == 2 and local_x_in_subplot > 0.99:
            local_x_in_subplot = 1.0
        local_x_in_subplot = max(0.0, min(1.0, local_x_in_subplot))
        
        # tight_layout과 bbox_inches='tight'로 저장된 이미지의 실제 플롯 영역 추정
        # 각 서브플롯마다 컬러바가 오른쪽에 있으므로 우측 여백을 더 크게 설정
        # 상단 여백 (제목 포함): 약 12%
        # 하단 여백 (레이블 포함): 약 10%
        # 좌측 여백 (레이블 + 약간의 여유): 약 8%
        # 우측 여백 (컬러바 + 약간의 여유): 약 15% (컬러바가 서브플롯 내부에 있음)
        # 서브플롯 간 간격: 약 2% (각 서브플롯이 약 32% 너비)
        
        plot_margin_top = 0.12
        plot_margin_bottom = 0.10
        
        # 서브플롯 내 플롯 영역 계산 (각 서브플롯의 실제 플롯 영역)
        # 컬러바가 각 서브플롯 오른쪽에 있으므로 우측 여백을 더 크게 설정
        subplot_margin_left = 0.08
        subplot_margin_right = 0.15  # 컬러바를 고려하여 증가
        
        # 서브플롯 내 플롯 영역의 상대 위치 계산 (0~1)
        plot_x_in_subplot = (local_x_in_subplot - subplot_margin_left) / (1 - subplot_margin_left - subplot_margin_right)
        plot_x_in_subplot = max(0.0, min(1.0, plot_x_in_subplot))
        
        # 전체 이미지 높이 기준 플롯 영역 (0~1)
        plot_y = (norm_y - plot_margin_bottom) / (1 - plot_margin_top - plot_margin_bottom)
        plot_y = max(0.0, min(1.0, plot_y))
        
        # 데이터 값으로 변환 (모든 서브플롯이 동일한 데이터 범위 사용)
        # contourf는 x축이 degree, y축이 height
        clicked_degree = degree_min + plot_x_in_subplot * (degree_max - degree_min)
        clicked_height = height_min + plot_y * (height_max - height_min)
        
        # 입력 필드 업데이트 (높이는 0.001 단위까지)
        self.height_var.set(f"{clicked_height:.3f}")
        self.degree_var.set(f"{clicked_degree:.1f}")
        
        # 클릭 시 자동으로 저장하고 이미지 재생성
        self.save_and_refresh_image()

    def disable_ui(self):
        """UI 비활성화 (로딩 중)"""
        self.root.config(cursor="wait")
        if hasattr(self, 'confirm_btn'):
            self.confirm_btn.config(state=tk.DISABLED)
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
        if hasattr(self, 'confirm_btn'):
            self.confirm_btn.config(state=tk.NORMAL)
        if hasattr(self, 'delete_btn'):
            self.delete_btn.config(state=tk.NORMAL)
        if hasattr(self, 'prev_btn'):
            self.prev_btn.config(state=tk.NORMAL if self.has_previous_image() else tk.DISABLED)
        if hasattr(self, 'next_btn'):
            self.next_btn.config(state=tk.NORMAL if self.has_next_image() else tk.DISABLED)
        if hasattr(self, 'save_btn'):
            self.save_btn.config(state=tk.NORMAL)
        # Entry 필드 활성화
        if hasattr(self, 'height_entry'):
            self.height_entry.config(state=tk.NORMAL)
        if hasattr(self, 'degree_entry'):
            self.degree_entry.config(state=tk.NORMAL)
        self.root.update()
    
    def confirm_image(self):
        """현재 이미지를 확정 처리하여 이후에 다시 표시하지 않음"""
        if not self.image_files or self.current_image_idx >= len(self.image_files):
            return
        
        # UI 비활성화
        self.disable_ui()
        
        try:
            current_img_file = self.image_files[self.current_image_idx]
            break_info_file = self.break_info_files.get(current_img_file)
            
            # 현재 입력값으로 파단 위치 정보 업데이트
            try:
                current_height = float(self.height_var.get())
                current_degree = float(self.degree_var.get())
            except ValueError:
                current_height = self.current_height
                current_degree = self.current_degree
            
            # break_info에 confirmed 플래그 및 파단 위치 정보 저장
            if break_info_file and break_info_file.exists():
                info = load_break_info(str(break_info_file))
                info['confirmed'] = True
                info['breakheight'] = current_height
                info['breakdegree'] = current_degree
                save_break_info(str(break_info_file), info)
            elif current_img_file:
                # break_info 파일이 없으면 생성
                break_info_file = current_img_file.parent / current_img_file.name.replace("_2d_plot.png", "_break_info.json")
                if self.break_info:
                    info = self.break_info.copy()
                else:
                    info = {
                        'poleid': self.poleid,
                        'project_name': self.pole_dir.parent.name,
                        'breakstate': 'B',
                        'breakheight': current_height,
                        'breakdegree': current_degree
                    }
                info['confirmed'] = True
                info['breakheight'] = current_height
                info['breakdegree'] = current_degree
                save_break_info(str(break_info_file), info)
            
            # 확정된 파일을 5. edit_data에 저장 (현재 이미지만)
            confirmed_info = {
                'breakheight': current_height,
                'breakdegree': current_degree
            }
            # 현재 이미지 파일명 전달 (해당 파일만 확정 처리)
            img_filename = current_img_file.name
            copy_confirmed_files_to_edit_data(self.pole_dir, confirmed_info, img_filename)
            
            # 현재 이미지 목록에서 제거
            try:
                del self.break_info_files[current_img_file]
            except KeyError:
                pass
            
            del self.image_files[self.current_image_idx]
            
            # 정보 라벨 업데이트
            if hasattr(self, 'image_count_label'):
                self.image_count_label.config(text=f"이미지 수: {len(self.image_files)}개")
            
            # 이미지 목록 업데이트
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            
            # 더 이상 이미지가 없으면 다음 전주로 자동 이동 시도
            if not self.image_files:
                # 다음 전주가 있으면 자동으로 이동
                if len(self.all_pole_dirs) > 1 and self.current_pole_idx < len(self.all_pole_dirs) - 1:
                    # 다음 전주로 자동 이동
                    self.current_pole_idx += 1
                    self.load_pole(self.all_pole_dirs[self.current_pole_idx])
                    return
                else:
                    # 모든 전주를 확인했거나 단일 전주 모드
                    messagebox.showinfo("알림", "확정되지 않은 이미지가 없습니다.")
                    if hasattr(self, 'prev_btn'):
                        self.prev_btn.config(state=tk.DISABLED)
                    if hasattr(self, 'next_btn'):
                        self.next_btn.config(state=tk.DISABLED)
                    # 이미지가 없으므로 화면 비우기
                    self.ax.clear()
                    self.canvas.draw()
                    return
            
            # 인덱스 조정: 현재 이미지가 삭제되었으므로 같은 인덱스는 다음 이미지가 됨
            # 하지만 인덱스가 범위를 벗어나면 마지막 이미지로 조정
            if self.current_image_idx >= len(self.image_files):
                self.current_image_idx = len(self.image_files) - 1
            
            # 다음 이미지 로드
            if self.current_image_idx >= 0 and self.current_image_idx < len(self.image_files):
                self.load_image(self.current_image_idx)
        finally:
            # UI 활성화
            self.enable_ui()
    
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
        """전주 전환 전 저장 확인"""
        # 입력값이 변경되었는지 확인
        try:
            height_changed = abs(float(self.height_var.get()) - self.current_height) > 0.001
            degree_changed = abs(float(self.degree_var.get()) - self.current_degree) > 0.001
            
            if height_changed or degree_changed:
                response = messagebox.askyesnocancel(
                    "저장 확인",
                    "파단 위치 값이 변경되었습니다.\n저장하시겠습니까?\n\n예: 저장 후 이동\n아니오: 저장하지 않고 이동\n취소: 이동 취소"
                )
                
                if response is None:  # 취소
                    return False
                elif response:  # 예 - 저장
                    self.save_data()
                    return True
                else:  # 아니오 - 저장하지 않고 이동
                    return True
        except ValueError:
            pass
        
        return True
    
    def load_pole(self, pole_dir: Path):
        """전주 데이터 로드"""
        self.pole_dir = pole_dir
        self.poleid = pole_dir.name
        
        # 5. edit_data 디렉토리 경로 설정
        project_name = pole_dir.parent.name
        edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
        self.edit_data_pole_dir = edit_data_dir / project_name / self.poleid
        
        # 전주 인덱스 업데이트
        if pole_dir in self.all_pole_dirs:
            self.current_pole_idx = self.all_pole_dirs.index(pole_dir)
        
        # 이미지 파일 목록 (확정되지 않은 이미지만)
        all_images = sorted(list(pole_dir.glob("*_OUT_processed_2d_plot.png")))
        self.break_info_files = {}
        self.image_files = []
        for img_file in all_images:
            break_info_file = img_file.parent / img_file.name.replace("_2d_plot.png", "_break_info.json")
            
            # 삭제 여부는 5. edit_data에서 확인
            is_deleted = False
            if self.edit_data_pole_dir.exists():
                edit_data_break_info_file = self.edit_data_pole_dir / img_file.name.replace("_2d_plot.png", "_break_info.json")
                if edit_data_break_info_file.exists():
                    info = load_break_info(str(edit_data_break_info_file))
                    if info.get('deleted') is True:
                        is_deleted = True
            
            # 삭제된 이미지만 건너뜀 (확정된 이미지는 포함)
            if is_deleted:
                continue
            
            # break_info 파일이 있으면 추가
            if break_info_file.exists():
                self.break_info_files[img_file] = break_info_file
            # 삭제되지 않은 이미지는 모두 포함 (확정 여부와 관계없이)
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
        
        # 현재 파단 정보 파일 경로 (첫 번째 이미지 기준)
        self.current_break_info_file = None
        if self.image_files:
            self.current_break_info_file = self.break_info_files.get(self.image_files[0])
        
        # break_info 읽기 (첫 번째 이미지 기준)
        if self.current_break_info_file:
            self.break_info = load_break_info(str(self.current_break_info_file))
        else:
            self.break_info = {}
        
        # 현재 파단 위치 정보
        self.current_height = self.break_info.get('breakheight', 0.0)
        self.current_degree = self.break_info.get('breakdegree', 0.0)
        
        # 입력 필드 업데이트
        self.height_var.set(str(self.current_height))
        self.degree_var.set(str(self.current_degree))
        
        # GUI 제목 업데이트
        self.root.title(f"파단 위치 수정 - {self.poleid}")
        
        # 정보 라벨 업데이트
        if hasattr(self, 'poleid_label'):
            self.poleid_label.config(text=f"전주ID: {self.poleid}")
        if hasattr(self, 'image_count_label'):
            self.image_count_label.config(text=f"이미지 수: {len(self.image_files)}개")
        
        # 이미지 목록 업데이트
        if hasattr(self, 'image_list_tree'):
            self.update_image_list()
        
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
                # 이미지 파일 자체가 없는 경우
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
        """클릭 시 자동 저장 및 이미지 재생성"""
        # UI 비활성화
        self.disable_ui()
        
        try:
            # 입력값 검증
            new_height = float(self.height_var.get())
            new_degree = float(self.degree_var.get())
            
            # 현재 이미지에 대응하는 파단 정보 파일 확인
            if not self.current_break_info_file:
                return
            
            # break_info 업데이트
            if not self.break_info:
                # 기본 정보 설정
                self.break_info = {
                    'poleid': self.poleid,
                    'project_name': self.pole_dir.parent.name,
                    'breakstate': 'B'
                }
            
            self.break_info['breakheight'] = new_height
            self.break_info['breakdegree'] = new_degree
            
            # 저장
            if save_break_info(str(self.current_break_info_file), self.break_info):
                # 내부 상태 업데이트
                self.current_height = new_height
                self.current_degree = new_degree
                self.saved = True
                
                # 저장된 파단 위치를 반영하여 현재 이미지만 자동 재생성
                if self.image_files and self.current_image_idx < len(self.image_files):
                    current_img_file = self.image_files[self.current_image_idx]
                    # 이미지 파일명: {poleid}_{measno}_OUT_processed_2d_plot.png
                    # CSV 파일명: {poleid}_{measno}_OUT_processed.csv
                    csv_filename = current_img_file.name.replace("_2d_plot.png", ".csv")
                    csv_file = self.pole_dir / csv_filename
                    
                    if csv_file.exists():
                        try:
                            regenerate_image(
                                str(csv_file),
                                None,
                                str(self.current_break_info_file)
                            )
                            # 현재 이미지 다시 로드
                            self.load_image(self.current_image_idx)
                        except Exception:
                            pass  # 이미지 재생성 실패해도 무시
        except (ValueError, Exception):
            pass  # 오류 발생 시 무시
        finally:
            # UI 활성화
            self.enable_ui()
    
    def save_data(self):
        """데이터 저장"""
        # UI 비활성화
        self.disable_ui()
        
        try:
            # 입력값 검증
            new_height = float(self.height_var.get())
            new_degree = float(self.degree_var.get())
            
            # 현재 이미지에 대응하는 파단 정보 파일 확인
            if not self.current_break_info_file:
                messagebox.showerror("오류", "파단 정보 파일을 찾을 수 없습니다.")
                return
            
            # break_info 업데이트
            if not self.break_info:
                # 기본 정보 설정
                self.break_info = {
                    'poleid': self.poleid,
                    'project_name': self.pole_dir.parent.name,
                    'breakstate': 'B'
                }
            
            self.break_info['breakheight'] = new_height
            self.break_info['breakdegree'] = new_degree
            
            # 저장
            if save_break_info(str(self.current_break_info_file), self.break_info):
                # 내부 상태 업데이트
                self.current_height = new_height
                self.current_degree = new_degree
                self.saved = True

                # 저장된 파단 위치를 반영하여 현재 이미지만 자동 재생성
                if self.image_files and self.current_image_idx < len(self.image_files):
                    current_img_file = self.image_files[self.current_image_idx]
                    # 이미지 파일명: {poleid}_{measno}_OUT_processed_2d_plot.png
                    # CSV 파일명: {poleid}_{measno}_OUT_processed.csv
                    csv_filename = current_img_file.name.replace("_2d_plot.png", ".csv")
                    csv_file = self.pole_dir / csv_filename
                    
                    if csv_file.exists():
                        try:
                            regenerate_image(
                                str(csv_file),
                                None,
                                str(self.current_break_info_file)
                            )
                            # 현재 이미지 다시 로드
                            self.load_image(self.current_image_idx)
                            messagebox.showinfo(
                                "저장 완료",
                                f"파단 위치가 저장되고 이미지가 자동으로 재설정되었습니다.\n"
                                f"높이: {new_height}m\n각도: {new_degree}°"
                            )
                        except Exception as e:
                            messagebox.showwarning(
                                "저장 완료 (이미지 오류)",
                                f"파단 위치는 저장되었지만 이미지 재생성 중 오류가 발생했습니다.\n"
                                f"오류: {e}"
                            )
                    else:
                        messagebox.showinfo(
                            "저장 완료",
                            f"파단 위치가 저장되었습니다.\n"
                            f"높이: {new_height}m\n각도: {new_degree}°\n"
                            f"대응하는 CSV 파일을 찾을 수 없어 이미지는 재설정되지 않았습니다."
                        )
                else:
                    messagebox.showinfo(
                        "저장 완료",
                        f"파단 위치가 저장되었습니다.\n"
                        f"높이: {new_height}m\n각도: {new_degree}°"
                    )
            else:
                messagebox.showerror("저장 실패", "파일 저장에 실패했습니다.")
                
        except ValueError:
            messagebox.showerror("입력 오류", "높이와 각도는 숫자여야 합니다.")
        finally:
            # UI 활성화
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
            
            # 5. edit_data의 break_info.json에 deleted 플래그 설정
            if hasattr(self, 'edit_data_pole_dir') and self.edit_data_pole_dir.exists():
                edit_data_break_info_file = self.edit_data_pole_dir / break_info_filename
                
                # break_info 파일이 있으면 deleted 플래그 추가, 없으면 생성
                if edit_data_break_info_file.exists():
                    info = load_break_info(str(edit_data_break_info_file))
                else:
                    info = {
                        'poleid': self.poleid,
                        'project_name': self.pole_dir.parent.name,
                        'breakstate': 'B',
                        'breakheight': self.current_height if hasattr(self, 'current_height') else 0.0,
                        'breakdegree': self.current_degree if hasattr(self, 'current_degree') else 0.0
                    }
                
                info['deleted'] = True
                info['confirmed'] = False  # 삭제된 것은 확정 해제
                save_break_info(str(edit_data_break_info_file), info)
            
            # 4. merge_data의 break_info.json에도 deleted 플래그 설정 (선택사항)
            break_info_file = self.pole_dir / break_info_filename
            if break_info_file.exists():
                info = load_break_info(str(break_info_file))
                info['deleted'] = True
                info['confirmed'] = False
                save_break_info(str(break_info_file), info)
            
            # 현재 이미지를 목록에서 제거
            try:
                del self.break_info_files[current_img_file]
            except KeyError:
                pass
            
            del self.image_files[self.current_image_idx]
            self.deleted = True
            
            # 정보 라벨 업데이트
            if hasattr(self, 'image_count_label'):
                self.image_count_label.config(text=f"이미지 수: {len(self.image_files)}개")
            
            # 이미지 목록 업데이트
            if hasattr(self, 'image_list_tree'):
                self.update_image_list()
            
            # 이미지가 더 이상 없으면 다음 전주로 자동 이동 시도
            if not self.image_files:
                # 다음 전주가 있으면 자동으로 이동
                if len(self.all_pole_dirs) > 1 and self.current_pole_idx < len(self.all_pole_dirs) - 1:
                    # 다음 전주로 자동 이동
                    self.current_pole_idx += 1
                    self.load_pole(self.all_pole_dirs[self.current_pole_idx])
                    return
                else:
                    # 모든 전주를 확인했거나 단일 전주 모드
                    messagebox.showinfo("알림", "이 전주의 모든 이미지가 삭제 처리되었습니다.")
                    if hasattr(self, 'prev_btn'):
                        self.prev_btn.config(state=tk.DISABLED)
                    if hasattr(self, 'next_btn'):
                        self.next_btn.config(state=tk.DISABLED)
                    # 이미지가 없으므로 화면 비우기
                    self.ax.clear()
                    self.canvas.draw()
                    return
            else:
                # 인덱스 조정: 현재 이미지가 삭제되었으므로 같은 인덱스는 다음 이미지가 됨
                # 하지만 인덱스가 범위를 벗어나면 마지막 이미지로 조정
                if self.current_image_idx >= len(self.image_files):
                    self.current_image_idx = len(self.image_files) - 1
                
                # 다음 이미지 로드
                self.load_image(self.current_image_idx)
            
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
    
    # 파단 정보 파일 또는 이미지 파일 찾기 (둘 중 하나라도 있으면 진행)
    break_info_files = list(pole_dir.glob("*_OUT_processed_break_info.json"))
    image_files = list(pole_dir.glob("*_OUT_processed_2d_plot.png"))
    csv_files = list(pole_dir.glob("*_OUT_processed.csv"))
    
    if not break_info_files and not image_files and not csv_files:
        print(f"  경고: {poleid} 전주에 파단 정보 파일, 이미지 파일, 또는 CSV 파일이 없습니다.")
        return 'error'
    
    # GUI 실행
    root = tk.Tk()
    app = BreakInfoEditorGUI(root, pole_dir)
    root.mainloop()
    
    return app.get_result() if hasattr(app, 'get_result') else 'cancelled'


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
    
    # 모든 전주 디렉토리 수집 (개별 파단 정보 파일이 있는 전주)
    all_pole_dirs = []
    for project_dir in sorted(projects):
        pole_dirs = sorted([d for d in project_dir.iterdir() if d.is_dir()])
        for pole_dir in pole_dirs:
            # 개별 파단 정보 파일, 이미지 파일, 또는 CSV 파일이 있으면 포함
            # 하나의 glob으로 모든 패턴을 검색 (더 빠름)
            all_files = list(pole_dir.glob("*_OUT_processed*"))
            if all_files:
                all_pole_dirs.append(pole_dir)
    
    if not all_pole_dirs:
        print(f"경고: 파단 정보 파일, 이미지 파일, 또는 CSV 파일이 있는 전주를 찾을 수 없습니다.")
        return
    
    print(f"입력 디렉토리: {base_dir}")
    print(f"총 전주 수: {len(all_pole_dirs)}개")
    print(f"\nGUI 창에서 전주 간 이동이 가능합니다.\n")
    
    # 첫 번째 전주로 GUI 시작
    root = tk.Tk()
    app = BreakInfoEditorGUI(root, all_pole_dirs[0], all_pole_dirs, 0)
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
    
    # 출력 디렉토리 (5. edit_data)
    edit_data_dir = Path(current_dir) / "5. edit_data" / "break"
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
            
            # break_info.json 생성 (확정 정보 포함)
            break_info_file = output_pole_dir / csv_file.name.replace(".csv", "_break_info.json")
            break_info = {
                'poleid': poleid,
                'project_name': project_name,
                'breakstate': 'B',
                'breakheight': confirmed_break_info.get('breakheight'),
                'breakdegree': confirmed_break_info.get('breakdegree'),
                'confirmed': True
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
        
        # break_info.json 생성 (확정 정보 포함)
        break_info_file = output_pole_dir / csv_file.name.replace(".csv", "_break_info.json")
        break_info = {
            'poleid': poleid,
            'project_name': project_name,
            'breakstate': 'B',
            'breakheight': confirmed_break_info.get('breakheight'),
            'breakdegree': confirmed_break_info.get('breakdegree'),
            'confirmed': True
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
