#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지를 보면서 break_info.json의 파단 위치(breakheight, breakdegree)를 수정하는 프로그램
"""

import os
import json
import argparse
import sys
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def edit_single_pole(pole_dir: Path):
    """단일 전주 폴더의 break_info.json 수정"""
    poleid = pole_dir.name
    
    # break_info.json 파일 찾기
    break_info_json = pole_dir / f"{poleid}_break_info.json"
    
    if not break_info_json.exists():
        print(f"  경고: {poleid}_break_info.json 파일이 없습니다.")
        return False
    
    # break_info.json 읽기
    break_info = load_break_info(str(break_info_json))
    
    if not break_info:
        print(f"  경고: break_info.json을 읽을 수 없습니다.")
        return False
    
    # 현재 파단 위치 정보
    current_height = break_info.get('breakheight')
    current_degree = break_info.get('breakdegree')
    
    print(f"\n{'='*60}")
    print(f"전주ID: {poleid}")
    print(f"{'='*60}")
    print(f"현재 파단 높이: {current_height}")
    print(f"현재 파단 각도: {current_degree}")
    
    # 이미지 파일 찾기 및 표시
    image_files = list(pole_dir.glob("*_2d_plot.png"))
    if image_files:
        # 첫 번째 이미지 파일 표시
        img_file = image_files[0]
        print(f"\n이미지 표시 중: {img_file.name}")
        
        try:
            # 이미지 읽기 및 표시
            img = mpimg.imread(str(img_file))
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis('off')
            plt.title(f"{poleid} - 파단 위치 확인", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # 이미지 표시 (비블로킹)
            plt.show(block=False)
            plt.pause(0.5)  # 이미지가 표시될 시간을 줌
            
            input("\n이미지를 확인하셨으면 엔터를 누르세요...")
            
            # 이미지 창 닫기
            plt.close('all')
        except Exception as e:
            print(f"  경고: 이미지 표시 중 오류 발생 - {e}")
            print(f"  이미지 경로: {img_file}")
    else:
        print(f"\n경고: 이미지 파일을 찾을 수 없습니다.")
    
    # 새 값 입력 받기 또는 삭제 옵션
    print(f"\n새로운 파단 위치를 입력하세요 (엔터만 누르면 현재 값 유지)")
    print(f"또는 'd' 또는 'delete'를 입력하면 이 전주 데이터를 삭제합니다")
    
    # 삭제 옵션 확인
    delete_choice = input(f"\n이 전주 데이터를 삭제하시겠습니까? (y/n, 또는 엔터로 수정 계속): ").strip().lower()
    if delete_choice in ['y', 'yes', 'd', 'delete']:
        confirm_delete = input(f"정말로 {poleid} 전주 데이터를 삭제하시겠습니까? (yes/no): ").strip().lower()
        if confirm_delete == 'yes':
            try:
                # 전주 폴더 전체 삭제
                shutil.rmtree(pole_dir)
                print(f"  ✓ 삭제 완료: {pole_dir}")
                return 'deleted'
            except Exception as e:
                print(f"  ✗ 삭제 실패: {e}")
                return False
        else:
            print(f"  → 삭제 취소")
    
    # 높이 입력
    new_height_str = input(f"파단 높이 (현재: {current_height}): ").strip()
    if new_height_str:
        try:
            new_height = float(new_height_str)
            break_info['breakheight'] = new_height
            print(f"  → 파단 높이를 {new_height}로 변경")
        except ValueError:
            print(f"  경고: 잘못된 숫자 형식입니다. 현재 값 유지")
    else:
        print(f"  → 파단 높이 유지: {current_height}")
    
    # 각도 입력
    new_degree_str = input(f"파단 각도 (현재: {current_degree}): ").strip()
    if new_degree_str:
        try:
            new_degree = float(new_degree_str)
            break_info['breakdegree'] = new_degree
            print(f"  → 파단 각도를 {new_degree}로 변경")
        except ValueError:
            print(f"  경고: 잘못된 숫자 형식입니다. 현재 값 유지")
    else:
        print(f"  → 파단 각도 유지: {current_degree}")
    
    # 저장 확인
    print(f"\n변경된 파단 위치:")
    print(f"  높이: {break_info.get('breakheight')}")
    print(f"  각도: {break_info.get('breakdegree')}")
    
    confirm = input("\n저장하시겠습니까? (y/n): ").strip().lower()
    if confirm == 'y':
        if save_break_info(str(break_info_json), break_info):
            print(f"  ✓ 저장 완료: {break_info_json}")
            
            # 이미지 재생성
            print(f"\n수정된 정보로 이미지를 재생성합니다...")
            csv_files = list(pole_dir.glob("*_processed.csv"))
            if csv_files:
                # 첫 번째 CSV 파일에 대해 이미지 재생성
                csv_file = csv_files[0]
                try:
                    regenerate_image(
                        str(csv_file),
                        None,  # output_path는 자동 생성
                        str(break_info_json)
                    )
                    print(f"  ✓ 이미지 재생성 완료")
                    
                    # 재생성된 이미지 표시
                    image_files = list(pole_dir.glob("*_2d_plot.png"))
                    if image_files:
                        img_file = image_files[0]
                        print(f"\n수정된 이미지를 확인하세요:")
                        try:
                            img = mpimg.imread(str(img_file))
                            fig, ax = plt.subplots(figsize=(12, 8))
                            ax.imshow(img)
                            ax.axis('off')
                            plt.title(f"{poleid} - 수정된 파단 위치 확인\n높이: {break_info.get('breakheight')}, 각도: {break_info.get('breakdegree')}", 
                                     fontsize=14, fontweight='bold', pad=20)
                            plt.tight_layout()
                            plt.show(block=False)
                            plt.pause(0.5)
                            
                            input("\n수정된 이미지를 확인하셨으면 엔터를 누르세요...")
                            plt.close('all')
                        except Exception as e:
                            print(f"  경고: 이미지 표시 중 오류 발생 - {e}")
                except Exception as e:
                    print(f"  경고: 이미지 재생성 중 오류 발생 - {e}")
            
            return True
        else:
            print(f"  ✗ 저장 실패")
            return False
    else:
        print(f"  → 저장 취소")
        return False


def edit_all_poles(base_dir: str):
    """모든 전주 폴더의 break_info.json 수정"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"오류: 디렉토리가 존재하지 않습니다: {base_dir}")
        return
    
    # 모든 프로젝트 디렉토리 찾기
    projects = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not projects:
        print(f"경고: 프로젝트 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"입력 디렉토리: {base_dir}")
    print(f"총 프로젝트 수: {len(projects)}개")
    print(f"\n각 전주 폴더를 순차적으로 확인합니다.")
    print(f"(프로젝트/전주ID 구조를 따라갑니다)")
    print(f"\n중단하려면 Ctrl+C를 누르세요.\n")
    
    total_processed = 0
    total_updated = 0
    total_deleted = 0
    
    try:
        for project_dir in sorted(projects):
            project_name = project_dir.name
            print(f"\n{'='*60}")
            print(f"프로젝트: {project_name}")
            print(f"{'='*60}")
            
            # 프로젝트 아래의 모든 전주 디렉토리 찾기
            pole_dirs = sorted([d for d in project_dir.iterdir() if d.is_dir()])
            
            for pole_idx, pole_dir in enumerate(pole_dirs):
                poleid = pole_dir.name
                break_info_json = pole_dir / f"{poleid}_break_info.json"
                
                if not break_info_json.exists():
                    continue
                
                total_processed += 1
                
                result = edit_single_pole(pole_dir)
                if result == 'deleted':
                    total_deleted += 1
                    # 다음 전주로 넘어갈지 확인
                    if pole_idx < len(pole_dirs) - 1:
                        continue_choice = input("\n다음 전주로 넘어가시겠습니까? (y/n/q: 종료): ").strip().lower()
                        if continue_choice == 'q':
                            print("\n작업을 중단합니다.")
                            return
                        elif continue_choice != 'y':
                            continue
                    continue
                elif result:
                    total_updated += 1
                
                # 다음 전주로 넘어갈지 확인 (마지막 전주가 아니면)
                if pole_idx < len(pole_dirs) - 1:
                    continue_choice = input("\n다음 전주로 넘어가시겠습니까? (y/n/q: 종료): ").strip().lower()
                    if continue_choice == 'q':
                        print("\n작업을 중단합니다.")
                        return
                    elif continue_choice != 'y':
                        print("다시 시도합니다.")
                        edit_single_pole(pole_dir)
                        continue_choice = input("\n다음 전주로 넘어가시겠습니까? (y/n/q: 종료): ").strip().lower()
                        if continue_choice == 'q':
                            return
        
    except KeyboardInterrupt:
        print("\n\n작업이 중단되었습니다.")
    
    print(f"\n{'='*60}")
    print(f"완료!")
    print(f"  처리된 전주 수: {total_processed}개")
    print(f"  수정된 전주 수: {total_updated}개")
    print(f"  삭제된 전주 수: {total_deleted}개")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="이미지를 보면서 break_info.json의 파단 위치 수정"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="4.1 select_pole",
        help="입력 디렉토리 (기본값: 4.1 select_pole)",
    )
    parser.add_argument(
        "--pole-dir",
        type=str,
        default=None,
        help="단일 전주 디렉토리만 처리 (예: 4.1 select_pole/프로젝트/전주ID)",
    )
    
    args = parser.parse_args()
    
    # 현재 디렉토리 기준으로 경로 변환
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(current_dir, args.input_dir)
    if args.pole_dir and not os.path.isabs(args.pole_dir):
        args.pole_dir = os.path.join(current_dir, args.pole_dir)
    
    if args.pole_dir:
        # 단일 전주 디렉토리 처리
        pole_path = Path(args.pole_dir)
        if not pole_path.exists():
            print(f"오류: 디렉토리가 존재하지 않습니다: {args.pole_dir}")
            return
        
        edit_single_pole(pole_path)
    else:
        # 전체 디렉토리 처리
        edit_all_poles(args.input_dir)


if __name__ == "__main__":
    main()
