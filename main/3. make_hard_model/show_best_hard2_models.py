#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""베스트 2차 모델 정보를 보기 좋게 출력하는 유틸리티"""

import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BEST_ALIAS_DIR = CURRENT_DIR / "best_hard_model_2nd"

def show_best_models():
    print("\n" + "=" * 80)
    print("🏆 베스트 Hard 2차 모델 현황")
    print("=" * 80)
    
    by_first_stage_dir = BEST_ALIAS_DIR / "by_first_stage"
    
    if not by_first_stage_dir.exists():
        print("⚠️  1차 모델별 베스트 디렉터리가 없습니다.")
        return
    
    # 1차 모델별 베스트 출력
    first_stage_dirs = sorted([d for d in by_first_stage_dir.iterdir() if d.is_dir()])
    
    if not first_stage_dirs:
        print("⚠️  등록된 베스트 모델이 없습니다.")
        return
    
    print(f"\n📂 1차 모델별 베스트 2차 모델 ({len(first_stage_dirs)}개):\n")
    
    for first_stage_dir in first_stage_dirs:
        first_stage_name = first_stage_dir.name
        selection_file = first_stage_dir / "best_model_selection.json"
        
        if not selection_file.exists():
            print(f"  ❌ {first_stage_name}: 정보 없음")
            continue
        
        try:
            with open(selection_file, 'r', encoding='utf-8') as f:
                selection = json.load(f)
            
            selected = selection.get("selected", {})
            model_run = selected.get("model_run", "N/A")
            metrics = selected.get("metrics", {})
            
            f1 = metrics.get("overall_best_f1", 0)
            auc = metrics.get("overall_auc_high_iou", 0)
            sep = metrics.get("overall_separation", 0)
            
            print(f"  ✅ 1차 모델: {first_stage_name}")
            print(f"     └─ 2차 모델: {model_run}")
            print(f"        ├─ F1 Score: {f1:.4f}")
            print(f"        ├─ AUC (High IoU): {auc:.4f}")
            print(f"        └─ Separation: {sep:.4f}")
            print()
        except Exception as e:
            print(f"  ❌ {first_stage_name}: 오류 ({e})")
    
    # 전체 베스트 출력
    overall_best_dir = BEST_ALIAS_DIR / "overall_best"
    overall_selection_file = overall_best_dir / "best_model_selection.json"
    
    if overall_selection_file.exists():
        try:
            with open(overall_selection_file, 'r', encoding='utf-8') as f:
                overall_selection = json.load(f)
            
            selected = overall_selection.get("selected", {})
            first_stage = selected.get("first_stage_run", "N/A")
            model_run = selected.get("model_run", "N/A")
            metrics = selected.get("metrics", {})
            
            f1 = metrics.get("overall_best_f1", 0)
            auc = metrics.get("overall_auc_high_iou", 0)
            sep = metrics.get("overall_separation", 0)
            
            print("=" * 80)
            print("🌟 전체 베스트 2차 모델:")
            print(f"   1차 모델: {first_stage}")
            print(f"   2차 모델: {model_run}")
            print(f"   F1 Score: {f1:.4f}")
            print(f"   AUC (High IoU): {auc:.4f}")
            print(f"   Separation: {sep:.4f}")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"\n⚠️  전체 베스트 정보 로드 실패: {e}\n")
    else:
        print("\n⚠️  전체 베스트 정보가 아직 없습니다.\n")

if __name__ == "__main__":
    show_best_models()
