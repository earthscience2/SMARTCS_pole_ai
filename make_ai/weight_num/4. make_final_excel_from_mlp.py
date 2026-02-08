#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
3번에서 만든 MLP 모델과 3구간 threshold로 기존 엑셀 데이터를 재판정하여
'final' 제목의 엑셀을 만듦.

컬럼: 서버, 프로젝트, 전주, 측정번호, 판정(파단/보류/정상), threshold(MLP확률), 파단위치_높이, 파단위치_각도
- 파단위치: x/y/z 박스 중 신뢰도(score) 가장 높은 박스 1개만 → 높이, 각도

사용 순서:
  1) 1. export_2nd_eval_to_excel.py 로 엑셀 생성
  2) 3. eval_mlp_from_agg_excel.py 로 모델·3구간 threshold 생성
  3) python "4. make_final_excel_from_mlp.py"

출력: export_2nd_eval_to_excel/final_predictions.xlsx
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


def extract_max_score_from_boxes(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    scores = []
    for m in re.finditer(r"score=([0-9]*\.?[0-9]+)", text):
        try:
            scores.append(float(m.group(1)))
        except ValueError:
            continue
    return float(max(scores)) if scores else 0.0


def parse_best_box_from_axes(x_boxes: str, y_boxes: str, z_boxes: str):
    """
    x_boxes, y_boxes, z_boxes 문자열에서 모든 박스를 파싱해
    score가 가장 높은 박스 1개의 (높이, 각도) 반환.
    형식: rank=..,score=0.82,deg=[dmin,dmax],h=[hmin,hmax],...
    반환: (높이문자열, 각도문자열) 또는 ("", "")
    """
    def parse_one_box(part):
        score_m = re.search(r"score=([0-9]*\.?[0-9]+)", part)
        deg_m = re.search(r"deg=\[([0-9.-]+),([0-9.-]+)\]", part)
        h_m = re.search(r"h=\[([0-9.-]+),([0-9.-]+)\]", part)
        if not score_m:
            return None
        try:
            score = float(score_m.group(1))
        except ValueError:
            return None
        dmin = dmax = hmin = hmax = ""
        if deg_m:
            dmin, dmax = deg_m.group(1), deg_m.group(2)
        if h_m:
            hmin, hmax = h_m.group(1), h_m.group(2)
        return (score, dmin, dmax, hmin, hmax)

    best_score = -1.0
    best_h = best_deg = ""
    for text in (x_boxes or "", y_boxes or "", z_boxes or ""):
        if not isinstance(text, str):
            continue
        for part in text.split("|"):
            part = part.strip()
            if not part:
                continue
            one = parse_one_box(part)
            if one is None:
                continue
            score, dmin, dmax, hmin, hmax = one
            if score > best_score:
                best_score = score
                try:
                    hmin_f, hmax_f = float(hmin), float(hmax)
                    dmin_f, dmax_f = float(dmin), float(dmax)
                    best_h = f"{hmin_f:.2f}~{hmax_f:.2f}"
                    best_deg = f"{dmin_f:.2f}~{dmax_f:.2f}"
                except (ValueError, TypeError):
                    best_h = f"{hmin}~{hmax}" if (hmin and hmax) else ""
                    best_deg = f"{dmin}~{dmax}" if (dmin and dmax) else ""
    return (best_h, best_deg) if best_score >= 0 else ("", "")


def main():
    if not HAS_TF:
        print("TensorFlow가 필요합니다: pip install tensorflow")
        return

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "export_2nd_eval_to_excel"
    xlsx_path = out_dir / "hard_2nd_merge_data_predictions.xlsx"
    model_path = out_dir / "eval_mlp_model.keras"
    thresholds_path = out_dir / "eval_mlp_3zone_thresholds.txt"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"엑셀을 찾을 수 없습니다: {xlsx_path}\n1. export_2nd_eval_to_excel.py 를 먼저 실행하세요.")
    if not model_path.exists():
        raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}\n3. eval_mlp_from_agg_excel.py 를 먼저 실행하세요.")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"3구간 threshold를 찾을 수 없습니다: {thresholds_path}\n3. eval_mlp_from_agg_excel.py 를 먼저 실행하세요.")

    # 3구간 threshold 로드
    thresh_normal = 0.65
    thresh_pardan = 0.95
    with open(thresholds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            key, val = line.split("\t", 1)
            if key == "정상_상한":
                thresh_normal = float(val)
            elif key == "파단_하한":
                thresh_pardan = float(val)

    df = pd.read_excel(xlsx_path)
    # 컬럼명 공백 제거 (엑셀 읽기 시 가끔 생김)
    df.columns = df.columns.str.strip() if hasattr(df.columns, "str") else df.columns
    for c in ["x_boxes", "y_boxes", "z_boxes"]:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 없음: {c}")

    # 서버, 프로젝트, 전주 (없으면 빈칸)
    server = df["server"].astype(str).fillna("").values if "server" in df.columns else [""] * len(df)
    project = df["project"].astype(str).fillna("").values if "project" in df.columns else [""] * len(df)
    pole_id = df["pole_id"].astype(str).fillna("").values if "pole_id" in df.columns else [""] * len(df)

    # 측정번호: mesonu (같은 전주에 1, 2, ...). 엑셀에 측정번호 컬럼 있으면 사용, 없으면 순번
    if "측정번호" in df.columns:
        측정번호 = df["측정번호"].astype(str).fillna("").values
    else:
        측정번호 = [str(i) for i in range(1, len(df) + 1)]

    # 특징: light, max_x, max_y, max_z
    use_light = "light_prob_break" in df.columns
    df["max_score_x"] = df["x_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_y"] = df["y_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_z"] = df["z_boxes"].astype(str).apply(extract_max_score_from_boxes)
    if use_light:
        X = np.column_stack([
            df["light_prob_break"].astype(float).fillna(0.0).values,
            df["max_score_x"].values.astype(float),
            df["max_score_y"].values.astype(float),
            df["max_score_z"].values.astype(float),
        ])
    else:
        X = np.column_stack([
            df["max_score_x"].values.astype(float),
            df["max_score_y"].values.astype(float),
            df["max_score_z"].values.astype(float),
        ])
    X = X.astype(np.float32)

    model = keras.models.load_model(model_path, compile=False)
    proba = model.predict(X, verbose=0).ravel()

    # 3구간 판정
    판정 = []
    for p in proba:
        if p < thresh_normal:
            판정.append("정상")
        elif p < thresh_pardan:
            판정.append("보류")
        else:
            판정.append("파단")

    # 행별 신뢰도 최고 박스 1개 -> 높이, 각도
    파단위치_높이 = []
    파단위치_각도 = []
    for _, row in df.iterrows():
        h, deg = parse_best_box_from_axes(
            row.get("x_boxes", ""),
            row.get("y_boxes", ""),
            row.get("z_boxes", ""),
        )
        파단위치_높이.append(h)
        파단위치_각도.append(deg)

    # 기존 파단/정상 여부: label(0/1) 또는 label_text(break/normal)에서 채움
    기존_파단정상 = []
    if "label" in df.columns:
        label_col = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        for v in label_col:
            기존_파단정상.append("파단" if v == 1 else "정상")
    elif "label_text" in df.columns:
        for v in df["label_text"].astype(str).str.strip().str.lower():
            if v == "break":
                기존_파단정상.append("파단")
            elif v == "normal":
                기존_파단정상.append("정상")
            else:
                기존_파단정상.append(v if v and v != "nan" else "")
    else:
        기존_파단정상 = [""] * len(df)

    result = pd.DataFrame({
        "서버": server,
        "프로젝트": project,
        "전주": pole_id,
        "측정번호": 측정번호,
        "판정": 판정,
        "기존_파단정상": 기존_파단정상,
        "threshold": np.round(proba, 4),
        "파단위치_높이": 파단위치_높이,
        "파단위치_각도": 파단위치_각도,
    })

    out_path = out_dir / "final_predictions.xlsx"
    result.to_excel(out_path, index=False)
    print(f"저장: {out_path}")
    print(f"  행 수: {len(result)}, 판정 분포: {result['판정'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
