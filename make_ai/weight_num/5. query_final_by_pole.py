#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
서버와 전주 ID를 입력하면 final_predictions.xlsx에서 해당 전주의 측정번호별 결과를 터미널에 출력.

사용:
  python "5. query_final_by_pole.py" --server <서버> --pole_id <전주ID>
  python "5. query_final_by_pole.py" -s <서버> -p <전주ID>

예:
  python "5. query_final_by_pole.py" --server main --pole_id 4114X505
  python "5. query_final_by_pole.py" -s main -p 4114X505
"""

import argparse
from pathlib import Path

import pandas as pd


def _numeric_sort_key(x):
    """측정번호 정렬용: 숫자면 int, 아니면 원본."""
    try:
        return (0, int(x))
    except (ValueError, TypeError):
        return (1, str(x))


def main():
    parser = argparse.ArgumentParser(description="서버·전주 ID로 측정번호별 결과 조회")
    parser.add_argument("-s", "--server", required=True, help="서버 (예: main, is, kh)")
    parser.add_argument("-p", "--pole_id", "--전주", dest="pole_id", required=True, help="전주 ID (예: 4114X505)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    xlsx_path = base_dir / "export_2nd_eval_to_excel" / "final_predictions.xlsx"

    if not xlsx_path.exists():
        print(f"파일 없음: {xlsx_path}\n먼저 4. make_final_excel_from_mlp.py 를 실행하세요.")
        return

    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip() if hasattr(df.columns, "str") else df.columns

    server = str(args.server).strip()
    pole_id = str(args.pole_id).strip()

    mask = (df["서버"].astype(str).str.strip() == server) & (df["전주"].astype(str).str.strip() == pole_id)
    sub = df.loc[mask].copy()

    if sub.empty:
        print(f"해당 조건의 데이터가 없습니다. 서버={server}, 전주={pole_id}")
        return

    # 측정번호 순 정렬 (숫자 우선)
    sub["_ord"] = sub["측정번호"].astype(str).apply(_numeric_sort_key)
    sub = sub.sort_values("_ord").drop(columns=["_ord"])

    print("=" * 70)
    print(f"서버: {server}  |  전주 ID: {pole_id}  |  조회 건수: {len(sub)}")
    print("=" * 70)

    for _, row in sub.iterrows():
        측정번호 = row.get("측정번호", "")
        프로젝트 = row.get("프로젝트", "")
        판정 = row.get("판정", "")
        기존 = row.get("기존_파단정상", "")
        th = row.get("threshold", "")
        높이 = row.get("파단위치_높이", "")
        각도 = row.get("파단위치_각도", "")
        print(f"  [측정번호 {측정번호}]  프로젝트: {프로젝트}")
        print(f"      판정: {판정}  |  기존(파단/정상): {기존}  |  threshold: {th}")
        print(f"      파단위치 - 높이: {높이}  |  각도: {각도}")
        print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()
