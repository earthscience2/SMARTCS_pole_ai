import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return dict(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        recall=float(recall),
        precision=float(precision),
        accuracy=float(accuracy),
        f1=float(f1),
    )


def extract_max_score_from_boxes(text: str) -> float:
    """
    export_2nd_eval_to_excel 에서 생성된
    'rank=..,score=0.82,deg=[..],...' 형식 문자열에서 score 값들 중 최대값을 추출.
    박스가 없거나 score가 없으면 0.0 반환.
    """
    if not isinstance(text, str) or not text:
        return 0.0
    scores = []
    for m in re.finditer(r"score=([0-9]*\.?[0-9]+)", text):
        try:
            scores.append(float(m.group(1)))
        except ValueError:
            continue
    if not scores:
        return 0.0
    return float(max(scores))


def main():
    base_dir = Path(__file__).resolve().parent
    xlsx_path = base_dir / "export_2nd_eval_to_excel" / "hard_2nd_merge_data_predictions.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # 필수 컬럼 체크
    required_cols = ["label", "x_boxes", "y_boxes", "z_boxes"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {c}")

    # 전주(측정 단위) 단위이므로 그대로 사용
    y_true = df["label"].astype(int).values

    # 축별 최대 score 계산 (일부 규칙에서 y 사용)
    df["max_score_x"] = df["x_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_y"] = df["y_boxes"].astype(str).apply(extract_max_score_from_boxes)
    df["max_score_z"] = df["z_boxes"].astype(str).apply(extract_max_score_from_boxes)

    max_x = df["max_score_x"].values.astype(float)
    max_y = df["max_score_y"].values.astype(float)
    max_z = df["max_score_z"].values.astype(float)

    # recall + precision >= 1 인 점만 출력/플롯
    def ok(m):
        return (m["recall"] + m["precision"]) >= 1.0

    # plot용: 조건 만족 결과 수집 (rule, param_str, recall, precision, f1)
    all_plot = []

    print("총 샘플 수:", len(df))
    print("양성(파단) 샘플 수:", int(np.sum(y_true == 1)))
    print("음성(정상) 샘플 수:", int(np.sum(y_true == 0)))
    print("(조건: recall + precision >= 1 인 결과만 출력)\n")

    # ---- 0) 라이트 모델만: light_prob_break >= t 이면 파단 ----
    if "light_prob_break" in df.columns:
        prob_light = df["light_prob_break"].astype(float).fillna(0.0).values
        ths_light = np.linspace(0.1, 0.9, 9)
        light_results = []
        for t in ths_light:
            y_pred = (prob_light >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["threshold"] = float(t)
            light_results.append(m)
        filtered = [r for r in light_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0] LIGHT", "param": f"t={r['threshold']:.2f}"})
        print("=== [0] LIGHT만 — light_prob_break >= t 이면 파단 ===")
        if filtered:
            for r in filtered:
                print(
                    f"  t={r['threshold']:.2f}  "
                    f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                    f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
                )
        else:
            print("  (조건 만족 없음)")
        print()

        # ---- 0-1) 라이트(t=0.5) OR 하드 x만: (light>=0.5) OR (max_score_x >= t_x), t_x 변화 ----
        light_base = (prob_light >= 0.5).astype(int)
        or_x_results = []
        for t_x in np.linspace(0.1, 0.9, 9):
            y_pred = (light_base | (max_x >= t_x)).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t_x"] = float(t_x)
            or_x_results.append(m)
        filtered = [r for r in or_x_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-1] LIGHT OR x", "param": f"t_x={r['t_x']:.2f}"})
        print("=== [0-1] 라이트(t=0.5) OR 하드 x만 — (light_prob_break >= 0.5) OR (max_score_x >= t_x) ===")
        if filtered:
            for r in filtered:
                print(
                    f"  t_x={r['t_x']:.2f}  "
                    f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                    f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
                )
        else:
            print("  (조건 만족 없음)")
        print()

        # ---- 0-2) 라이트(t=0.5) OR 하드 x만(가중치): (light>=0.5) OR (wx * max_score_x >= t) ----
        or_xw_results = []
        for wx in [1.0, 1.5, 2.0]:
            x_weighted = wx * max_x
            for t in np.linspace(0.1, 0.9, 9):
                y_pred = (light_base | (x_weighted >= t)).astype(int)
                m = compute_metrics(y_true, y_pred)
                m["wx"], m["t"] = float(wx), float(t)
                or_xw_results.append(m)
        filtered = [r for r in or_xw_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-2] LIGHT OR x(w)", "param": f"wx={r['wx']:.1f},t={r['t']:.2f}"})
        print("=== [0-2] 라이트(t=0.5) OR 하드 x만(가중치) — (light >= 0.5) OR (wx * max_score_x >= t) ===")
        if filtered:
            for r in filtered:
                print(
                    f"  wx={r['wx']:.1f}, t={r['t']:.2f}  "
                    f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                    f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
                )
        else:
            print("  (조건 만족 없음)")
        print()

        # ---- 0-3) 라이트 + x + z 다양한 논리 (OR/AND 외) ----
        ths_coarse = [0.2, 0.4, 0.6]  # 그리드 간소화
        # OR3: (light>=t_l) OR (max_x>=t_x) OR (max_z>=t_z)
        or3_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = ((prob_light >= t_l) | (max_x >= t_x) | (max_z >= t_z)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    or3_results.append(m)
        filtered = [r for r in or3_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] OR3(L,x,z)", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] OR3 — (light>=t_l) OR (max_x>=t_x) OR (max_z>=t_z) 이면 파단 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # AND3: (light>=t_l) AND (max_x>=t_x) AND (max_z>=t_z)
        and3_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = ((prob_light >= t_l) & (max_x >= t_x) & (max_z >= t_z)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    and3_results.append(m)
        filtered = [r for r in and3_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] AND3(L,x,z)", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] AND3 — (light>=t_l) AND (max_x>=t_x) AND (max_z>=t_z) 이면 파단 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Majority: (light>=t_l)+(max_x>=t_x)+(max_z>=t_z) >= 2
        maj_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    vote = ((prob_light >= t_l).astype(int) + (max_x >= t_x).astype(int) + (max_z >= t_z).astype(int))
                    y_pred = (vote >= 2).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    maj_results.append(m)
        filtered = [r for r in maj_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Majority(L,x,z)", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] Majority — (light>=t_l)+(max_x>=t_x)+(max_z>=t_z) >= 2 이면 파단 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Light AND (x OR z): (light>=t_l) AND ((max_x>=t_x) OR (max_z>=t_z))
        land_or_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = ((prob_light >= t_l) & ((max_x >= t_x) | (max_z >= t_z))).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    land_or_results.append(m)
        filtered = [r for r in land_or_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] L AND (x OR z)", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] L AND (x OR z) — (light>=t_l) AND ((max_x>=t_x) OR (max_z>=t_z)) ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Light OR (x AND z): (light>=t_l) OR ((max_x>=t_x) AND (max_z>=t_z))
        lor_and_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = ((prob_light >= t_l) | ((max_x >= t_x) & (max_z >= t_z))).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    lor_and_results.append(m)
        filtered = [r for r in lor_and_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] L OR (x AND z)", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] L OR (x AND z) — (light>=t_l) OR ((max_x>=t_x) AND (max_z>=t_z)) ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [0-3-ext] L OR (x AND z) 세밀 그리드 (상위 점 특성: t_l 0.5~0.65, tx/tz 0.2~0.5)
        ths_tl_fine = [0.50, 0.55, 0.60, 0.65]
        ths_xz_fine = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        lor_and_fine_results = []
        for t_l in ths_tl_fine:
            for t_x in ths_xz_fine:
                for t_z in ths_xz_fine:
                    y_pred = ((prob_light >= t_l) | ((max_x >= t_x) & (max_z >= t_z))).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    lor_and_fine_results.append(m)
        filtered = [r for r in lor_and_fine_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3-ext] L OR (x AND z) fine", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3-ext] L OR (x AND z) fine - t_l in [0.5,0.55,0.6,0.65], tx/tz 세밀 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-(x["recall"] + x["precision"]), -x["f1"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, r+p={r['recall']+r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [0-3-ext] L OR (가중 x AND z): (light>=0.6) OR ((wx*max_x>=tx) AND (wz*max_z>=tz))
        lor_wxz_results = []
        for wx in [1.0, 1.5, 2.0]:
            for wz in [1.0, 1.5, 2.0]:
                for tx in [0.2, 0.4, 0.6]:
                    for tz in [0.2, 0.4, 0.6]:
                        pred_x = (wx * max_x >= tx)
                        pred_z = (wz * max_z >= tz)
                        y_pred = ((prob_light >= 0.6) | (pred_x & pred_z)).astype(int)
                        m = compute_metrics(y_true, y_pred)
                        m["wx"], m["wz"], m["tx"], m["tz"] = float(wx), float(wz), float(tx), float(tz)
                        lor_wxz_results.append(m)
        filtered = [r for r in lor_wxz_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3-ext] L OR (wx*x AND wz*z)", "param": f"t_l=0.60,wx={r['wx']:.1f},wz={r['wz']:.1f},tx={r['tx']:.2f},tz={r['tz']:.2f}"})
        print("=== [0-3-ext] L OR (wx*x AND wz*z) - t_l=0.6, wx/wz in [1,1.5,2], tx/tz in [0.2,0.4,0.6] ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-(x["recall"] + x["precision"]), -x["f1"]))[:15]:
                print(f"  wx={r['wx']:.1f}, wz={r['wz']:.1f}, tx={r['tx']:.2f}, tz={r['tz']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [0-3-ext] LIGHT 단일 세밀: t in 0.45~0.65 (상위에 LIGHT 0.5, 0.6 포함)
        ths_light_fine = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
        light_fine_results = []
        for t in ths_light_fine:
            y_pred = (prob_light >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["threshold"] = float(t)
            light_fine_results.append(m)
        filtered = [r for r in light_fine_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-ext] LIGHT fine", "param": f"t={r['threshold']:.2f}"})
        print("=== [0-ext] LIGHT fine - t in [0.45, 0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65] ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['threshold']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, r+p={r['recall']+r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [0-3-ext] L OR x 세밀: (light>=0.5) OR (max_x>=tx), tx 세밀
        ths_x_fine = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        lor_x_fine_results = []
        for tx in ths_x_fine:
            y_pred = ((prob_light >= 0.5) | (max_x >= tx)).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t_x"] = float(tx)
            lor_x_fine_results.append(m)
        filtered = [r for r in lor_x_fine_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-ext] LIGHT OR x fine", "param": f"t_l=0.50,tx={r['t_x']:.2f}"})
        print("=== [0-ext] LIGHT OR x fine - t_l=0.5, tx 세밀 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-(x["recall"] + x["precision"])))[:15]:
                print(f"  tx={r['t_x']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [new] 2단 L + (x AND z): (L>=0.62) OR ( (L>=0.5) AND (x>=tx) AND (z>=tz) ) — 상위 1위 LIGHT 0.62 반영
        two_tier_results = []
        for tx in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
            for tz in [0.35, 0.4, 0.5, 0.55, 0.6]:
                high_l = prob_light >= 0.62
                low_l_and_xz = (prob_light >= 0.5) & (max_x >= tx) & (max_z >= tz)
                y_pred = (high_l | low_l_and_xz).astype(int)
                m = compute_metrics(y_true, y_pred)
                m["tx"], m["tz"] = float(tx), float(tz)
                two_tier_results.append(m)
        filtered = [r for r in two_tier_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[new] 2-tier L+(x AND z)", "param": f"L>=0.62 or (L>=0.5 and x>={r['tx']:.2f},z>={r['tz']:.2f})"})
        print("=== [new] 2-tier: (L>=0.62) OR (L>=0.5 AND x>=tx AND z>=tz) ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-(x["recall"] + x["precision"]), -x["f1"]))[:12]:
                print(f"  tx={r['tx']:.2f}, tz={r['tz']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, r+p={r['recall']+r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [new] L OR (x,z 평균): (L>=t_l) OR ((max_x+max_z)/2 >= t_avg)
        avg_xz = (max_x + max_z) / 2.0
        lor_avg_results = []
        for t_l in [0.58, 0.60, 0.62, 0.65]:
            for t_avg in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                y_pred = ((prob_light >= t_l) | (avg_xz >= t_avg)).astype(int)
                m = compute_metrics(y_true, y_pred)
                m["t_l"], m["t_avg"] = float(t_l), float(t_avg)
                lor_avg_results.append(m)
        filtered = [r for r in lor_avg_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[new] L OR (x+z)/2", "param": f"t_l={r['t_l']:.2f}, (x+z)/2>={r['t_avg']:.2f}"})
        print("=== [new] L OR (x+z)/2 - (light>=t_l) OR ((max_x+max_z)/2 >= t_avg) ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-(x["recall"] + x["precision"]), -x["f1"]))[:12]:
                print(f"  t_l={r['t_l']:.2f}, t_avg={r['t_avg']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [new] 가중합 L 우세 세밀: (3*L + x + z)/5 >= t (상위 구간 t 0.46~0.62)
        sum_l_heavy = (3.0 * prob_light + max_x + max_z) / 5.0
        sum_heavy_results = []
        for t in np.linspace(0.46, 0.62, 9):
            y_pred = (sum_l_heavy >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            sum_heavy_results.append(m)
        filtered = [r for r in sum_heavy_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[new] Sum(3L+x+z)/5", "param": f"t={r['t']:.2f}"})
        print("=== [new] (3*L+x+z)/5 >= t, t in [0.46..0.62] ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, r+p={r['recall']+r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [new] max(L, (x+z)/2) >= t — L과 x,z 평균 중 큰 값
        max_l_avg = np.maximum(prob_light, (max_x + max_z) / 2.0)
        max_l_avg_results = []
        for t in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
            y_pred = (max_l_avg >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            max_l_avg_results.append(m)
        filtered = [r for r in max_l_avg_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[new] max(L,(x+z)/2)", "param": f"t={r['t']:.2f}"})
        print("=== [new] max(L, (x+z)/2) >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # [new] (L + max(x,z))/2 >= t — L과 더 좋은 축의 평균
        l_plus_best = (prob_light + np.maximum(max_x, max_z)) / 2.0
        l_best_results = []
        for t in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62]:
            y_pred = (l_plus_best >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            l_best_results.append(m)
        filtered = [r for r in l_best_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[new] (L+max(x,z))/2", "param": f"t={r['t']:.2f}"})
        print("=== [new] (L + max(x,z))/2 >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # 가중합 (Light, x, z): (a*light + b*max_x + c*max_z) / (a+b+c) >= t
        weight_triples = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 1, 2)]
        sum3_results = []
        for (a, b, c) in weight_triples:
            s = (a * prob_light + b * max_x + c * max_z) / (a + b + c)
            for t in np.linspace(0.2, 0.6, 5):
                y_pred = (s >= t).astype(int)
                m = compute_metrics(y_true, y_pred)
                m["w_l"], m["w_x"], m["w_z"] = a, b, c
                m["t"] = float(t)
                sum3_results.append(m)
        filtered = [r for r in sum3_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Sum(L,x,z)", "param": f"w=({r['w_l']},{r['w_x']},{r['w_z']}),t={r['t']:.2f}"})
        print("=== [0-3] Sum(L,x,z) — (a*light+b*max_x+c*max_z)/(a+b+c) >= t 이면 파단 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  w=({r['w_l']},{r['w_x']},{r['w_z']}), t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Max3: max(light, max_x, max_z) >= t
        max3_results = []
        for t in np.linspace(0.1, 0.7, 7):
            s = np.maximum(np.maximum(prob_light, max_x), max_z)
            y_pred = (s >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            max3_results.append(m)
        filtered = [r for r in max3_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Max3(L,x,z)", "param": f"t={r['t']:.2f}"})
        print("=== [0-3] Max3 — max(light, max_x, max_z) >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Min3: min(light, max_x, max_z) >= t
        min3_results = []
        for t in np.linspace(0.1, 0.5, 5):
            s = np.minimum(np.minimum(prob_light, max_x), max_z)
            y_pred = (s >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            min3_results.append(m)
        filtered = [r for r in min3_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Min3(L,x,z)", "param": f"t={r['t']:.2f}"})
        print("=== [0-3] Min3 — min(light, max_x, max_z) >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Gmean: (light * max_x * max_z)^(1/3) >= t
        gmean_results = []
        eps = 1e-9
        g = (np.maximum(prob_light, eps) * np.maximum(max_x, eps) * np.maximum(max_z, eps)) ** (1.0 / 3)
        for t in np.linspace(0.1, 0.5, 5):
            y_pred = (g >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            gmean_results.append(m)
        filtered = [r for r in gmean_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Gmean(L,x,z)", "param": f"t={r['t']:.2f}"})
        print("=== [0-3] Gmean — (light*max_x*max_z)^(1/3) >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # (L AND x) OR z
        landx_orz_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = (((prob_light >= t_l) & (max_x >= t_x)) | (max_z >= t_z)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    landx_orz_results.append(m)
        filtered = [r for r in landx_orz_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] (L AND x) OR z", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] (L AND x) OR z ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # (L AND z) OR x
        landz_orx_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = (((prob_light >= t_l) & (max_z >= t_z)) | (max_x >= t_x)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    landz_orx_results.append(m)
        filtered = [r for r in landz_orx_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] (L AND z) OR x", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] (L AND z) OR x ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # (L OR x) AND z
        lorx_andz_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = (((prob_light >= t_l) | (max_x >= t_x)) & (max_z >= t_z)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    lorx_andz_results.append(m)
        filtered = [r for r in lorx_andz_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] (L OR x) AND z", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] (L OR x) AND z ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # (L OR z) AND x
        lorz_andx_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    y_pred = (((prob_light >= t_l) | (max_z >= t_z)) & (max_x >= t_x)).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    lorz_andx_results.append(m)
        filtered = [r for r in lorz_andx_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] (L OR z) AND x", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] (L OR z) AND x ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Median(L,x,z) >= t
        stack = np.stack([prob_light, max_x, max_z], axis=1)
        med = np.median(stack, axis=1)
        median_results = []
        for t in np.linspace(0.1, 0.6, 6):
            y_pred = (med >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t"] = float(t)
            median_results.append(m)
        filtered = [r for r in median_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Median(L,x,z)", "param": f"t={r['t']:.2f}"})
        print("=== [0-3] Median — median(light, max_x, max_z) >= t ===")
        if filtered:
            for r in filtered:
                print(f"  t={r['t']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # Vote == 2 only (정확히 2표)
        vote2_results = []
        for t_l in ths_coarse:
            for t_x in ths_coarse:
                for t_z in ths_coarse:
                    vote = ((prob_light >= t_l).astype(int) + (max_x >= t_x).astype(int) + (max_z >= t_z).astype(int))
                    y_pred = (vote == 2).astype(int)
                    m = compute_metrics(y_true, y_pred)
                    m["t_l"], m["t_x"], m["t_z"] = float(t_l), float(t_x), float(t_z)
                    vote2_results.append(m)
        filtered = [r for r in vote2_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-3] Vote==2", "param": f"t_l={r['t_l']:.2f},tx={r['t_x']:.2f},tz={r['t_z']:.2f}"})
        print("=== [0-3] Vote==2 — 정확히 2개만 넘으면 파단 ===")
        if filtered:
            for r in sorted(filtered, key=lambda x: (-x["f1"], -x["recall"]))[:15]:
                print(f"  t_l={r['t_l']:.2f}, t_x={r['t_x']:.2f}, t_z={r['t_z']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}")
        else:
            print("  (조건 만족 없음)")
        print()

        # ---- [0-4] 5번이어도 x,y,z 중 하나라도 t 이상이어야 파단 (그렇지 않으면 탈락) ----
        # 파단 = (light >= 0.5) AND (max_x >= t OR max_y >= t OR max_z >= t)
        rule5_drop_results = []
        for t in np.linspace(0.2, 0.5, 4):  # 0.2, 0.3, 0.4, 0.5
            five_ok = (prob_light >= 0.5)
            any_axis_ok = (max_x >= t) | (max_y >= t) | (max_z >= t)
            y_pred = (five_ok & any_axis_ok).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["t_xyz"] = float(t)
            rule5_drop_results.append(m)
        filtered = [r for r in rule5_drop_results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[0-4] 5번 & (x|y|z>=t) 탈락", "param": f"t={r['t_xyz']:.2f}"})
        print("=== [0-4] 5번 & (x|y|z>=t) - 5번이어도 x,y,z 모두 t 미만이면 탈락(파단 아님) ===")
        if filtered:
            for r in filtered:
                print(
                    f"  t={r['t_xyz']:.2f}  "
                    f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                    f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
                )
        else:
            print("  (조건 만족 없음)")
        print()
    else:
        print("(light_prob_break 컬럼 없음 — 라이트 모델 평가 생략)\n")

    # x, z 축만 사용. 가중치 세트: (wx, wz) → 가중 score = (wx*max_x + wz*max_z) / (wx+wz)
    weight_sets = [
        {"name": "equal", "wx": 1.0, "wz": 1.0},
        {"name": "x_high", "wx": 2.0, "wz": 1.0},
        {"name": "z_high", "wx": 1.0, "wz": 2.0},
        {"name": "xz_both_high", "wx": 2.0, "wz": 2.0},
    ]

    ths = np.linspace(0.1, 0.9, 9)
    ths_2d = np.linspace(0.1, 0.9, 9)  # tx, tz 그리드

    # ---- 1) THRESH: 가중 평균 한 개 threshold ----
    print("=== [1] THRESH — 가중 score = (wx*max_x + wz*max_z)/(wx+wz), score >= t 이면 파단 ===")
    any_thresh = False
    for w in weight_sets:
        wx, wz = w["wx"], w["wz"]
        scores = (wx * max_x + wz * max_z) / (wx + wz)
        results = []
        for t in ths:
            y_pred = (scores >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["threshold"] = float(t)
            m["rule"] = "THRESH"
            m["weight_name"] = w["name"]
            results.append(m)
        filtered = [r for r in results if ok(r)]
        for r in filtered:
            all_plot.append({**r, "rule": "[1] THRESH", "param": f"{w['name']} t={r['threshold']:.2f}"})
        if filtered:
            any_thresh = True
            print(f"  가중치 {w['name']} (wx={wx}, wz={wz}):")
            for r in filtered:
                print(
                    f"    t={r['threshold']:.2f}  recall={r['recall']:.3f}, precision={r['precision']:.3f}, "
                    f"F1={r['f1']:.3f}, tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
                )
    if not any_thresh:
        print("  (조건 만족 없음)")
    print()

    # ---- 2) OR: (max_x >= tx) OR (max_z >= tz) 이면 파단 ----
    or_results = []
    for tx in ths_2d:
        for tz in ths_2d:
            y_pred = ((max_x >= tx) | (max_z >= tz)).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["tx"] = float(tx)
            m["tz"] = float(tz)
            m["rule"] = "OR"
            or_results.append(m)
    or_filtered = [r for r in or_results if ok(r)]
    for r in or_filtered:
        all_plot.append({**r, "rule": "[2] OR", "param": f"tx={r['tx']:.2f},tz={r['tz']:.2f}"})
    or_sorted = sorted(or_filtered, key=lambda r: (r["f1"], r["recall"], r["precision"]), reverse=True)
    print("=== [2] OR — (max_score_x >= tx) OR (max_score_z >= tz) 이면 파단 ===")
    if or_sorted:
        for r in or_sorted:
            print(
                f"  tx={r['tx']:.2f}, tz={r['tz']:.2f}  "
                f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
            )
    else:
        print("  (조건 만족 없음)")
    print()

    # ---- 3) AND: (max_x >= tx) AND (max_z >= tz) 이면 파단 ----
    and_results = []
    for tx in ths_2d:
        for tz in ths_2d:
            y_pred = ((max_x >= tx) & (max_z >= tz)).astype(int)
            m = compute_metrics(y_true, y_pred)
            m["tx"] = float(tx)
            m["tz"] = float(tz)
            m["rule"] = "AND"
            and_results.append(m)
    and_filtered = [r for r in and_results if ok(r)]
    for r in and_filtered:
        all_plot.append({**r, "rule": "[3] AND", "param": f"tx={r['tx']:.2f},tz={r['tz']:.2f}"})
    and_sorted = sorted(and_filtered, key=lambda r: (r["f1"], r["recall"], r["precision"]), reverse=True)
    print("=== [3] AND — (max_score_x >= tx) AND (max_score_z >= tz) 이면 파단 ===")
    if and_sorted:
        for r in and_sorted:
            print(
                f"  tx={r['tx']:.2f}, tz={r['tz']:.2f}  "
                f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
            )
    else:
        print("  (조건 만족 없음)")
    print()

    # ---- 4) MAX: max(max_x, max_z) >= t (기존과 동일, 단일 threshold) ----
    max_scores = np.maximum(max_x, max_z)
    max_results = []
    for t in ths:
        y_pred = (max_scores >= t).astype(int)
        m = compute_metrics(y_true, y_pred)
        m["threshold"] = float(t)
        m["rule"] = "MAX"
        max_results.append(m)
    max_filtered = [r for r in max_results if ok(r)]
    for r in max_filtered:
        all_plot.append({**r, "rule": "[4] MAX", "param": f"t={r['threshold']:.2f}"})
    print("=== [4] MAX — max(max_score_x, max_score_z) >= t 이면 파단 ===")
    if max_filtered:
        for r in max_filtered:
            print(
                f"  t={r['threshold']:.2f}  "
                f"recall={r['recall']:.3f}, precision={r['precision']:.3f}, F1={r['f1']:.3f}  "
                f"tp={r['tp']}, fp={r['fp']}, fn={r['fn']}, tn={r['tn']}"
            )
    else:
        print("  (조건 만족 없음)")

    # ---- Plot: recall+precision 상위 50개만 표시, 점별 번호 + 번호-가중치 목록 ----
    if HAS_PLOT and all_plot:
        # recall + precision 합이 큰 상위 50개만
        plot_subset = sorted(
            all_plot, key=lambda p: p["recall"] + p["precision"], reverse=True
        )[:50]
        for i, p in enumerate(plot_subset, start=1):
            p["_idx"] = i

        rules = sorted(set(p["rule"] for p in plot_subset))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(rules), 1)))
        rule_to_color = {r: colors[i % len(colors)] for i, r in enumerate(rules)}

        fig, ax = plt.subplots(figsize=(10, 8))
        all_texts = []
        for rule in rules:
            pts = [p for p in plot_subset if p["rule"] == rule]
            if not pts:
                continue
            rec = [p["recall"] for p in pts]
            prec = [p["precision"] for p in pts]
            f1 = [p["f1"] for p in pts]
            sizes = [80 + 120 * f for f in f1]
            ax.scatter(
                rec, prec, s=sizes, c=[rule_to_color[rule]], label=rule,
                alpha=0.7, edgecolors="k", linewidths=0.5
            )
            for p in pts:
                t = ax.annotate(
                    str(p["_idx"]),
                    (p["recall"], p["precision"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
                )
                all_texts.append(t)
        # 겹치는 레이블 밀어내기: adjustText 있으면 사용, 없으면 소량 랜덤 지터
        if HAS_ADJUST_TEXT and all_texts:
            adjust_text(
                all_texts,
                expand_points=(1.2, 1.2),
                expand_text=(1.05, 1.2),
                force_text=(0.3, 0.5),
                force_points=(0.2, 0.4),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        elif all_texts and not HAS_ADJUST_TEXT:
            # adjustText 미설치 시: 레이블에 소량 랜덤 지터 적용 (겹침 완화)
            rng = np.random.default_rng(42)
            for t in all_texts:
                jx = rng.uniform(-8, 12)
                jy = rng.uniform(-8, 12)
                t.set_position((jx, jy))
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("파단 판단 규칙 비교 (recall+precision 상위 50개)", fontsize=11)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        out_dir = base_dir / "export_2nd_eval_to_excel"
        out_plot = out_dir / "eval_recall_precision.png"
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()

        # 번호별 규칙·가중치 목록 저장 (상위 50개만)
        out_labels = out_dir / "eval_recall_precision_labels.txt"
        with open(out_labels, "w", encoding="utf-8") as f:
            f.write("번호\t규칙\t파라미터(가중치/임계값)\trecall\tprecision\trecall+precision\tF1\n")
            for p in plot_subset:
                rp = p["recall"] + p["precision"]
                f.write(
                    f"{p['_idx']}\t{p['rule']}\t{p['param']}\t"
                    f"{p['recall']:.3f}\t{p['precision']:.3f}\t{rp:.3f}\t{p['f1']:.3f}\n"
                )
        print(f"\nPlot 저장: {out_plot}")
        print(f"번호-가중치 목록 저장: {out_labels}")
    elif not HAS_PLOT and all_plot:
        print("\n(matplotlib 없음 — plot 생략)")


if __name__ == "__main__":
    main()

