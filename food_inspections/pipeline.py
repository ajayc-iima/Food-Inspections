"""End-to-end run: data -> panel -> model selection -> models -> evaluation -> outputs."""
import json
import time

import pandas as pd

from . import charts, config as C
from .analysis import shortlist_profiles, uncertainty
from .data import clean, load_raw
from .evaluation import run_evaluation, shortlists, summarise
from .features import load_panel
from .models import (design_matrix, logistic_coefficients, make_logistic, make_xgboost,
                     select_xgboost)


def run(refresh: bool = False, rebuild_panel: bool = False) -> dict:
    t0 = time.time()
    for d in (C.DATA_DIR, C.RESULTS_DIR, C.CHARTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data
    df = clean(load_raw(refresh=refresh))
    print(f"[clean] {len(df):,} routine inspections, {df['license_'].nunique():,} establishments, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    # 2. point-in-time panel
    panel = load_panel(df, rebuild=rebuild_panel or refresh)
    is_train = panel["period"] == "train"
    print(f"[panel] {len(panel):,} examples (train {is_train.sum():,}, "
          f"test {(panel['period'] == 'test').sum():,})")
    X = design_matrix(panel)
    y = panel["label"].astype(int)

    # 3. choose the XGBoost configuration on training dates only
    print("[select] validating XGBoost configurations on pre-2020 dates ...")
    train_panel = panel[is_train]
    selected, table = select_xgboost(train_panel, X[is_train])
    table.to_csv(C.RESULTS_DIR / "model_selection.csv", index_label="config")
    print(table.round(3).to_string())
    print(f"[select] chosen: {selected}")

    # 4. fit on all training dates, score everything
    print("[model] fitting logistic regression and XGBoost on all training dates ...")
    lr = make_logistic().fit(X[is_train], y[is_train])
    xgb = make_xgboost(C.XGB_CONFIGS[selected]).fit(X[is_train], y[is_train])
    panel["score_logistic"] = lr.predict_proba(X)[:, 1]
    panel["score_xgboost"] = xgb.predict_proba(X)[:, 1]

    # 5. evaluation and outputs
    print("[eval] computing Precision@K per decision date ...")
    metrics = run_evaluation(panel)
    metrics.to_csv(C.RESULTS_DIR / "metrics_by_date.csv", index=False)

    importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    importance.to_csv(C.RESULTS_DIR / "feature_importance.csv", header=["importance"])
    coef = logistic_coefficients(lr, list(X.columns))
    coef.to_csv(C.RESULTS_DIR / "logistic_coefficients.csv", header=["std_coefficient"])

    sl = shortlists(panel)
    sl.to_csv(C.RESULTS_DIR / "shortlists_all_test_dates.csv.gz", index=False, compression="gzip")
    keep = ["approach", "rank", "license_", "dba_name", "zip", "facility_grouped", "risk_latest",
            "hist_fail_rate_smooth", "score_logistic", "score_xgboost", "label"]
    latest = sl[sl["decision_date"] == sl["decision_date"].max()]
    latest[keep].to_csv(C.RESULTS_DIR / "shortlist_latest.csv", index=False)

    profile, facility_mix = shortlist_profiles(panel, "baseline", "logistic")
    profile.to_csv(C.RESULTS_DIR / "shortlist_profiles.csv", index_label="feature (mean)")
    facility_mix.to_csv(C.RESULTS_DIR / "shortlist_facility_mix.csv", index_label="facility")

    summary = summarise(df, panel, metrics, selected)
    summary["test_period"]["uncertainty_logistic_vs_baseline"] = uncertainty(metrics, "logistic")
    summary["test_period"]["uncertainty_xgboost_vs_baseline"] = uncertainty(metrics, "xgboost")
    with open(C.RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[charts] rendering figures ...")
    charts.render_all(df, panel, metrics, importance, coef, profile, facility_mix)

    tp = summary["test_period"]
    print("\n============ HEADLINE RESULTS (test period 2022-2025, out-of-sample) ============")
    print(f"random selection (mean candidate failure rate): {tp['mean_base_rate'] * 100:.1f}%")
    for a in C.APPROACHES:
        p = tp["precision_at_k"][str(C.K)][a] * 100
        print(f"{C.APPROACH_LABELS[a]:<36} Precision@{C.K} = {p:5.1f}%")
    lv = tp["logistic_vs_baseline_at_100"]
    print(f"logistic vs baseline: {lv['mean_difference'] * 100:+.1f} points; ahead on "
          f"{lv['dates_ahead']}, tied on {lv['dates_tied']}, behind on {lv['dates_behind']} of "
          f"{len(C.TEST_DATES)} dates")
    u = tp["uncertainty_logistic_vs_baseline"]["precision_at_100"]
    lo, hi = u["ci95_block_bootstrap"]
    print(f"  95% block-bootstrap interval: [{lo * 100:+.1f}, {hi * 100:+.1f}] points")
    v = tp["violations_found_in_top_100_per_date"]
    print("violations cited in the top 100 per date: " +
          ", ".join(f"{k} {x:.0f}" for k, x in v.items()))
    print(f"total runtime: {time.time() - t0:.0f}s")
    return summary
