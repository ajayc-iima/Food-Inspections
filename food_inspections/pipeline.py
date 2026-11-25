"""End-to-end run: data -> panel -> model selection -> models -> evaluation -> outputs."""
import json
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from . import charts, config as C
from .analysis import shortlist_profiles, uncertainty
from .data import clean, load_raw
from .evaluation import run_evaluation, shortlists, summarise
from .features import load_panel
from .models import (BASELINE_NAME, EXTRA_TREES_CONFIGS, LOGISTIC_CONFIGS, NEURAL_CONFIGS,
                     SPLINE_CONFIGS, SPLINE_FEATURES,
                     SPLINE_PARAMS, design_matrix, logistic_coefficients, make_candidate,
                     make_logistic, make_xgboost, select_model)


@threadpool_limits.wrap(limits=1, user_api="blas")
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

    # 3. choose among candidate families using training dates only
    print("[select] validating model candidates on 2016-2019 decision dates ...")
    train_panel = panel[is_train]
    selected, selected_xgb, table = select_model(train_panel, X[is_train])
    table.to_csv(C.RESULTS_DIR / "model_selection.csv", index_label="config")
    print(table.round(3).to_string())
    print(f"[select] chosen: {selected}; XGBoost reference: {selected_xgb}")
    criterion = "criterion (mean of purged)"
    selected_neural = table.loc[list(NEURAL_CONFIGS), criterion].idxmax()
    selected_extra = table.loc[list(EXTRA_TREES_CONFIGS), criterion].idxmax()
    print(f"[select] neural reference: {selected_neural}; Extra Trees reference: {selected_extra}")

    # 4. fit on all training dates, score everything
    print("[model] fitting logistic regression and XGBoost on all training dates ...")
    lr = make_logistic().fit(X[is_train], y[is_train])
    xgb = make_xgboost(C.XGB_CONFIGS[selected_xgb]).fit(X[is_train], y[is_train])
    panel["score_logistic"] = lr.predict_proba(X)[:, 1]
    panel["score_xgboost"] = xgb.predict_proba(X)[:, 1]
    neural = make_candidate(selected_neural).fit(X[is_train], y[is_train])
    extra = make_candidate(selected_extra).fit(X[is_train], y[is_train])
    panel["score_neural"] = neural.predict_proba(X)[:, 1]
    panel["score_extra_trees"] = extra.predict_proba(X)[:, 1]
    if selected == BASELINE_NAME:
        panel["score_selected"] = panel["hist_fail_rate_smooth"]
    elif selected == "logistic (reference)":
        panel["score_selected"] = panel["score_logistic"]
    elif selected == selected_xgb:
        panel["score_selected"] = panel["score_xgboost"]
    elif selected == selected_neural:
        panel["score_selected"] = panel["score_neural"]
    elif selected == selected_extra:
        panel["score_selected"] = panel["score_extra_trees"]
    else:
        winner = make_candidate(selected).fit(X[is_train], y[is_train])
        panel["score_selected"] = winner.predict_proba(X)[:, 1]

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
            "hist_fail_rate_smooth", "score_logistic", "score_xgboost", "score_neural",
            "score_extra_trees", "score_selected", "label"]
    latest = sl[sl["decision_date"] == sl["decision_date"].max()]
    latest[keep].to_csv(C.RESULTS_DIR / "shortlist_latest.csv", index=False)

    profile, facility_mix = shortlist_profiles(panel, "baseline", "logistic")
    profile.to_csv(C.RESULTS_DIR / "shortlist_profiles.csv", index_label="feature (mean)")
    facility_mix.to_csv(C.RESULTS_DIR / "shortlist_facility_mix.csv", index_label="facility")

    summary = summarise(df, panel, metrics, selected_xgb)
    family = ("baseline" if selected == BASELINE_NAME else
              "logistic" if selected in LOGISTIC_CONFIGS else
              "spline_logistic" if selected in SPLINE_CONFIGS else
              "neural" if selected in NEURAL_CONFIGS else
              "extra_trees" if selected in EXTRA_TREES_CONFIGS else "xgboost")
    params = ({"C": LOGISTIC_CONFIGS[selected]} if family == "logistic" else
              {"C": SPLINE_CONFIGS[selected], "spline_features": SPLINE_FEATURES,
               "spline": SPLINE_PARAMS} if family == "spline_logistic" else
              NEURAL_CONFIGS[selected] if family == "neural" else
              EXTRA_TREES_CONFIGS[selected] if family == "extra_trees" else
              C.XGB_CONFIGS[selected] if family == "xgboost" else {})
    summary["selected_model"] = {
        "name": selected, "family": family, "params": params,
        "validation_mean_precision_at_k": float(table.loc[selected, "criterion (mean of purged)"]),
        "selection_rule": "highest mean Precision@K across purged training folds",
    }
    summary["neural_reference"] = {
        "name": selected_neural, "params": NEURAL_CONFIGS[selected_neural],
        "solver": "adam", "batch_size": 1024, "max_iter": 60,
        "n_iter_no_change": 5, "early_stopping": False, "random_state": C.SEED,
        "fit_epochs": int(neural.named_steps["clf"].n_iter_),
        "validation_mean_precision_at_k": float(table.loc[selected_neural, criterion]),
    }
    summary["extra_trees_reference"] = {
        "name": selected_extra, "params": EXTRA_TREES_CONFIGS[selected_extra],
        "validation_mean_precision_at_k": float(table.loc[selected_extra, criterion]),
    }
    summary["test_period"]["uncertainty_logistic_vs_baseline"] = uncertainty(metrics, "logistic")
    summary["test_period"]["uncertainty_xgboost_vs_baseline"] = uncertainty(metrics, "xgboost")
    summary["test_period"]["uncertainty_selected_vs_baseline"] = uncertainty(metrics, "selected")
    summary["test_period"]["uncertainty_neural_vs_baseline"] = uncertainty(metrics, "neural")
    summary["test_period"]["uncertainty_extra_trees_vs_baseline"] = uncertainty(metrics, "extra_trees")
    with open(C.RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[charts] rendering figures ...")
    charts.render_all(df, panel, metrics, importance, coef, profile, facility_mix)

    tp = summary["test_period"]
    print("\n============ EXPLORATORY RESULTS (previously examined 2022-2025 period) ============")
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
