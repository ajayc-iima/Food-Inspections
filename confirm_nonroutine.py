#!/usr/bin/env python3
"""Pre-registered confirmation of the non-routine history features.

Implements docs/preregistration_nonroutine.md exactly: validation gate on the purged
2016-2019 folds, one scoring of the untouched 2025-09-30 decision date, and a
re-statement of 2022-2025. Writes results/confirm_nonroutine.json only.

    python confirm_nonroutine.py --download   # first run: fetch non-routine data to 2025-09-30
    python confirm_nonroutine.py              # reuse the cached extract
"""
import argparse
import json
import urllib.parse
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiment_nonroutine import EXTRA, clean_other, history_features, sha256
from food_inspections import config as C
from food_inspections.analysis import block_bootstrap_diff
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import mean_precision_at_k, rank_order
from food_inspections.features import features_at_date, load_panel
from food_inspections.models import check_purged, design_matrix, make_logistic, make_neural_network

HOLDOUT_DATE = pd.Timestamp("2025-09-30")
RAW_OTHER = C.DATA_DIR / "raw_other_inspections_through_2025_09_30.csv.gz"
OUTPUT = C.RESULTS_DIR / "confirm_nonroutine.json"
N_BOOT = 2000


def download():
    frames, offset = [], 0
    while True:
        params = {
            "$select": "inspection_id,license_,inspection_date,inspection_type,results,violations",
            "$where": "inspection_type != 'Canvass' AND inspection_date <= '2025-09-30T23:59:59'",
            "$order": "inspection_id", "$limit": "50000", "$offset": str(offset),
        }
        with urllib.request.urlopen(f"{C.DATASET_URL}?{urllib.parse.urlencode(params)}", timeout=300) as r:
            page = pd.read_csv(StringIO(r.read().decode("utf-8")), dtype=str)
        print(f"[download] offset {offset:,}: {len(page):,} rows", flush=True)
        frames.append(page)
        if len(page) < 50000:
            break
        offset += 50000
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("inspection_id")
    raw.to_csv(RAW_OTHER, index=False, compression="gzip")
    return raw


def add_features(panel, other):
    frames = []
    for date in sorted(panel["decision_date"].unique()):
        f = history_features(other, pd.Timestamp(date))
        f["decision_date"] = pd.Timestamp(date)
        frames.append(f)
    rows = panel.merge(pd.concat(frames, ignore_index=True), on=["decision_date", "license_"],
                       how="left", validate="many_to_one")
    for col in EXTRA:
        rows[col] = rows[col].fillna(3650 if col.startswith("nr_days_") else 0.0)
    base = design_matrix(rows)
    return rows, base, pd.concat([base, rows[EXTRA].astype(float)], axis=1)


def per_date_precision(rows, score):
    out = {}
    for T, r in rows.groupby("decision_date"):
        order = rank_order(r, score.loc[r.index])
        out[str(pd.Timestamp(T).date())] = float(r["label"].to_numpy()[order[:C.K]].mean())
    return out


def single_date_bootstrap(rows, s_a, s_b, seed=C.SEED):
    """95% interval for P@100(a) - P@100(b) on one date, resampling candidates."""
    rng = np.random.default_rng(seed)
    y = rows["label"].to_numpy()
    lic = rows[["license_"]].reset_index(drop=True)
    a, b = np.asarray(s_a), np.asarray(s_b)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(rows), len(rows))
        sub = lic.iloc[idx].reset_index(drop=True)
        oa, ob = rank_order(sub, a[idx])[:C.K], rank_order(sub, b[idx])[:C.K]
        diffs[i] = y[idx][oa].mean() - y[idx][ob].mean()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


@threadpool_limits.wrap(limits=1, user_api="blas")
def run(df, other):
    panel = load_panel(df)
    holdout = features_at_date(df, HOLDOUT_DATE)
    if df["date"].max() < HOLDOUT_DATE + pd.Timedelta(days=C.LABEL_WINDOW_DAYS):
        raise SystemExit("Holdout label window is not complete in this snapshot")
    holdout["period"] = "holdout"
    holdout["panel_feature_version"] = panel["panel_feature_version"].iloc[0]
    holdout["label_date"] = holdout["decision_date"] + pd.to_timedelta(holdout["days_to_next"], unit="D")
    allrows = pd.concat([panel, holdout], ignore_index=True)
    rows, Xb, Xn = add_features(allrows, other)
    y = rows["label"].astype(int)
    out = {"preregistration": "docs/preregistration_nonroutine.md", "holdout_date": str(HOLDOUT_DATE.date()),
           "other_raw_sha256": sha256(RAW_OTHER), "panel_sha256": sha256(C.PANEL_FILE),
           "holdout_candidates": int((rows["period"] == "holdout").sum()),
           "holdout_base_rate": float(rows.loc[rows["period"] == "holdout", "label"].mean())}

    # 1. validation gate on purged folds
    gate = {}
    for name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        assert check_purged(spec["fit"], spec["val"])
        fit, val = rows["decision_date"].isin(spec["fit"]), rows["decision_date"].isin(spec["val"])
        res = {}
        for tag, X in [("logistic", Xb), ("logistic_nonroutine", Xn)]:
            m = make_logistic().fit(X[fit], y[fit])
            res[tag] = mean_precision_at_k(rows[val], pd.Series(m.predict_proba(X[val])[:, 1], index=rows.index[val]))
        gate[name] = res
    gate_gain = float(np.mean([g["logistic_nonroutine"] - g["logistic"] for g in gate.values()]))
    out["validation_gate"] = {"folds": gate, "mean_gain_vs_logistic": gate_gain}

    # 2 + 3. fit on all training dates, score holdout and 2022-25 once
    train = rows["period"] == "train"
    scores = {"baseline": rows["hist_fail_rate_smooth"]}
    for tag, factory, X in [("logistic", make_logistic, Xb), ("logistic_nonroutine", make_logistic, Xn),
                            ("neural", lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0}), Xb),
                            ("neural_nonroutine", lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0}), Xn)]:
        m = factory().fit(X[train], y[train])
        scores[tag] = pd.Series(m.predict_proba(X)[:, 1], index=rows.index)

    for period, key in [("holdout", "untouched_holdout_2025_09_30"), ("test", "previously_examined_2022_2025")]:
        sel = rows["period"] == period
        r = rows[sel]
        prec = {tag: per_date_precision(r, s) for tag, s in scores.items()}
        block = {"precision_at_100": {t: float(np.mean(list(v.values()))) for t, v in prec.items()}}
        comps = {}
        for a, b in [("logistic_nonroutine", "logistic"), ("logistic_nonroutine", "baseline"),
                     ("logistic", "baseline"), ("neural_nonroutine", "neural")]:
            d = np.array([prec[a][k] - prec[b][k] for k in sorted(prec[a])])
            if period == "holdout":
                ci = single_date_bootstrap(r, scores[a].loc[r.index], scores[b].loc[r.index])
                method = "bootstrap over candidate establishments"
            else:
                ci = list(block_bootstrap_diff(d, block=C.LABEL_WINDOW_DAYS // 91))
                method = "4-quarter moving-block bootstrap over dates"
            comps[f"{a} minus {b}"] = {"mean_difference": float(d.mean()), "ci95": ci, "method": method,
                                       "dates_ahead": int((d > 0).sum()), "dates_tied": int((d == 0).sum()),
                                       "dates_behind": int((d < 0).sum())}
        block["comparisons"] = comps
        if period == "test":
            block["note"] = "Not untouched: the exploratory script already reported these dates."
        out[key] = block

    h = out["untouched_holdout_2025_09_30"]["comparisons"]["logistic_nonroutine minus logistic"]
    t = out["previously_examined_2022_2025"]["comparisons"]["logistic_nonroutine minus logistic"]
    rule = {"a_validation_gain_above_zero": gate_gain > 0,
            "b_holdout_not_worse_than_logistic": h["mean_difference"] >= 0,
            "c_2022_2025_interval_excludes_zero": t["ci95"][0] > 0 or t["ci95"][1] < 0}
    rule["confirmed"] = all(rule.values())
    out["decision_rule"] = rule
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["holdout_candidates", "holdout_base_rate", "validation_gate",
                                          "untouched_holdout_2025_09_30", "decision_rule"]}, indent=1))
    print(f"Saved {OUTPUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    raw_other = download() if args.download or not RAW_OTHER.exists() else pd.read_csv(RAW_OTHER, dtype=str, low_memory=False)
    run(clean(load_raw()), clean_other(raw_other))


if __name__ == "__main__":
    main()
