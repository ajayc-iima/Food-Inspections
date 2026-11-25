#!/usr/bin/env python3
"""Test past non-routine Chicago inspection history without changing the main model.

The source extract is a current snapshot, not a historical publication-time
archive. Inspection dates bound the features, but later corrections to records
cannot be ruled out. Model comparisons are exploratory: the folds and later
period have been examined in earlier experiments.

Run with --download once to cache a separate public-data extract, then rerun
without it. Nothing overwrites the original raw snapshot or cached panel.
"""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from food_inspections import config as C
from food_inspections.data import count_violations, clean, load_raw
from food_inspections.evaluation import mean_precision_at_k
from food_inspections.features import load_panel
from food_inspections.models import check_purged, design_matrix, make_logistic, make_neural_network

RAW_OTHER = C.DATA_DIR / "raw_other_inspections_through_2025_06_30.csv.gz"
OUTPUT = C.RESULTS_DIR / "experiment_nonroutine.json"
EXTRA = ["nr_total", "nr_fail", "nr_complaint_total", "nr_complaint_fail",
         "nr_12m", "nr_fail_12m", "nr_complaint_12m", "nr_complaint_fail_12m",
         "nr_reinspect_12m", "nr_reinspect_fail_12m", "nr_days_last", "nr_days_last_fail",
         "nr_days_last_complaint", "nr_last_failed", "nr_viol_last"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def download():
    frames = []
    offset = 0
    while True:
        params = {
            "$select": "inspection_id,license_,inspection_date,inspection_type,results,violations",
            "$where": "inspection_type != 'Canvass' AND inspection_date <= '2025-06-30T23:59:59'",
            "$order": "inspection_id",
            "$limit": "50000", "$offset": str(offset),
        }
        url = f"{C.DATASET_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=300) as response:
            page = pd.read_csv(StringIO(response.read().decode("utf-8")), dtype=str)
        print(f"[download] offset {offset:,}: {len(page):,} rows", flush=True)
        frames.append(page)
        if len(page) < 50000:
            break
        offset += 50000
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("inspection_id")
    if raw.empty:
        raise ValueError("Non-routine extract is empty")
    raw.to_csv(RAW_OTHER, index=False, compression="gzip")
    return raw


def clean_other(raw):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["inspection_date"], errors="coerce").dt.normalize()
    df["license_"] = df["license_"].astype("string").str.strip()
    df = df[df["date"].notna() & df["license_"].notna()
            & ~df["license_"].isin(["", "0", "nan"])].copy()
    df["inspection_type"] = df["inspection_type"].fillna("").str.strip()
    df = df[df["inspection_type"].str.lower() != "canvass"].copy()
    df["results"] = df["results"].fillna("").str.strip()
    df["is_fail"] = df["results"].eq("Fail").astype(int)
    kind = df["inspection_type"].str.lower()
    df["is_complaint"] = kind.str.contains("complaint", regex=False).astype(int)
    df["is_reinspect"] = kind.str.contains("re-inspection", regex=False).astype(int)
    df["complaint_fail"] = df["is_complaint"] * df["is_fail"]
    df["reinspect_fail"] = df["is_reinspect"] * df["is_fail"]
    df["viol_count"] = df["violations"].map(count_violations).astype(int)
    df = df.sort_values(["license_", "date", "inspection_id"], kind="mergesort")
    return df.drop_duplicates(["license_", "date", "inspection_type", "results"]).reset_index(drop=True)


def history_features(df, date):
    """Return past features indexed by licence; no records after date contribute."""
    hist = df[df["date"] <= date]
    if hist.empty:
        return pd.DataFrame(columns=["license_", *EXTRA])
    g = hist.groupby("license_", sort=False)
    all_counts = g.agg(nr_total=("is_fail", "size"), nr_fail=("is_fail", "sum"),
                       nr_complaint_total=("is_complaint", "sum"),
                       nr_complaint_fail=("complaint_fail", "sum"))
    recent = hist[hist["date"] > date - pd.Timedelta(days=365)]
    recent_counts = recent.groupby("license_").agg(
        nr_12m=("is_fail", "size"), nr_fail_12m=("is_fail", "sum"),
        nr_complaint_12m=("is_complaint", "sum"), nr_complaint_fail_12m=("complaint_fail", "sum"),
        nr_reinspect_12m=("is_reinspect", "sum"), nr_reinspect_fail_12m=("reinspect_fail", "sum"))
    last = g.tail(1).set_index("license_")
    last_fail = hist[hist["is_fail"].eq(1)].groupby("license_")["date"].max()
    last_complaint = hist[hist["is_complaint"].eq(1)].groupby("license_")["date"].max()
    out = all_counts.join(recent_counts)
    out["nr_days_last"] = (date - last["date"]).dt.days.clip(upper=3650)
    out["nr_days_last_fail"] = (date - last_fail.reindex(out.index)).dt.days.clip(upper=3650)
    out["nr_days_last_complaint"] = (date - last_complaint.reindex(out.index)).dt.days.clip(upper=3650)
    out["nr_last_failed"] = last["is_fail"]
    out["nr_viol_last"] = last["viol_count"]
    out = out.fillna({"nr_days_last_fail": 3650, "nr_days_last_complaint": 3650}).fillna(0)
    return out[EXTRA].reset_index()


def build_matrix(panel, other):
    frames = []
    for date in sorted(panel["decision_date"].unique()):
        date = pd.Timestamp(date)
        features = history_features(other, date)
        features["decision_date"] = date
        frames.append(features)
    rows = panel.merge(pd.concat(frames, ignore_index=True), on=["decision_date", "license_"],
                       how="left", validate="many_to_one")
    for col in EXTRA:
        rows[col] = rows[col].fillna(3650 if col.startswith("nr_days_") else 0.0)
    original = design_matrix(rows)
    return rows, original, pd.concat([original, rows[EXTRA].astype(float)], axis=1)


@threadpool_limits.wrap(limits=1, user_api="blas")
def run(panel, other):
    panel, original, extended = build_matrix(panel, other)
    y = panel["label"].astype(int)
    candidates = {
        "logistic": lambda: make_logistic(),
        "neural": lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0}),
    }
    out = {"description": "Separate exploratory non-routine inspection feature comparison",
           "source": C.DATASET_PAGE,
           "source_note": "Downloaded after the original snapshot; inspection dates are known, "
                          "but later publication or record corrections cannot be ruled out.",
           "other_raw_sha256": sha256(RAW_OTHER), "original_panel_sha256": sha256(C.PANEL_FILE),
           "other_cleaned_rows": int(len(other)), "features": EXTRA,
           "folds": {}, "validation_means": {}, "exploratory_test": {}}
    for name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        if not check_purged(spec["fit"], spec["val"]):
            raise ValueError(f"unpurged fold: {name}")
        fit = panel["decision_date"].isin(spec["fit"])
        val = panel["decision_date"].isin(spec["val"])
        fold = {}
        for candidate, factory in candidates.items():
            for key, matrix in [("original", original), ("nonroutine", extended)]:
                model = factory().fit(matrix[fit], y[fit])
                score = pd.Series(model.predict_proba(matrix[val])[:, 1], index=panel.index[val])
                tag = f"{candidate}_{key}"
                fold[tag] = mean_precision_at_k(panel[val], score)
                if candidate == "neural":
                    fold[f"{tag}_epochs"] = int(model.named_steps["clf"].n_iter_)
                print(f"[val] {name}: {tag}={fold[tag]:.4f}", flush=True)
        out["folds"][name] = fold
    for tag in ["logistic_original", "logistic_nonroutine", "neural_original", "neural_nonroutine"]:
        out["validation_means"][tag] = float(np.mean([f[tag] for f in out["folds"].values()]))

    fit = panel["period"].eq("train")
    val = panel["period"].eq("test")
    out["exploratory_test"]["note"] = "Previously examined 2022-2025 decision dates; not an untouched test."
    for candidate, factory in candidates.items():
        for key, matrix in [("original", original), ("nonroutine", extended)]:
            model = factory().fit(matrix[fit], y[fit])
            score = pd.Series(model.predict_proba(matrix[val])[:, 1], index=panel.index[val])
            tag = f"{candidate}_{key}"
            out["exploratory_test"][tag] = mean_precision_at_k(panel[val], score)
            if candidate == "neural":
                out["exploratory_test"][f"{tag}_epochs"] = int(model.named_steps["clf"].n_iter_)
            print(f"[exploratory test] {tag}={out['exploratory_test'][tag]:.4f}", flush=True)
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download a separate non-routine extract")
    args = parser.parse_args()
    raw = download() if args.download else pd.read_csv(RAW_OTHER, dtype=str, low_memory=False)
    other = clean_other(raw)
    panel = load_panel(clean(load_raw()))
    run(panel, other)


if __name__ == "__main__":
    main()
