#!/usr/bin/env python3
"""Adoption pack: a live shortlist with reasons, and a fairness check.

1. Live shortlist as of the latest quarter-end in the snapshot (2026-09-30): every
   establishment with a routine inspection in the last two years that is not
   recorded as out of business, scored by the pre-registered final model
   (trained on 2016-2019, as validated). Each of the top 100 gets the three
   facts that pushed it up the list, in plain language.
2. Fairness check on 2022-2025: does the final model concentrate visits in a
   few ZIP codes, or in the strictest areas, more than the history rule?

    python adoption_pack.py --download   # first run: non-routine data to 2026-09-30
Writes results/shortlist_2026_q4.csv and results/adoption_pack.json.
"""
import argparse
import json
import urllib.parse
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from confirm_nonroutine import add_features
from experiment_nonroutine import clean_other
from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import rank_order, top_k
from food_inspections.features import features_at_date, load_panel
from food_inspections.local_features import LOCAL_COLUMNS, build_local
from food_inspections.models import make_logistic

LIVE_DATE = pd.Timestamp("2026-09-30")
ZIP_CAP = 5  # guardrail: no ZIP code gets more than 5 of the 100 early slots (fixed before looking)
RAW_OTHER_LIVE = C.DATA_DIR / "raw_other_inspections_through_2026_09_30.csv.gz"
FACILITY_WORDS = {"fac_School": "a school kitchen", "fac_Long Term Care": "a long-term care home",
                  "fac_Hospital": "a hospital kitchen", "fac_Bakery": "a bakery",
                  "fac_Daycare (2 Under 2 Years)": "a day-care centre", "fac_Daycare Above 2 Years": "a day-care centre",
                  "fac_Grocery Store": "a grocery store", "fac_Restaurant": "a restaurant"}


def download():
    frames, offset = [], 0
    while True:
        params = {"$select": "inspection_id,license_,inspection_date,inspection_type,results,violations",
                  "$where": f"inspection_type != 'Canvass' AND inspection_date <= '{LIVE_DATE.date()}T23:59:59'",
                  "$order": "inspection_id", "$limit": "50000", "$offset": str(offset)}
        with urllib.request.urlopen(f"{C.DATASET_URL}?{urllib.parse.urlencode(params)}", timeout=300) as r:
            page = pd.read_csv(StringIO(r.read().decode("utf-8")), dtype=str)
        print(f"[download] offset {offset:,}: {len(page):,} rows", flush=True)
        frames.append(page)
        if len(page) < 50000:
            break
        offset += 50000
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("inspection_id")
    raw.to_csv(RAW_OTHER_LIVE, index=False, compression="gzip")
    return raw


def reason(f: str, r: pd.Series, city: dict) -> str | None:
    """Plain-language fact behind a feature that raised the score."""
    if f in ("viol_mean_last3", "viol_mean_24m", "viol_last"):
        if max(r["viol_last"], r["viol_mean_last3"]) <= city["viol"]:
            return None
        return (f"{r['viol_last']:.0f} violations at its last routine inspection, {r['viol_mean_last3']:.0f} on "
                f"average over its last 3 (typical: {city['viol']:.0f})")
    if f in ("n_fail_prev", "hist_fail_rate_smooth", "has_failed_before"):
        if r["n_fail_prev"] < 1:
            return None
        return f"failed {r['n_fail_prev']:.0f} of its {r['n_outcome_prev']:.0f} past routine inspections"
    if f in ("n_fail_24m", "fail_rate_24m"):
        if r["n_fail_24m"] < 1:
            return None
        return f"failed {r['n_fail_24m']:.0f} routine inspection(s) in the last 2 years"
    if f == "days_since_last_canvass":
        return f"{r['days_since_last_canvass']:.0f} days since its last routine inspection"
    if f.startswith("near_rate") or f.startswith("zip_rate") or f == "near_rate_trend":
        if r["near_rate_180d"] <= city["near"]:
            return None  # only state it when the area really is above the city's typical rate
        return (f"routine inspections within 1 km failed {r['near_rate_180d']:.0%} of the time in the last "
                f"6 months (city typical: {city['near']:.0%})")
    if f in ("nr_complaint_12m", "nr_complaint_total", "nr_complaint_fail", "nr_complaint_fail_12m",
             "nr_days_last_complaint"):
        if r["nr_complaint_total"] < 1:
            return None
        return f"{r['nr_complaint_total']:.0f} complaint inspection(s) on record, {r['nr_complaint_12m']:.0f} in the last year"
    if f in ("nr_fail", "nr_fail_12m", "nr_last_failed", "nr_viol_last", "nr_reinspect_fail_12m"):
        if r["nr_fail"] < 1:
            return None
        return f"failed {r['nr_fail']:.0f} non-routine inspection(s) (complaint or re-inspection)"
    if f == "risk_latest" and r["risk_latest"] == 1:
        return "risk category 1 (highest)"
    if f in FACILITY_WORDS and r.get(f, 0) == 1:
        return FACILITY_WORDS[f]
    return None


def fit_final(rows, Xl, y):
    train = rows["period"] == "train"
    return make_logistic().fit(Xl[train], y[train])


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    raw_other = download() if args.download or not RAW_OTHER_LIVE.exists() else \
        pd.read_csv(RAW_OTHER_LIVE, dtype=str, low_memory=False)
    other = clean_other(raw_other)
    df = clean(load_raw())
    panel = load_panel(df)

    live = features_at_date(df, LIVE_DATE, require_label=False)
    last_result = (df[df["date"] <= LIVE_DATE].groupby("license_")["results"].last())
    live = live[live["license_"].map(last_result).isin(["Pass", "Fail", "Pass w/ Conditions", "No Entry",
                                                         "Not Ready"])]
    live = live[live["n_canvass_24m"] > 0].copy()
    live["period"] = "live"
    rows = pd.concat([panel, live], ignore_index=True)
    rows = rows.merge(build_local(df, rows), on=["decision_date", "license_"], how="left", validate="one_to_one")
    rows, Xb, Xn = add_features(rows, other)
    Xl = pd.concat([Xn, rows[LOCAL_COLUMNS].astype(float)], axis=1)
    y = rows["label"].fillna(0).astype(int)
    model = fit_final(rows, Xl, y)
    rows["score_final"] = model.predict_proba(Xl)[:, 1]

    # 1. live shortlist with reasons
    lv = rows[rows["period"] == "live"].copy()
    ordered = lv.iloc[rank_order(lv, lv["score_final"])]
    top = ordered[ordered.groupby("zip", dropna=False).cumcount() < ZIP_CAP].head(C.K).copy()  # with guardrail
    scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
    contrib = pd.DataFrame(scaler.transform(Xl.loc[top.index]) * clf.coef_[0], index=top.index, columns=Xl.columns)
    city = {"viol": float(lv["viol_mean_last3"].median()), "near": float(lv["near_rate_180d"].median())}
    reasons = []
    for i, r in top.iterrows():
        facts, seen = [], set()
        rr = r.combine_first(Xl.loc[i])  # raw values, plus one-hot facility columns
        for f in contrib.loc[i].sort_values(ascending=False).index:
            if contrib.loc[i, f] <= 0 or len(facts) == 3:
                break
            t = reason(f, rr, city)
            if t and t not in seen:
                facts.append(t); seen.add(t)
        reasons.append(facts + [""] * (3 - len(facts)))
    top["rank"] = np.arange(1, len(top) + 1)
    top[["reason_1", "reason_2", "reason_3"]] = reasons
    cols = ["rank", "license_", "dba_name", "zip", "facility_grouped", "risk_latest", "days_since_last_canvass",
            "score_final", "reason_1", "reason_2", "reason_3"]
    top[cols].rename(columns={"score_final": "estimated_fail_probability"}).to_csv(
        C.RESULTS_DIR / "shortlist_2026_q4.csv", index=False)

    # 2. fairness check on 2022-2025, and a simple guardrail: at most ZIP_CAP of the 100 per ZIP code
    te = rows[rows["period"] == "test"]
    variants = [("history rule", "hist_fail_rate_smooth", False), ("final model", "score_final", False),
                ("history rule, ZIP cap", "hist_fail_rate_smooth", True),
                ("final model, ZIP cap", "score_final", True)]
    stats = {v[0]: [] for v in variants}
    for T, r in te.groupby("decision_date"):
        q5 = pd.qcut(r["near_rate_180d"].rank(method="first"), 5, labels=False)
        for name, col, cap in variants:
            order = r.iloc[rank_order(r, r[col])]
            if cap:
                order = order[order.groupby("zip", dropna=False).cumcount() < ZIP_CAP]
            t = order.head(C.K)
            zc = t["zip"].value_counts()
            stats[name].append({"precision_at_100": float(t["label"].mean()),
                                "distinct_zips": int(zc.size), "top5_zip_share": float(zc.head(5).sum() / len(t)),
                                "max_from_one_zip": int(zc.max()),
                                "share_from_strictest_fifth_of_areas": float((q5.loc[t.index] == 4).mean()),
                                "share_from_most_lenient_fifth": float((q5.loc[t.index] == 0).mean())})
    fair = {k: pd.DataFrame(v).mean().round(4).to_dict() for k, v in stats.items()}
    fair["zip_cap"] = ZIP_CAP
    fair["all candidates"] = {"distinct_zips": float(te.groupby("decision_date")["zip"].nunique().mean()),
                              "share_from_strictest_fifth_of_areas": 0.2, "share_from_most_lenient_fifth": 0.2}
    out = {"live_date": str(LIVE_DATE.date()), "live_candidates": int(len(lv)),
           "live_note": "Live list scores every establishment with a routine inspection in the last 2 years "
                        "that is not out of business; it has no outcome yet.",
           "live_top100_facility_mix": top["facility_grouped"].value_counts().to_dict(),
           "live_top100_distinct_zips": int(top["zip"].nunique()),
           "fairness_2022_2025_mean_per_date": fair,
           "example_rows": top[cols].head(5).to_dict(orient="records")}
    (C.RESULTS_DIR / "adoption_pack.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["live_candidates", "live_top100_facility_mix", "live_top100_distinct_zips",
                                          "fairness_2022_2025_mean_per_date"]}, indent=1))
    print(top[cols].head(8).to_string())


if __name__ == "__main__":
    main()
