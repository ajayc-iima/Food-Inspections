#!/usr/bin/env python3
"""Business impact: how much sooner would failures be caught if the city's own
routine inspections were done in the model's order?

Same design as Chicago's 2014 pilot (CDPH/DoIT): for each decision date T, take
the routine inspections the city actually carried out in the next 90 days, keep
the same set of inspection days (the same capacity), and hand those days out in
the order each approach ranks the establishments. A failure found on day 60 in
reality but placed on day 20 is caught 40 days earlier.

Models are the pre-registered ones from confirm_local.py, refitted identically on
2016-2019. Assumes an establishment's result does not depend on when in the
quarter it is inspected (the pilot's assumption too). Writes
results/days_earlier.json.
"""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from confirm_nonroutine import HOLDOUT_DATE, RAW_OTHER, add_features
from experiment_nonroutine import clean_other
from food_inspections import config as C
from food_inspections.analysis import block_bootstrap_diff
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import rank_order
from food_inspections.features import features_at_date, load_panel
from food_inspections.local_features import LOCAL_COLUMNS, build_local
from food_inspections.models import make_logistic

HORIZON = 90
OUTPUT = C.RESULTS_DIR / "days_earlier.json"


def reorder(q: pd.DataFrame, score: pd.Series) -> dict:
    """q: one date's inspections within the horizon; returns per-date impact."""
    actual = q["days_to_next"].to_numpy(dtype=float)
    slots = np.sort(actual)
    assigned = np.empty(len(q))
    assigned[rank_order(q, score.loc[q.index])] = slots
    fail = q["label"].to_numpy() == 1
    half = np.median(slots)
    return {"days_earlier_per_failure": float((actual[fail] - assigned[fail]).mean()),
            "failure_days_saved": float((actual[fail] - assigned[fail]).sum()),
            "share_failures_in_first_half": float((assigned[fail] <= half).mean()),
            "share_failures_in_first_half_actual": float((actual[fail] <= half).mean()),
            "inspections": int(len(q)), "failures": int(fail.sum())}


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    panel = load_panel(df)
    holdout = features_at_date(df, HOLDOUT_DATE)
    holdout["period"] = "holdout"
    rows = pd.concat([panel, holdout], ignore_index=True)
    rows = rows.merge(build_local(df, rows), on=["decision_date", "license_"], how="left", validate="one_to_one")
    rows, Xb, Xn = add_features(rows, clean_other(pd.read_csv(RAW_OTHER, dtype=str, low_memory=False)))
    Xl = pd.concat([Xn, rows[LOCAL_COLUMNS].astype(float)], axis=1)
    y = rows["label"].astype(int)
    train = rows["period"] == "train"
    scores = {"history rule": rows["hist_fail_rate_smooth"]}
    for tag, X in [("logistic", Xb), ("final model", Xl)]:
        m = make_logistic().fit(X[train], y[train])
        scores[tag] = pd.Series(m.predict_proba(X)[:, 1], index=rows.index)

    out = {"horizon_days": HORIZON, "design": __doc__.split("\n\n")[1].replace("\n", " ")}
    for period, key in [("test", "period_2022_2025"), ("holdout", "date_2025_09_30")]:
        per = {a: {} for a in scores}
        for T, r in rows[rows["period"] == period].groupby("decision_date"):
            q = r[r["days_to_next"] <= HORIZON]
            for a, s in scores.items():
                per[a][str(pd.Timestamp(T).date())] = reorder(q, s)
        summ = {}
        for a in scores:
            v = pd.DataFrame(per[a]).T
            summ[a] = {"mean_days_earlier_per_failure": float(v["days_earlier_per_failure"].mean()),
                       "mean_failure_days_saved_per_quarter": float(v["failure_days_saved"].mean()),
                       "share_failures_in_first_half": float(v["share_failures_in_first_half"].mean()),
                       "share_failures_in_first_half_actual": float(v["share_failures_in_first_half_actual"].mean()),
                       "mean_inspections_per_quarter": float(v["inspections"].mean()),
                       "mean_failures_per_quarter": float(v["failures"].mean())}
        block = {"summary": summ, "per_date": per}
        if period == "test":
            comps = {}
            for a, b in [("final model", "history rule"), ("final model", "logistic"),
                         ("history rule", "actual order")]:
                da = np.array([per[a][k]["days_earlier_per_failure"] for k in sorted(per[a])])
                db = (np.zeros_like(da) if b == "actual order" else
                      np.array([per[b][k]["days_earlier_per_failure"] for k in sorted(per[b])]))
                d = da - db
                comps[f"{a} minus {b} (days earlier per failure)"] = {
                    "mean": float(d.mean()), "ci95": list(block_bootstrap_diff(d, block=C.LABEL_WINDOW_DAYS // 91)),
                    "dates_ahead": int((d > 0).sum()), "dates_behind": int((d < 0).sum())}
            d = np.array([per["final model"][k]["days_earlier_per_failure"] for k in sorted(per["final model"])])
            comps["final model minus actual order (days earlier per failure)"] = {
                "mean": float(d.mean()), "ci95": list(block_bootstrap_diff(d, block=C.LABEL_WINDOW_DAYS // 91)),
                "dates_ahead": int((d > 0).sum()), "dates_behind": int((d < 0).sum())}
            block["comparisons"] = comps
        out[key] = block
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key in ["period_2022_2025", "date_2025_09_30"]:
        print(key)
        for a, v in out[key]["summary"].items():
            print("  ", a, {k: round(x, 3) for k, x in v.items()})
    for k, v in out["period_2022_2025"]["comparisons"].items():
        print("  ", k, round(v["mean"], 2), [round(x, 2) for x in v["ci95"]], v["dates_ahead"], v["dates_behind"])


if __name__ == "__main__":
    main()
