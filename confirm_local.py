#!/usr/bin/env python3
"""Pre-registered test of local recent failure rates + complaint history.

Implements docs/preregistration_local.md: fit on 2016-2019, score 2022-2025 and
2025-09-30 once, compare with the baseline, plain logistic and logistic +
complaint history. Writes results/confirm_local.json only.
"""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from confirm_nonroutine import (HOLDOUT_DATE, RAW_OTHER, add_features, per_date_precision,
                                single_date_bootstrap)
from experiment_nonroutine import EXTRA, clean_other
from food_inspections import config as C
from food_inspections.analysis import block_bootstrap_diff
from food_inspections.data import clean, load_raw
from food_inspections.features import features_at_date, load_panel
from food_inspections.local_features import LOCAL_COLUMNS, build_local
from food_inspections.models import make_logistic, make_neural_network

OUTPUT = C.RESULTS_DIR / "confirm_local.json"
NEURAL = {"hidden_layer_sizes": (16, 8), "alpha": 10.0}


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    panel = load_panel(df)
    holdout = features_at_date(df, HOLDOUT_DATE)
    holdout["period"] = "holdout"
    allrows = pd.concat([panel, holdout], ignore_index=True)
    local = build_local(df, allrows)
    allrows = allrows.merge(local, on=["decision_date", "license_"], how="left", validate="one_to_one")
    rows, Xb, Xn = add_features(allrows, clean_other(pd.read_csv(RAW_OTHER, dtype=str, low_memory=False)))
    Xl = pd.concat([Xn, rows[LOCAL_COLUMNS].astype(float)], axis=1)
    y = rows["label"].astype(int)
    train = rows["period"] == "train"

    scores = {"baseline": rows["hist_fail_rate_smooth"]}
    for tag, factory, X in [("logistic", make_logistic, Xb),
                            ("logistic_complaints", make_logistic, Xn),
                            ("logistic_local_complaints", make_logistic, Xl),
                            ("neural", lambda: make_neural_network(NEURAL), Xb),
                            ("neural_local_complaints", lambda: make_neural_network(NEURAL), Xl)]:
        m = factory().fit(X[train], y[train])
        scores[tag] = pd.Series(m.predict_proba(X)[:, 1], index=rows.index)
        if tag == "logistic_local_complaints":
            coef = pd.Series(m.named_steps["clf"].coef_[0], index=X.columns)

    out = {"preregistration": "docs/preregistration_local.md"}
    pairs = [("logistic_local_complaints", "logistic"), ("logistic_local_complaints", "logistic_complaints"),
             ("logistic_local_complaints", "baseline"), ("logistic", "baseline"),
             ("neural_local_complaints", "neural")]
    for period, key in [("test", "period_2022_2025"), ("holdout", "date_2025_09_30")]:
        r = rows[rows["period"] == period]
        prec = {t: per_date_precision(r, s) for t, s in scores.items()}
        block = {"precision_at_100": {t: float(np.mean(list(v.values()))) for t, v in prec.items()},
                 "per_date": prec, "comparisons": {}}
        for a, b in pairs:
            d = np.array([prec[a][k] - prec[b][k] for k in sorted(prec[a])])
            if period == "holdout":
                ci = single_date_bootstrap(r, scores[a].loc[r.index], scores[b].loc[r.index])
            else:
                ci = list(block_bootstrap_diff(d, block=C.LABEL_WINDOW_DAYS // 91))
            block["comparisons"][f"{a} minus {b}"] = {
                "mean_difference": float(d.mean()), "ci95": ci, "dates_ahead": int((d > 0).sum()),
                "dates_tied": int((d == 0).sum()), "dates_behind": int((d < 0).sum())}
        out[key] = block

    # pre-registered description: failure rate by past-failure band x local-rate quintile, 2022-25
    t = rows[rows["period"] == "test"].copy()
    t["past_failure_band"] = pd.qcut(t["hist_fail_rate_smooth"].rank(method="first"), 3,
                                     labels=["low", "middle", "high"])
    t["local_quintile"] = t.groupby("decision_date")["near_rate_180d"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    tab = t.pivot_table(index="past_failure_band", columns="local_quintile", values="label",
                        aggfunc="mean", observed=True)
    out["failure_rate_by_band_and_local_quintile_2022_2025"] = {
        str(i): {str(int(c)): float(v) for c, v in row.items()} for i, row in tab.iterrows()}
    out["primary_model_coefficients_top"] = coef.reindex(coef.abs().sort_values(ascending=False).index)[:15].round(4).to_dict()

    te = out["period_2022_2025"]
    rule = {
        "a_2022_25_interval_vs_logistic_above_zero":
            te["comparisons"]["logistic_local_complaints minus logistic"]["ci95"][0] > 0,
        "b_2022_25_mean_above_logistic_complaints":
            te["comparisons"]["logistic_local_complaints minus logistic_complaints"]["mean_difference"] > 0,
        "c_holdout_not_worse_than_logistic_complaints":
            out["date_2025_09_30"]["comparisons"]["logistic_local_complaints minus logistic_complaints"]["mean_difference"] >= 0,
    }
    rule["confirmed"] = all(rule.values())
    out["decision_rule"] = rule
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key in ["period_2022_2025", "date_2025_09_30"]:
        print(key, {k: round(v, 4) for k, v in out[key]["precision_at_100"].items()})
        for k, v in out[key]["comparisons"].items():
            print("   ", k, round(v["mean_difference"] * 100, 2), [round(x * 100, 2) for x in v["ci95"]],
                  v["dates_ahead"], v["dates_tied"], v["dates_behind"])
    print(pd.DataFrame(out["failure_rate_by_band_and_local_quintile_2022_2025"]).T.round(3))
    print(out["primary_model_coefficients_top"])
    print(rule)


if __name__ == "__main__":
    main()
