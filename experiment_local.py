#!/usr/bin/env python3
"""Stage 1 (validation only): do recent nearby failure rates help?

Uses only 2016-2019 decision dates and the two purged folds. No 2022+ label is
read. Writes results/experiment_local_validation.json.
"""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from confirm_nonroutine import RAW_OTHER
from experiment_nonroutine import EXTRA, clean_other, history_features
from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import mean_precision_at_k
from food_inspections.features import load_panel
from food_inspections.local_features import LOCAL_COLUMNS, build_local
from food_inspections.models import check_purged, design_matrix, make_logistic, make_neural_network

OUTPUT = C.RESULTS_DIR / "experiment_local_validation.json"


def nonroutine(panel, other):
    frames = []
    for T in sorted(panel["decision_date"].unique()):
        f = history_features(other, pd.Timestamp(T))
        f["decision_date"] = pd.Timestamp(T)
        frames.append(f)
    nr = pd.concat(frames, ignore_index=True)
    rows = panel.merge(nr, on=["decision_date", "license_"], how="left", validate="many_to_one")
    for col in EXTRA:
        rows[col] = rows[col].fillna(3650 if col.startswith("nr_days_") else 0.0)
    return rows


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    panel = load_panel(df)
    panel = panel[panel["period"] == "train"].reset_index(drop=True)
    local = build_local(df, panel)
    rows = panel.merge(local, on=["decision_date", "license_"], how="left", validate="one_to_one")
    rows = nonroutine(rows, clean_other(pd.read_csv(RAW_OTHER, dtype=str, low_memory=False)))
    base = design_matrix(rows)
    sets = {
        "original": base,
        "+local": pd.concat([base, rows[LOCAL_COLUMNS]], axis=1),
        "+complaints": pd.concat([base, rows[EXTRA]], axis=1),
        "+local+complaints": pd.concat([base, rows[LOCAL_COLUMNS + EXTRA]], axis=1),
    }
    y = rows["label"].astype(int)
    models = {"logistic": make_logistic,
              "neural": lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0})}
    res = {}
    for name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        assert check_purged(spec["fit"], spec["val"])
        fit, val = rows["decision_date"].isin(spec["fit"]), rows["decision_date"].isin(spec["val"])
        res[name] = {"baseline": mean_precision_at_k(rows[val], rows.loc[val, "hist_fail_rate_smooth"])}
        for mname, factory in models.items():
            for sname, X in sets.items():
                m = factory().fit(X[fit], y[fit])
                s = pd.Series(m.predict_proba(X[val])[:, 1], index=rows.index[val])
                res[name][f"{mname} {sname}"] = mean_precision_at_k(rows[val], s)
                print(f"{name} | {mname} {sname}: {res[name][f'{mname} {sname}']:.4f}", flush=True)
        # single-feature check: rank by the local rate alone
        res[name]["near_rate_180d alone"] = mean_precision_at_k(rows[val], rows.loc[val, "near_rate_180d"])
    mean = {k: float(np.mean([f[k] for f in res.values()])) for k in next(iter(res.values()))}
    corr = rows[LOCAL_COLUMNS + ["label"]].corr()["label"].drop("label").round(4).to_dict()
    OUTPUT.write_text(json.dumps({"description": "Validation-only test of local recent failure rates",
                                  "folds": res, "mean": mean, "correlation_with_label_train": corr}, indent=2))
    print(json.dumps(mean, indent=1))


if __name__ == "__main__":
    main()
