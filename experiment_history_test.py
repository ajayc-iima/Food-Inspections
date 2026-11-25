#!/usr/bin/env python3
"""Exploratory comparison on previously examined 2022–25 test dates.

This period has already influenced prior model development, so this is NOT a
fresh holdout. Historical features are reconstructed as of each decision date.
No existing model or results files are overwritten.
"""
import json

import pandas as pd
from threadpoolctl import threadpool_limits

from experiment_history import EXTRA_COLUMNS, history_features
from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import mean_precision_at_k, rank_order
from food_inspections.features import load_panel
from food_inspections.models import design_matrix, make_neural_network


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    panel = load_panel(df)
    panel = panel[panel["period"].isin(["train", "test"])].copy()
    additional = []
    for date in sorted(panel["decision_date"].unique()):
        date = pd.Timestamp(date)
        frame = history_features(df, date)
        frame["decision_date"] = date
        additional.append(frame)
    extra = pd.concat(additional, ignore_index=True)
    panel = panel.merge(extra, on=["decision_date", "license_"], how="left", validate="many_to_one")
    panel[EXTRA_COLUMNS] = panel[EXTRA_COLUMNS].fillna(0)
    original = design_matrix(panel)
    enhanced = pd.concat([original, panel[EXTRA_COLUMNS].astype(float)], axis=1)
    training, test = panel["period"].eq("train"), panel["period"].eq("test")
    target = panel["label"].astype(int)
    params = {"hidden_layer_sizes": (16, 8), "alpha": 10.0}
    scores = {}
    details = {}
    for name, matrix, limit in [("original", original, 60), ("enhanced_60", enhanced, 60),
                                ("enhanced_150", enhanced, 150)]:
        fitted = make_neural_network(params).set_params(clf__max_iter=limit)
        fitted.fit(matrix[training], target[training])
        clf = fitted.named_steps["clf"]
        details[name] = {"epochs": int(clf.n_iter_), "max_epochs": limit,
                         "reached_epoch_limit": bool(clf.n_iter_ >= limit)}
        scores[name] = pd.Series(fitted.predict_proba(matrix[test])[:, 1], index=panel.index[test])
    test_rows = panel[test]
    result = {"status": "complete", "evaluation": "exploratory previously examined 2022–2025 period",
              "fit_details": details, "by_date": {}, "mean_precision_at_100": {}}
    for date, rows in test_rows.groupby("decision_date"):
        daily = {}
        for name, score in scores.items():
            order = rank_order(rows, score.loc[rows.index])
            daily[name] = float(rows["label"].to_numpy()[order[:C.K]].mean())
        result["by_date"][str(date.date())] = daily
    for name, score in scores.items():
        result["mean_precision_at_100"][name] = mean_precision_at_k(test_rows, score)
    path = C.RESULTS_DIR / "experiment_history_test.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"fit_details": details, "mean_precision_at_100": result["mean_precision_at_100"]}, indent=2))


if __name__ == "__main__":
    main()
