#!/usr/bin/env python3
"""Exploratory purged-fold comparison of additional inspection-history features.

Run: python experiment_history.py

This does not use the 2022–25 evaluation period, change the panel cache, or
alter model selection. Every experimental feature is reconstructed from
inspection rows dated at or before the decision date.
"""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import mean_precision_at_k
from food_inspections.features import load_panel
from food_inspections.models import check_purged, design_matrix, make_logistic, make_neural_network


EXTRA_COLUMNS = [
    "last_outcome_fail", "last_outcome_conditional", "days_since_last_outcome",
    "last3_fails", "last3_conditional", "outcomes_12m", "fails_12m",
    "conditional_12m", "prior_outcome_viol_count", "outcome_viol_delta",
    "days_since_last_conditional", "ever_conditional",
]


def history_features(df: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    """Summarize only outcome-bearing records available at the decision date."""
    hist = df[(df["date"] <= decision_date) & df["is_outcome"]].copy()
    # Keep the input's tie order for same-day records, as in clean().
    hist = hist.sort_values(["license_", "date"], kind="mergesort")
    hist["conditional"] = hist["results"].eq("Pass w/ Conditions").astype(int)
    g = hist.groupby("license_", sort=False)
    last = g.tail(1).set_index("license_")
    last3 = g.tail(3).groupby("license_").agg(
        last3_fails=("is_fail", "sum"), last3_conditional=("conditional", "sum"))
    recent = hist[hist["date"] > decision_date - pd.Timedelta(days=365)]
    recent = recent.groupby("license_").agg(
        outcomes_12m=("is_outcome", "size"), fails_12m=("is_fail", "sum"),
        conditional_12m=("conditional", "sum"))
    latest_conditional = hist[hist["conditional"].eq(1)].groupby("license_")["date"].max()
    hist["prior_viol"] = g["viol_count"].shift()
    latest_viol = hist.groupby("license_", sort=False).tail(1).set_index("license_")
    out = pd.DataFrame(index=last.index)
    out["last_outcome_fail"] = last["is_fail"]
    out["last_outcome_conditional"] = last["conditional"]
    out["days_since_last_outcome"] = (decision_date - last["date"]).dt.days
    out = out.join(last3).join(recent)
    out["prior_outcome_viol_count"] = latest_viol["prior_viol"].fillna(0)
    out["outcome_viol_delta"] = (last["viol_count"] - latest_viol["prior_viol"]).fillna(0)
    out["days_since_last_conditional"] = (
        decision_date - latest_conditional.reindex(out.index)).dt.days.fillna(3650).clip(upper=3650)
    out["ever_conditional"] = out.index.isin(latest_conditional.index).astype(int)
    return out[EXTRA_COLUMNS].fillna(0).reset_index()


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    panel = load_panel(df)
    panel = panel[panel["period"].eq("train")].copy()
    additional = []
    for date in sorted(panel["decision_date"].unique()):
        date = pd.Timestamp(date)
        f = history_features(df, date)
        f["decision_date"] = date
        additional.append(f)
    extra = pd.concat(additional, ignore_index=True)
    panel = panel.merge(extra, on=["decision_date", "license_"], how="left", validate="many_to_one")
    # Rows without an outcome-bearing historical canvass have no experimental history.
    panel[EXTRA_COLUMNS] = panel[EXTRA_COLUMNS].fillna(0)
    X_original = design_matrix(panel)
    X_extra = pd.concat([X_original, panel[EXTRA_COLUMNS].astype(float)], axis=1)
    y = panel["label"].astype(int)
    candidates = {
        "logistic C=0.01": lambda: make_logistic(0.01),
        "neural 16-8 alpha=10": lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0}),
        "neural 16-8 alpha=10 max_iter=150": lambda: make_neural_network(
            {"hidden_layer_sizes": (16, 8), "alpha": 10.0}).set_params(clf__max_iter=150),
    }
    result = {
        "description": "Exploratory historical feature comparison on purged training folds only",
        "features_added": EXTRA_COLUMNS, "context": "Features use only historical outcome-bearing inspections dated <= decision date",
        "folds": {}, "fit_details": {}, "mean_purged_precision_at_100": {},
    }
    for name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        if not check_purged(spec["fit"], spec["val"]):
            raise ValueError(f"Unpurged fold: {name}")
        fit = panel["decision_date"].isin(spec["fit"])
        val = panel["decision_date"].isin(spec["val"])
        val_rows = panel[val]
        row = {}
        fit_details = {}
        for candidate, factory in candidates.items():
            for feature_set, matrix in [("original", X_original), ("history", X_extra)]:
                if candidate.endswith("max_iter=150") and feature_set == "original":
                    continue
                model = factory().fit(matrix[fit], y[fit])
                scores = pd.Series(model.predict_proba(matrix[val])[:, 1], index=val_rows.index)
                key = f"{candidate} + {feature_set}"
                row[key] = mean_precision_at_k(val_rows, scores)
                if candidate.startswith("neural"):
                    estimator = model.named_steps["clf"]
                    fit_details[key] = {
                        "epochs": int(estimator.n_iter_),
                        "max_epochs": int(estimator.max_iter),
                        "reached_epoch_limit": bool(estimator.n_iter_ >= estimator.max_iter),
                        "final_training_loss": float(estimator.loss_),
                    }
                print(f"{name} | {key}: {row[key]:.4f}", flush=True)
        result["folds"][name] = row
        result["fit_details"][name] = fit_details
    for key in next(iter(result["folds"].values())):
        result["mean_purged_precision_at_100"][key] = float(np.mean(
            [fold[key] for fold in result["folds"].values()]))
    output = C.RESULTS_DIR / "experiment_history.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
