#!/usr/bin/env python3
"""Exploratory, time-matured comparison of frozen and rolling training data.

Each 2023–2025 evaluation quarter is scored by models using only decision
dates with a fully mature 365-day outcome window. Since this project already
examined these dates, results do not constitute an untouched final test.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from food_inspections import config as C
from food_inspections.evaluation import rank_order
from food_inspections.models import design_matrix, make_logistic, make_neural_network, make_xgboost

OUT = C.RESULTS_DIR / "rolling_experiment.json"


def factory(name):
    if name == "logistic":
        return make_logistic()
    if name == "neural":
        return make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0})
    if name == "xgboost":
        return make_xgboost(C.XGB_CONFIGS["d5-400-mcw10"])
    raise ValueError(name)


def precision(rows, scores):
    if len(rows) < C.K:
        raise ValueError("Cannot evaluate a full top-K with fewer than K candidates")
    return float(rows["label"].to_numpy()[rank_order(rows, scores)[:C.K]].mean())


def save(result):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(OUT)


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    start = time.monotonic()
    panel = pd.read_csv(C.PANEL_FILE, parse_dates=["decision_date"], dtype={"license_": str, "zip": str})
    X = design_matrix(panel)
    y = panel["label"].astype(int)
    old = panel["period"].eq("train")
    dates = sorted(d for d in panel.loc[panel["period"].eq("test"), "decision_date"].unique()
                   if d >= pd.Timestamp("2023-03-31"))
    if not dates:
        raise ValueError("No evaluation dates")
    window = pd.Timedelta(days=C.LABEL_WINDOW_DAYS)
    if panel.loc[old, "decision_date"].max() + window > dates[0]:
        raise ValueError("The frozen training labels are not fully mature")

    result = {
        "status": "running", "complete": False,
        "interpretation": "Exploratory, previously examined periods and overlapping outcomes; "
                          "retrospective observed-inspection subset, not a deployment queue.",
        "eligibility_rule": "training decision_date + 365 days <= evaluation decision_date",
        "policies": {
            "historical": "2016–2019 training dates only, fitted once",
            "expanding": "historical dates plus fully mature 2022-onward panel dates",
            "recent": "fully mature 2022-onward panel dates only",
        },
        "dates": {},
    }
    save(result)
    try:
        names = ["logistic", "neural", "xgboost"]
        frozen = {name: factory(name).fit(X[old], y[old]) for name in names}
        for date in dates:
            date = pd.Timestamp(date)
            scored = panel["decision_date"].eq(date)
            recent = panel["period"].eq("test") & (panel["decision_date"] + window <= date)
            expanding = old | recent
            rows = panel.loc[scored]
            record = {
                "n_candidates": int(scored.sum()),
                "base_rate": float(rows["label"].mean()),
                "baseline_precision_at_100": precision(rows, rows["hist_fail_rate_smooth"]),
                "training_rows": {"historical": int(old.sum()), "recent": int(recent.sum()),
                                  "expanding": int(expanding.sum())},
                "latest_recent_training_decision_date": str(panel.loc[recent, "decision_date"].max().date()),
                "precision_at_100": {},
                "neural_training_iterations": {},
            }
            for policy, mask in [("historical", old), ("expanding", expanding), ("recent", recent)]:
                if mask.sum() == 0 or panel.loc[mask, "decision_date"].max() + window > date:
                    raise ValueError(f"Invalid training dates for {date}, {policy}")
                if y[mask].nunique() != 2:
                    raise ValueError(f"Training labels need both classes for {date}, {policy}")
                record["precision_at_100"][policy] = {}
                for name in names:
                    model = frozen[name] if policy == "historical" else factory(name).fit(X[mask], y[mask])
                    if name == "neural":
                        record["neural_training_iterations"][policy] = {
                            "iterations": int(model.named_steps["clf"].n_iter_),
                            "iteration_limit": int(model.named_steps["clf"].max_iter),
                        }
                    score = model.predict_proba(X[scored])[:, 1]
                    if not np.isfinite(score).all():
                        raise ValueError("Non-finite model predictions")
                    record["precision_at_100"][policy][name] = precision(rows, score)
            result["dates"][str(date.date())] = record
            print(f"{date.date()} candidates={len(rows):,} " +
                  " ".join(f"{policy}=" + ",".join(f"{name}:{value:.2f}" for name, value in scores.items())
                           for policy, scores in record["precision_at_100"].items()), flush=True)
            save(result)
        result["mean_precision_at_100"] = {
            policy: {name: float(np.mean([d["precision_at_100"][policy][name]
                                        for d in result["dates"].values()])) for name in names}
            for policy in result["policies"]
        }
        result["mean_baseline_precision_at_100"] = float(np.mean(
            [d["baseline_precision_at_100"] for d in result["dates"].values()]))
        result["status"] = "complete"
        result["complete"] = True
        result["elapsed_seconds"] = time.monotonic() - start
        save(result)
        print(json.dumps(result["mean_precision_at_100"], indent=2), flush=True)
    except BaseException as exc:
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.monotonic() - start
        save(result)
        raise


if __name__ == "__main__":
    main()
