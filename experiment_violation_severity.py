#!/usr/bin/env python3
"""Explore fixed past-inspection comment indicators on purged training folds.

Only pre-decision inspection comments are used. Patterns are hand-specified,
not learned from outcomes; no names, identifiers or arbitrary text enter the
model. They are imperfect keyword indicators, not verified severity labels.
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.evaluation import mean_precision_at_k
from food_inspections.features import load_panel
from food_inspections.models import check_purged, design_matrix, make_logistic, make_neural_network


PATTERNS = {
    "rodent_evidence": r"\b(?:MOUSE|MICE|RAT|RODENT)\b.{0,60}\b(?:DROPPINGS|FECES|EXCREMENT)\b|\b(?:DROPPINGS|FECES|EXCREMENT)\b.{0,60}\b(?:MOUSE|MICE|RAT|RODENT)\b",
    "live_pests": r"\bLIVE\b.{0,35}\b(?:ROACH(?:ES)?|FLIES|INSECTS?|RODENTS?|MOUSE|MICE|RATS?|BUGS?)\b",
    "hand_sink": r"\b(?:HAND\s?WASH(?:ING)?|HAND\s?SINK)\b",
    "hot_water_unavailable": r"\b(?:NO|LACK(?:ING)?|WITHOUT)\b.{0,35}\bHOT WATER\b",
    "unsafe_temp": r"\b(?:IMPROPER|INADEQUATE|UNSAFE|INCORRECT)\b.{0,40}\bTEMPERATURE\b|\bTEMPERATURE\b.{0,40}\b(?:IMPROPER|INADEQUATE|UNSAFE|INCORRECT)\b",
    "food_discarded": r"\b(?:FOOD|PRODUCT|MEAT|MILK)\b.{0,60}\b(?:DISCARD(?:ED)?|DESTROY(?:ED)?)\b|\b(?:DISCARD(?:ED)?|DESTROY(?:ED)?)\b.{0,60}\b(?:FOOD|PRODUCT|MEAT|MILK)\b",
    "previous_not_corrected": r"\bPREVIOUS\b.{0,80}\b(?:NOT CORRECTED|UNCORRECTED)\b",
    "serious_priority": r"\b(?:CRITICAL|SERIOUS|PRIORITY)\s+(?:FOUNDATION\s+)?VIOLATION\b",
    "citation_issued": r"\bCITATION\s+(?:WAS\s+)?ISSUED\b",
}
REGEX = {key: re.compile(pattern, re.I) for key, pattern in PATTERNS.items()}
COMMENTS = re.compile(r"\bComments:\s*(.*)$", re.I | re.S)


def comment_indicators(text):
    """Binary indicators based only on past comments following the heading."""
    comments = []
    if isinstance(text, str):
        for segment in text.split("|"):
            match = COMMENTS.search(segment)
            if match:
                comments.append(match.group(1))
    joined = " | ".join(comments)
    return {name: int(bool(pattern.search(joined))) for name, pattern in REGEX.items()}


def comment_history(df, decision_date):
    hist = df.loc[df["date"] <= decision_date].sort_values(["license_", "date"], kind="mergesort")
    cols = list(PATTERNS)
    recent = hist.groupby("license_", sort=False).tail(3).groupby("license_")[cols].mean()
    recent.columns = [f"comment_last3_{col}" for col in cols]
    last = hist.groupby("license_", sort=False).tail(1).set_index("license_")[cols]
    last.columns = [f"comment_last_{col}" for col in cols]
    return last.join(recent).reset_index()


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    df = pd.concat([df, pd.DataFrame.from_records(df["violations"].map(comment_indicators), index=df.index)], axis=1)
    panel = load_panel(df)
    panel = panel[panel["period"].eq("train")].copy()
    parts = []
    for date in sorted(panel["decision_date"].unique()):
        part = comment_history(df, pd.Timestamp(date))
        part["decision_date"] = date
        parts.append(part)
    extra = pd.concat(parts, ignore_index=True)
    panel = panel.merge(extra, on=["license_", "decision_date"], how="left", validate="many_to_one")
    cols = [f"comment_{period}_{name}" for period in ("last", "last3") for name in PATTERNS]
    panel[cols] = panel[cols].fillna(0.0)
    base = design_matrix(panel)
    enhanced = pd.concat([base, panel[cols].astype(float)], axis=1)
    y = panel["label"].astype(int)
    results = {"description": "Exploratory indicators extracted only from past inspector comments",
               "patterns": PATTERNS, "features": cols,
               "limitations": ["Keyword hits do not establish actual severity or account fully for negation",
                               "Comment vocabulary and inspection code changed over time",
                               "Validation folds were previously examined"],
               "folds": {}, "fit_details": {}}
    candidates = {"logistic": lambda: make_logistic(.01),
                  "neural": lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0})}
    for name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        if not check_purged(spec["fit"], spec["val"]):
            raise ValueError(f"Unpurged fold: {name}")
        fit = panel["decision_date"].isin(spec["fit"])
        val = panel["decision_date"].isin(spec["val"])
        val_rows = panel[val]
        scores, details = {}, {}
        for model_name, factory in candidates.items():
            for feature_name, X in [("original", base), ("comments", enhanced)]:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model = factory().fit(X[fit], y[fit])
                pred = pd.Series(model.predict_proba(X[val])[:, 1], index=val_rows.index)
                key = f"{model_name}_{feature_name}"
                scores[key] = mean_precision_at_k(val_rows, pred)
                if model_name == "neural":
                    clf = model.named_steps["clf"]
                    details[key] = {"epochs": int(clf.n_iter_), "max_epochs": int(clf.max_iter),
                                    "convergence_warning": any(issubclass(x.category, ConvergenceWarning)
                                                               for x in caught)}
                print(f"{name} | {key}: {scores[key]:.4f}", flush=True)
        results["folds"][name] = scores
        results["fit_details"][name] = details
    results["mean_purged_precision_at_100"] = {
        key: float(np.mean([fold[key] for fold in results["folds"].values()]))
        for key in next(iter(results["folds"].values()))
    }
    output = C.RESULTS_DIR / "experiment_violation_severity.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
