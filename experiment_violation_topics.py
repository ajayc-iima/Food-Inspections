#!/usr/bin/env python3
"""Explore point-in-time violation topics on the original purged folds.

Topics are extracted from violation headings, never the later inspection used
as the label. We intentionally avoid violation code numbers: Chicago changed
the code in 2018. Heading vocabulary still differs between eras, so the
result is an exploratory robustness check, not a stable risk taxonomy.
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


TOPICS = {
    "pest": r"RODENT|INSECT|VERMIN|PEST",
    "temperature": r"TEMPERATURE|THERMOMETER|COLD HOLD|HOT HOLD|COOLING",
    "hand_hygiene": r"HAND\s?WASH|HANDS WASH|HAND SINK",
    "manager": r"CERTIFIED FOOD MANAGER|FOOD SERVICE SANITATION CERTIFICATE|PERSON IN CHARGE",
    "prior_violation": r"PREVIOUS.*VIOLATION",
    "food_contamination": r"CONTAMINATION|FOOD PROTECTED|CROSS.CONTAM|FOOD SOURCE|ADULTERATED|UNSAFE FOOD",
    "sanitation": r"SANITI[ZS]|FOOD CONTACT.*CLEAN|FOOD.CONTACT SURFACES",
    "plumbing": r"PLUMBING|WATER SOURCE|WATER SUPPLY|SEWAGE|BACKFLOW",
}
HEADING = re.compile(r"^\s*#?\s*\d+\s*\.\s*(.*?)(?:\s*-\s*Comments:|$)", re.I)
TOPIC_PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in TOPICS.items()}


def extract_topics(text):
    """Count numbered violation headings matching each topic, excluding comments."""
    counts = dict.fromkeys(TOPICS, 0)
    if not isinstance(text, str):
        return counts
    for segment in text.split("|"):
        match = HEADING.match(segment)
        if match:
            title = match.group(1)
            for name, pattern in TOPIC_PATTERNS.items():
                counts[name] += int(bool(pattern.search(title)))
    return counts


def violation_topic_history(df: pd.DataFrame, decision_date: pd.Timestamp) -> pd.DataFrame:
    """Latest and last-three topic counts using rows dated no later than T."""
    hist = df.loc[df["date"] <= decision_date].sort_values(["license_", "date"], kind="mergesort")
    cols = list(TOPICS)
    last = hist.groupby("license_", sort=False).tail(1).set_index("license_")[cols]
    last.columns = [f"topic_last_{c}" for c in cols]
    recent = hist.groupby("license_", sort=False).tail(3).groupby("license_")[cols].mean()
    recent.columns = [f"topic_last3_mean_{c}" for c in cols]
    return last.join(recent).reset_index()


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    df = clean(load_raw())
    topic_values = pd.DataFrame.from_records(df["violations"].map(extract_topics), index=df.index)
    df = pd.concat([df, topic_values], axis=1)
    panel = load_panel(df)
    panel = panel[panel["period"].eq("train")].copy()
    parts = []
    for date in sorted(panel["decision_date"].unique()):
        date = pd.Timestamp(date)
        part = violation_topic_history(df, date)
        part["decision_date"] = date
        parts.append(part)
    topics = pd.concat(parts, ignore_index=True)
    panel = panel.merge(topics, on=["license_", "decision_date"], how="left", validate="many_to_one")
    features = [f"topic_{kind}_{topic}" for kind in ("last", "last3_mean") for topic in TOPICS]
    panel[features] = panel[features].fillna(0.0)
    base = design_matrix(panel)
    enriched = pd.concat([base, panel[features].astype(float)], axis=1)
    y = panel["label"].astype(int)
    results = {"description": "Exploratory validation of past violation heading topics",
               "features": features, "topic_patterns": TOPICS,
               "limitations": ["Quarterly rows can share labels", "Heading definitions changed in 2018",
                               "These validation folds have been examined during development"],
               "folds": {}, "fit_details": {}}
    candidates = {
        "logistic": lambda: make_logistic(.01),
        "neural": lambda: make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0}),
    }
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
            for feature_name, X in [("original", base), ("topics", enriched)]:
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
    output = C.RESULTS_DIR / "experiment_violation_topics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
