#!/usr/bin/env python3
"""Compare ranking-oriented models on purged historical folds only.

No 2022+ evaluation rows are read for fitting, selection, or scoring.  The
folds have been examined during development, so scores are exploratory.
"""
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBRanker, XGBRegressor

from food_inspections import config as C
from food_inspections.evaluation import mean_precision_at_k
from food_inspections.models import check_purged, design_matrix, make_neural_network


def percentile(panel, scores):
    """Fractional ranks computed independently within each decision date."""
    s = pd.Series(np.asarray(scores), index=panel.index)
    return s.groupby(panel["decision_date"]).rank(method="average", pct=True)


@threadpool_limits.wrap(limits=1, user_api="blas")
def main():
    start = time.monotonic()
    panel = pd.read_csv(C.PANEL_FILE, parse_dates=["decision_date"],
                        dtype={"license_": str, "zip": str})
    panel = panel[panel["period"].eq("train")].copy()
    X = design_matrix(panel)
    y = panel["label"].astype(int)
    output = {"description": "Exploratory ranking and auxiliary target experiments on two purged historical folds; no later evaluation dates used.",
              "folds": {}, "mean_purged_precision_at_100": {}}
    for fold_name, spec in C.VALIDATION_SCHEMES.items():
        if not spec["purged"]:
            continue
        if not check_purged(spec["fit"], spec["val"]):
            raise ValueError("The label windows overlap the validation dates")
        fit = panel[panel["decision_date"].isin(spec["fit"])].sort_values("decision_date", kind="mergesort")
        val = panel[panel["decision_date"].isin(spec["val"])].copy()
        if fit.empty or val.empty:
            raise ValueError("Empty fitting or validation fold")
        qid = pd.factorize(fit["decision_date"], sort=True)[0]
        assert np.all(qid[:-1] <= qid[1:])
        predictions = {}
        neural = make_neural_network({"hidden_layer_sizes": (16, 8), "alpha": 10.0})
        neural.fit(X.loc[fit.index], y.loc[fit.index])
        predictions["neural"] = neural.predict_proba(X.loc[val.index])[:, 1]
        predictions["history_baseline"] = val["hist_fail_rate_smooth"].to_numpy()
        for depth in [2, 4]:
            model = XGBRanker(objective="rank:ndcg", tree_method="hist", n_jobs=4,
                              random_state=C.SEED, max_depth=depth, n_estimators=300,
                              learning_rate=0.05, min_child_weight=0.01,
                              reg_lambda=0.1, subsample=0.8, colsample_bytree=0.8,
                              lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=100)
            model.fit(X.loc[fit.index], y.loc[fit.index], qid=qid)
            predictions[f"ranker_depth_{depth}"] = model.predict(X.loc[val.index])
            if np.ptp(predictions[f"ranker_depth_{depth}"]) < 1e-8:
                raise ValueError(f"Ranker depth {depth} returned constant scores")
        # The auxiliary label is available in the fit fold only.  Its model
        # predicts violation counts, not failure, and is evaluated solely on
        # failure precision after blending with neural ranks.
        if fit["label_viol_count"].isna().any():
            raise ValueError("Missing violation-count target in fitting rows")
        reg = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=4,
                           random_state=C.SEED, max_depth=3, n_estimators=250,
                           learning_rate=0.05, min_child_weight=50,
                           reg_lambda=3, subsample=0.8, colsample_bytree=0.8)
        reg.fit(X.loc[fit.index], np.log1p(fit["label_viol_count"].to_numpy()))
        predictions["violations_regressor"] = reg.predict(X.loc[val.index])
        nn_rank = percentile(val, predictions["neural"])
        for other in ["history_baseline", "ranker_depth_2", "ranker_depth_4", "violations_regressor"]:
            other_rank = percentile(val, predictions[other])
            for weight in [0.25, 0.5, 0.75]:
                predictions[f"neural_{1-weight:.2f}+{other}_{weight:.2f}"] = (
                    (1 - weight) * nn_rank + weight * other_rank)
        scores = {name: mean_precision_at_k(val, pd.Series(score, index=val.index))
                  for name, score in predictions.items()}
        output["folds"][fold_name] = {"fit_rows": len(fit), "validation_rows": len(val),
                                      "neural_epochs": int(neural.named_steps["clf"].n_iter_),
                                      "precision_at_100": scores}
        print(f"{fold_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()), flush=True)
    if len(output["folds"]) != 2:
        raise ValueError("Expected two purged folds")
    output["mean_purged_precision_at_100"] = {name: float(np.mean([
        fold["precision_at_100"][name] for fold in output["folds"].values()]))
        for name in next(iter(output["folds"].values()))["precision_at_100"]}
    output["elapsed_seconds"] = time.monotonic() - start
    destination = C.RESULTS_DIR / "experiment_ranking.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
