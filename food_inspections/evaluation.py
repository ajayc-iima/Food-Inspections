"""Ranking and Precision@K evaluation per decision date."""
import numpy as np
import pandas as pd

from . import config as C


def rank_order(rows: pd.DataFrame, score) -> np.ndarray:
    """Positions of `rows` from highest to lowest score.

    Ties (common for the baseline score) are broken by license number,
    ascending, so every ranking in the project is deterministic and identical
    whichever code path produces it.
    """
    score = np.asarray(score, dtype=float)
    lic = pd.to_numeric(rows["license_"], errors="coerce").fillna(np.inf).to_numpy()
    return np.lexsort((lic, -score))


def top_k(rows: pd.DataFrame, score_col: str, k: int = C.K) -> pd.DataFrame:
    return rows.iloc[rank_order(rows, rows[score_col])[:k]]


def mean_precision_at_k(panel: pd.DataFrame, score: pd.Series, k: int = C.K) -> float:
    """Mean over decision dates of the share of failures in the top k."""
    vals = []
    for _, rows in panel.groupby("decision_date"):
        order = rank_order(rows, score.loc[rows.index])
        vals.append(rows["label"].to_numpy()[order[:k]].mean())
    return float(np.mean(vals))


def evaluate_date(rows: pd.DataFrame) -> dict:
    out = {"n_candidates": len(rows), "base_rate": rows["label"].mean(),
           "failures_total": int(rows["label"].sum())}
    top_sets = {}
    for name in C.APPROACHES:
        ranked = rows.iloc[rank_order(rows, rows[C.SCORE_COLUMNS[name]])]
        for k in C.K_GRID:
            if len(ranked) >= k:
                out[f"prec{k}_{name}"] = ranked["label"].head(k).mean()
        top_sets[name] = set(ranked["license_"].head(C.K))
        out[f"fails_top{C.K}_{name}"] = int(ranked["label"].head(C.K).sum())
    for a, b in [("baseline", "logistic"), ("baseline", "xgboost"), ("logistic", "xgboost")]:
        out[f"overlap_{a}_{b}"] = len(top_sets[a] & top_sets[b])
    return out


def run_evaluation(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    for T, rows in panel.groupby("decision_date"):
        rec = evaluate_date(rows)
        rec["decision_date"] = T
        rec["period"] = rows["period"].iloc[0]
        records.append(rec)
    return pd.DataFrame(records)


def shortlists(panel: pd.DataFrame) -> pd.DataFrame:
    """Top-K shortlist of every approach for every test decision date."""
    parts = []
    for T, rows in panel[panel["period"] == "test"].groupby("decision_date"):
        for name in C.APPROACHES:
            top = top_k(rows, C.SCORE_COLUMNS[name]).copy()
            top["approach"] = name
            top["rank"] = np.arange(1, len(top) + 1)
            parts.append(top)
    return pd.concat(parts, ignore_index=True)


def summarise(df, panel, metrics, selected_xgb) -> dict:
    test_m = metrics[metrics["period"] == "test"]
    train_m = metrics[metrics["period"] == "train"]
    train_p = panel[panel["period"] == "train"]

    def paired(a, b, m):
        d = m[f"prec{C.K}_{a}"] - m[f"prec{C.K}_{b}"]
        return {"mean_difference": float(d.mean()), "dates_ahead": int((d > 0).sum()),
                "dates_tied": int((d == 0).sum()), "dates_behind": int((d < 0).sum())}

    return {
        "dataset": {
            "source": C.DATASET_PAGE,
            "rows_routine_inspections": int(len(df)),
            "establishments": int(df["license_"].nunique()),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "panel": {
            "train_examples": int(len(train_p)),
            "test_examples": int((panel["period"] == "test").sum()),
            "train_decision_dates": [str(d.date()) for d in C.TRAIN_DATES],
            "test_decision_dates": [str(d.date()) for d in C.TEST_DATES],
            "label_window_days": C.LABEL_WINDOW_DAYS,
            "K": C.K,
            "train_labels_from_2020_inspections": int((train_p["label_date"] >= "2020-01-01").sum()),
            "mean_rows_per_labelled_inspection": float(
                panel.groupby(["license_", "label_date"]).size().mean()),
        },
        "xgboost_selected_config": {"name": selected_xgb, "params": C.XGB_CONFIGS[selected_xgb]},
        "test_period": {
            "mean_base_rate": float(test_m["base_rate"].mean()),
            "precision_at_k": {
                str(k): {a: float(test_m[f"prec{k}_{a}"].mean()) for a in C.APPROACHES}
                for k in C.K_GRID},
            "failures_found_per_date": {
                a: float(test_m[f"fails_top{C.K}_{a}"].mean()) for a in C.APPROACHES},
            "logistic_vs_baseline_at_100": paired("logistic", "baseline", test_m),
            "xgboost_vs_baseline_at_100": paired("xgboost", "baseline", test_m),
            "mean_overlap": {
                p: float(test_m[f"overlap_{p}"].mean())
                for p in ["baseline_logistic", "baseline_xgboost", "logistic_xgboost"]},
            "chance_overlap": float((C.K * C.K / test_m["n_candidates"]).mean()),
            "total_failures_per_date": float(test_m["failures_total"].mean()),
        },
        "train_period_in_sample": {
            "mean_base_rate": float(train_m["base_rate"].mean()),
            "precision_at_100": {a: float(train_m[f"prec{C.K}_{a}"].mean())
                                 for a in C.APPROACHES},
        },
    }
