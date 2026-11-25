"""Why the shortlists differ, and how sure we are about the difference."""
import numpy as np
import pandas as pd

from . import config as C
from .evaluation import top_k

PROFILE_FEATURES = {
    "n_canvass_prev": "prior routine inspections",
    "n_fail_prev": "prior failures",
    "hist_fail_rate_smooth": "smoothed past failure rate",
    "has_failed_before": "share that failed before",
    "days_since_last_fail": "days since last failure",
    "viol_last": "violations at last inspection",
    "viol_mean_last3": "mean violations, last 3 inspections",
    "risk_latest": "risk category (1 = high)",
    "days_since_last_canvass": "days since last inspection",
    "label": "share whose next inspection FAILED",
}


def shortlist_profiles(panel: pd.DataFrame, a: str = "baseline", b: str = "logistic"):
    """Profile of establishments picked by only `a`, only `b`, or both.

    Pools the test decision dates. Returns (numeric profile, facility mix).
    """
    groups = []
    for _, rows in panel[panel["period"] == "test"].groupby("decision_date"):
        ta = set(top_k(rows, C.SCORE_COLUMNS[a])["license_"])
        tb = set(top_k(rows, C.SCORE_COLUMNS[b])["license_"])
        for name, ids in [(f"{a} only", ta - tb), ("both", ta & tb), (f"{b} only", tb - ta)]:
            groups.append(rows[rows["license_"].isin(ids)].assign(group=name))
        groups.append(rows.assign(group="all candidates"))
    g = pd.concat(groups, ignore_index=True)
    order = [f"{a} only", "both", f"{b} only", "all candidates"]

    numeric = g.groupby("group")[list(PROFILE_FEATURES)].mean().loc[order].T
    numeric.index = [PROFILE_FEATURES[c] for c in numeric.index]
    numeric.loc["establishment-dates in group"] = g.groupby("group").size().loc[order]

    fac = (g.groupby("group")["facility_grouped"].value_counts(normalize=True)
           .unstack(fill_value=0).loc[order].T)
    fac = fac.loc[fac.max(axis=1).sort_values(ascending=False).index]
    return numeric, fac


def block_bootstrap_diff(diff: np.ndarray, block: int, n_boot: int = 10_000,
                         seed: int = C.SEED) -> tuple:
    """95% interval for the mean of a per-date series with a moving-block bootstrap.

    Consecutive quarterly dates share labelled inspections (the label window is
    four quarters), so dates are resampled in runs of `block` to keep that
    dependence.
    """
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:n]
        means[i] = diff[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test, ties dropped (assumes independent dates)."""
    from math import comb
    n, k = wins + losses, min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def uncertainty(metrics: pd.DataFrame, a: str, b: str = "baseline") -> dict:
    test = metrics[metrics["period"] == "test"].sort_values("decision_date")
    block = C.LABEL_WINDOW_DAYS // 91          # 4 quarters
    out = {"block_length_quarters": block}
    for metric, col in [("precision_at_100", f"prec{C.K}"),
                        ("violations_in_top_100", f"viols_top{C.K}")]:
        d = (test[f"{col}_{a}"] - test[f"{col}_{b}"]).to_numpy(dtype=float)
        lo, hi = block_bootstrap_diff(d, block)
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        out[metric] = {"mean_difference": float(d.mean()), "ci95_block_bootstrap": [lo, hi],
                       "dates_ahead": wins, "dates_tied": int((d == 0).sum()),
                       "dates_behind": losses,
                       "sign_test_p_if_dates_independent": sign_test_p(wins, losses)}
    return out
