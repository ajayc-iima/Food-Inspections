"""Design matrices and model selection using time-purged validation."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from . import config as C
from .evaluation import mean_precision_at_k
from .extra_models import EXTRA_TREES_CONFIGS, make_extra_trees


def design_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[C.NUMERIC_FEATURES].astype(float).copy()
    # Categories are set by project configuration, not inferred from test rows.
    categories = [*C.TOP_FACILITIES, "Other"]
    facility = panel[C.CATEGORICAL_FEATURE].where(
        panel[C.CATEGORICAL_FEATURE].isin(categories), "Other")
    facility = pd.Series(pd.Categorical(facility, categories=categories), index=panel.index)
    fac = pd.get_dummies(facility, prefix="fac").astype(float)
    return pd.concat([X, fac], axis=1)


def make_logistic(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=2000, random_state=C.SEED)),
    ])


SPLINE_FEATURES = ["viol_mean_last3", "hist_fail_rate_smooth", "days_since_last_canvass"]
SPLINE_PARAMS = {"degree": 2, "n_knots": 4, "knots": "uniform",
                 "extrapolation": "constant", "include_bias": False}


def make_spline_logistic(c: float) -> Pipeline:
    """Fit a small additive nonlinear basis using only the supplied training rows."""
    return Pipeline([
        ("features", ColumnTransformer(
            [("splines", SplineTransformer(**SPLINE_PARAMS), SPLINE_FEATURES)],
            remainder="passthrough")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=2000, random_state=C.SEED)),
    ])


def make_neural_network(params: dict) -> Pipeline:
    """Train scaling and a compact multilayer perceptron on the fitting fold.

    There is deliberately no random early-stopping split: repeated labels in
    neighbouring quarterly observations would leak into such a split. The
    iteration limit and training-loss convergence rule are fixed in advance.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(random_state=C.SEED, solver="adam", batch_size=1024,
                              max_iter=60, n_iter_no_change=5, early_stopping=False,
                              **params)),
    ])


def make_xgboost(params: dict) -> XGBClassifier:
    return XGBClassifier(random_state=C.SEED, n_jobs=4, tree_method="hist",
                         eval_metric="logloss", **params)


def logistic_coefficients(lr: Pipeline, feature_names: list) -> pd.Series:
    """Coefficients on standardised features (log-odds per 1 SD)."""
    coef = lr.named_steps["clf"].coef_[0]
    return pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)


# --------------------------------------------------------------------------
# Model selection (training period only)
# --------------------------------------------------------------------------
def check_purged(fit_dates, val_dates) -> bool:
    """True when no fit-date label window reaches the first validation date."""
    return max(fit_dates) + pd.Timedelta(days=C.LABEL_WINDOW_DAYS) <= min(val_dates)


# Candidate comparison uses only training-period validation dates. A smaller C
# means stronger regularization; retain the default reference for comparison.
LOGISTIC_CONFIGS = {
    "logistic C=0.01": 0.01,
    "logistic C=0.1": 0.1,
    "logistic (reference)": 1.0,
    "logistic C=10": 10.0,
}
SPLINE_CONFIGS = {"spline logistic C=0.01": 0.01, "spline logistic C=0.1": 0.1}
NEURAL_CONFIGS = {
    "neural 16-8 alpha=10": {"hidden_layer_sizes": (16, 8), "alpha": 10.0},
}
BASELINE_NAME = "baseline"


def make_candidate(name: str):
    """Create an unfitted candidate; the baseline needs no fitted model."""
    if name == BASELINE_NAME:
        return None
    if name in LOGISTIC_CONFIGS:
        return make_logistic(LOGISTIC_CONFIGS[name])
    if name in SPLINE_CONFIGS:
        return make_spline_logistic(SPLINE_CONFIGS[name])
    if name in NEURAL_CONFIGS:
        return make_neural_network(NEURAL_CONFIGS[name])
    if name in EXTRA_TREES_CONFIGS:
        return make_extra_trees(EXTRA_TREES_CONFIGS[name])
    if name in C.XGB_CONFIGS:
        return make_xgboost(C.XGB_CONFIGS[name])
    raise ValueError(f"Unknown model candidate: {name}")


@threadpool_limits.wrap(limits=1, user_api="blas")
def select_model(panel: pd.DataFrame, X: pd.DataFrame, verbose: bool = True):
    """Select across baseline, logistic, neural and XGBoost using purged folds only.

    Returns (overall winner, best XGBoost, validation table). Unpurged folds
    are diagnostic only and never contribute to the selection criterion.
    """
    y = panel["label"].astype(int)
    rows = {}
    iterations = {}
    for scheme, spec in C.VALIDATION_SCHEMES.items():
        purged = check_purged(spec["fit"], spec["val"])
        if purged != spec["purged"]:
            raise ValueError(f"scheme '{scheme}' is mislabelled: purged={purged}")
        is_fit = panel["decision_date"].isin(spec["fit"])
        is_val = panel["decision_date"].isin(spec["val"])
        val_panel = panel[is_val]
        if verbose:
            print(f"    {scheme}: {is_fit.sum():,} fit rows, {is_val.sum():,} validation rows")
        candidates = [BASELINE_NAME, *LOGISTIC_CONFIGS, *SPLINE_CONFIGS,
                      *NEURAL_CONFIGS, *EXTRA_TREES_CONFIGS, *C.XGB_CONFIGS]
        for tag in candidates:
            # Expensive new candidates do not need leakage-diagnostic folds,
            # which cannot affect selection.
            if not purged and (tag in NEURAL_CONFIGS or tag in EXTRA_TREES_CONFIGS):
                continue
            if verbose:
                print(f"      {tag}", flush=True)
            model = make_candidate(tag)
            if model is None:
                score = val_panel["hist_fail_rate_smooth"]
            else:
                model.fit(X[is_fit], y[is_fit])
                if tag in NEURAL_CONFIGS:
                    iterations.setdefault(tag, {})[scheme] = model.named_steps["clf"].n_iter_
                score = pd.Series(model.predict_proba(X[is_val])[:, 1], index=val_panel.index)
            rows.setdefault(tag, {})[scheme] = mean_precision_at_k(val_panel, score)

    table = pd.DataFrame(rows).T[list(C.VALIDATION_SCHEMES)]
    purged_cols = [s for s, spec in C.VALIDATION_SCHEMES.items() if spec["purged"]]
    table["criterion (mean of purged)"] = table[purged_cols].mean(axis=1)
    xgb_rows = table.loc[list(C.XGB_CONFIGS)]
    selected_xgb = xgb_rows["criterion (mean of purged)"].idxmax()
    selected = table["criterion (mean of purged)"].idxmax()
    table["selected"] = table.index == selected
    table["selected_xgboost"] = table.index == selected_xgb
    for scheme in C.VALIDATION_SCHEMES:
        table[f"{scheme} neural epochs"] = pd.Series(
            {tag: counts[scheme] for tag, counts in iterations.items() if scheme in counts}, dtype=float)
    return selected, selected_xgb, table


def select_xgboost(panel: pd.DataFrame, X: pd.DataFrame, verbose: bool = True):
    """Compatibility wrapper returning the best XGBoost and its table."""
    _, selected_xgb, table = select_model(panel, X, verbose=verbose)
    table["selected"] = table["selected_xgboost"]
    return selected_xgb, table
