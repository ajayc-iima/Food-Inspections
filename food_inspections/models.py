"""Design matrix, the two fitted models, and pre-2020 XGBoost selection."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C
from .evaluation import mean_precision_at_k


def design_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[C.NUMERIC_FEATURES].astype(float).copy()
    fac = pd.get_dummies(panel[C.CATEGORICAL_FEATURE], prefix="fac").astype(float)
    return pd.concat([X, fac], axis=1)


def make_logistic() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=C.SEED)),
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


def select_xgboost(panel: pd.DataFrame, X: pd.DataFrame, verbose: bool = True):
    """Score every XGBoost config (and logistic, for reference) on each scheme.

    Returns (selected config name, table of validation Precision@K). The
    selection criterion is the mean over the purged schemes only.
    """
    y = panel["label"].astype(int)
    rows = {}
    for scheme, spec in C.VALIDATION_SCHEMES.items():
        purged = check_purged(spec["fit"], spec["val"])
        if purged != spec["purged"]:
            raise ValueError(f"scheme '{scheme}' is mislabelled: purged={purged}")
        is_fit = panel["decision_date"].isin(spec["fit"])
        is_val = panel["decision_date"].isin(spec["val"])
        val_panel = panel[is_val]
        if verbose:
            print(f"    {scheme}: {is_fit.sum():,} fit rows, {is_val.sum():,} validation rows")
        models = {"logistic (reference)": make_logistic()}
        models.update({tag: make_xgboost(p) for tag, p in C.XGB_CONFIGS.items()})
        for tag, model in models.items():
            model.fit(X[is_fit], y[is_fit])
            score = pd.Series(model.predict_proba(X[is_val])[:, 1], index=val_panel.index)
            rows.setdefault(tag, {})[scheme] = mean_precision_at_k(val_panel, score)

    table = pd.DataFrame(rows).T[list(C.VALIDATION_SCHEMES)]
    purged_cols = [s for s, spec in C.VALIDATION_SCHEMES.items() if spec["purged"]]
    table["criterion (mean of purged)"] = table[purged_cols].mean(axis=1)
    xgb_rows = table.loc[list(C.XGB_CONFIGS)]
    selected = xgb_rows["criterion (mean of purged)"].idxmax()
    table["selected"] = [tag == selected for tag in table.index]
    return selected, table
