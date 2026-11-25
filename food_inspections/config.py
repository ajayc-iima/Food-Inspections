"""Project-wide constants: paths, decision dates, model settings."""
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths (everything is relative to the project root)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"

RAW_FILE = DATA_DIR / "raw_canvass_inspections.csv.gz"
PANEL_FILE = DATA_DIR / "panel_examples.csv.gz"

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
DATASET_URL = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"
DATASET_PAGE = "https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5"
PAGE_SIZE = 50_000
SELECT_COLS = ("inspection_id,license_,dba_name,facility_type,risk,inspection_date,"
               "inspection_type,results,zip,latitude,longitude,violations")

# --------------------------------------------------------------------------
# Decision problem
# --------------------------------------------------------------------------
SEED = 42
K = 100                       # shortlist size
K_GRID = [25, 50, 100, 150, 200, 300]
LABEL_WINDOW_DAYS = 365       # follow-up window for the next routine inspection
HISTORY_WINDOW_DAYS = 730     # "recent" window for 24-month features

# Decision dates (quarter ends). 2020-2021 (COVID disruption) excluded.
TRAIN_DATES = pd.date_range("2016-03-31", "2019-12-31", freq="QE")
TEST_DATES = pd.date_range("2022-03-31", "2025-06-30", freq="QE")
DECISION_DATES = TRAIN_DATES.union(TEST_DATES)

# Shrinkage for smoothed rates (pseudo-counts added toward the global rate).
SMOOTH_EST = 6     # establishment historical failure rate
SMOOTH_ZIP = 20    # zip-level historical failure rate

TOP_FACILITIES = ["Restaurant", "Grocery Store", "Bakery", "Banquet",
                  "Daycare (2 Under 2 Years)", "Daycare Above 2 Years",
                  "School", "Mobile Food Dispenser", "Long Term Care",
                  "Hospital", "Shared Kitchen"]

NUMERIC_FEATURES = [
    "n_canvass_prev", "n_outcome_prev", "n_fail_prev",
    "hist_fail_rate_smooth", "days_since_last_canvass", "days_since_first_canvass",
    "days_since_last_fail", "has_failed_before",
    "n_canvass_24m", "n_fail_24m", "fail_rate_24m",
    "viol_last", "viol_mean_last3", "viol_mean_24m",
    "zip_hist_fail_rate", "risk_latest",
]
CATEGORICAL_FEATURE = "facility_grouped"

# --------------------------------------------------------------------------
# Approaches
# --------------------------------------------------------------------------
APPROACHES = ["baseline", "logistic", "xgboost", "neural", "extra_trees", "selected"]
SCORE_COLUMNS = {"baseline": "hist_fail_rate_smooth",
                 "logistic": "score_logistic",
                 "xgboost": "score_xgboost", "neural": "score_neural",
                 "extra_trees": "score_extra_trees", "selected": "score_selected"}
COLORS = {"baseline": "#7f7f7f", "logistic": "#1f77b4", "xgboost": "#d62728",
          "neural": "#9467bd", "extra_trees": "#ff7f0e", "selected": "#2ca02c"}
APPROACH_LABELS = {"baseline": "Baseline: historical failure rate",
                   "logistic": "Logistic regression",
                   "xgboost": "XGBoost", "neural": "Neural network",
                   "extra_trees": "Extra Trees",
                   "selected": "Validation-selected model"}

# Candidate XGBoost configurations for pre-2020 model selection.
XGB_CONFIGS = {
    "d6-500-mcw5": dict(max_depth=6, n_estimators=500, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.8,
                        min_child_weight=5, reg_lambda=1.0),
    "d5-400-mcw10": dict(max_depth=5, n_estimators=400, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=1.0),
    "d3-300-mcw50": dict(max_depth=3, n_estimators=300, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=50, reg_lambda=2.0),
    "d4-300-mcw100": dict(max_depth=4, n_estimators=300, learning_rate=0.05,
                          subsample=0.7, colsample_bytree=0.7,
                          min_child_weight=100, reg_lambda=3.0),
    "d3-150-mcw200-gamma2": dict(max_depth=3, n_estimators=150, learning_rate=0.1,
                                 subsample=0.7, colsample_bytree=0.6,
                                 min_child_weight=200, reg_lambda=5.0, gamma=2.0),
    "d2-200-mcw100": dict(max_depth=2, n_estimators=200, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.7,
                          min_child_weight=100, reg_lambda=3.0),
}


def _q(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="QE")


# Validation schemes, all inside the training period (no test data).
# A scheme is "purged" when every fit date T_f satisfies
#     T_f + LABEL_WINDOW_DAYS <= first validation date,
# so no inspection can be the label of both a fit row and a validation row.
# Only purged schemes enter the selection criterion. The two unpurged
# schemes are reported for reference: their fit and validation rows share
# labels, which rewards configurations that memorise those labels.
VALIDATION_SCHEMES = {
    "P1: fit 2016-17 -> val 2019": dict(fit=_q("2016-03-31", "2017-12-31"),
                                       val=_q("2019-03-31", "2019-12-31"),
                                       purged=True),
    "P2: fit 2016 -> val 2018": dict(fit=_q("2016-03-31", "2016-12-31"),
                                    val=_q("2018-03-31", "2018-12-31"),
                                    purged=True),
    "U1: fit 2016-18 -> val 2019 (leaky)": dict(fit=_q("2016-03-31", "2018-12-31"),
                                               val=_q("2019-03-31", "2019-12-31"),
                                               purged=False),
    "U2: fit 2016-17 -> val 2018 (leaky)": dict(fit=_q("2016-03-31", "2017-12-31"),
                                               val=_q("2018-03-31", "2018-12-31"),
                                               purged=False),
}
