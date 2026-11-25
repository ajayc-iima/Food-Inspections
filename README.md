# Which restaurants should be inspected first?

**Prospect-selection project — Machine Learning for Business Analytics (Plaksha MS-AI, Monsoon 2026)**
Setting: municipal food-safety inspection prioritisation on public data from the City of Chicago.

A municipal food safety department can move only **K = 100 establishments** to the front of its
routine-inspection queue. This project builds and evaluates data-driven ways of choosing those 100 so
that the **number of food-safety failures found** is maximised. Everything is reproducible with one
script on public data; no manual steps, no invented outcomes.

---

## 1. The decision problem

| Component | Formulation |
|---|---|
| Organisation (unnamed) | A city food-safety/health department |
| Opportunity | A licensed food establishment in Chicago |
| Limit | K = 100 establishments prioritised per decision date |
| Value of a good decision | More genuine food-safety failures found per 100 inspections; earlier detection of chronic violators |
| Prediction target | Whether the establishment's **next routine ("Canvass") inspection fails** |
| Evaluation | **Precision@100** = share of the shortlisted 100 whose next routine inspection actually failed (= failures found in the top 100) |
| Baseline | Prioritise by smoothed historical failure rate |
| ML approaches | Logistic regression (transparent) and XGBoost (flexible) |
| Validation | Features use only information dated on or before each decision date; models and model selection use only 2016–2019; test on 2022–2025 |

The outcome is the real inspection result recorded by the city. The gap between "next inspection
fails" and "restaurant is currently unsafe" is discussed in §7.

## 2. Data and source

| Item | Detail |
|---|---|
| Dataset | **Food Inspections**, City of Chicago Data Portal, dataset id `4ijn-s7e5` |
| Page | https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5 |
| API | Socrata Open Data API: `https://data.cityofchicago.org/resource/4ijn-s7e5.csv` (public, no key; Chicago Data Portal terms of use apply) |
| Extract used | All rows with `inspection_type = 'Canvass'` (routine inspections), 2010-01-04 → 2026-11-20 |
| Snapshot | `data/raw_canvass_inspections.csv.gz` — 164,361 rows, downloaded **2026-11-24** |
| Reproducibility | The script reuses the snapshot by default; `--refresh` re-downloads the live dataset (headline numbers correspond to the 2026-11-24 snapshot) |

Scope: **routine canvass inspections only**. Complaint, license and task-force inspections are
triggered by events rather than by a scheduling decision, so the "which establishment next" decision
lives in the canvass queue. Re-inspections are excluded for the same reason.

Cleaning (`food_inspections/data.py`): rows with missing/zero license numbers are dropped; duplicate
(license, date, type, result) rows are collapsed; `Fail` / `Pass` / `Pass w/ Conditions` are
outcome-bearing, `Out of Business` / `Not Ready` / `No Entry` are not; risk is parsed to 1/2/3;
facility types are normalised to 11 groups + "Other"; the number of cited violations is parsed from the
pipe-separated `violations` text. Result: **163,316 routine inspections across 36,991 establishments**.

## 3. Formulation: point-in-time examples

Each observation is an **(establishment, decision date)** pair at a quarter-end date T. Features use
only inspections dated ≤ T; the label is the result of the first outcome-bearing canvass in
(T, T + 365 days]. An example exists only if the establishment has ≥ 1 prior inspection **and** its
next routine outcome is observed within the window (see §7).

| Feature | Meaning |
|---|---|
| `hist_fail_rate_smooth` | Historical failure rate over prior outcome-bearing canvasses, shrunk toward the city-wide rate as of T with 6 pseudo-observations (the **baseline score**) |
| `n_canvass_prev`, `n_outcome_prev`, `n_fail_prev`, `has_failed_before` | Inspection and failure history |
| `days_since_last_canvass`, `days_since_first_canvass`, `days_since_last_fail` | Recency: last visit, establishment age, last failure (capped at 10 years) |
| `n_canvass_24m`, `n_fail_24m`, `fail_rate_24m` | Activity and failures in the last 24 months |
| `viol_last`, `viol_mean_last3`, `viol_mean_24m` | Violations cited on the last canvass, mean of the last three, mean over 24 months |
| `risk_latest` | Latest known risk category (1 high / 2 medium / 3 low) |
| `facility_grouped` (one-hot) | Restaurant, grocery store, bakery, banquet, daycares, school, hospital, long-term care, … |
| `zip_hist_fail_rate` | Historical canvass failure rate of the establishment's ZIP as of T (shrunk toward the city-wide rate with 20 pseudo-observations) |

Panel: **30 decision dates** (16 train: 2016Q1–2019Q4; 14 test: 2022Q1–2025Q2), 100,130 train and
87,797 test examples. 2020–2021 decision dates are excluded (COVID disruption). Two caveats:

- Decision dates in 2019 have label windows that reach into 2020, so **12,530 training labels (12.5%)
  come from inspections in 2020**. 2020–2021 inspections also feed the history features of 2022+ dates.
- Decision dates are three months apart but the label window is twelve months, so **one inspection
  labels about 3 consecutive rows** of the same establishment. This matters for validation (§6).

## 4. The approaches

1. **Baseline — smoothed historical failure rate.** Rank by `hist_fail_rate_smooth`: inspect the
   places that failed most often before. Shrinkage stops one-off failures in low-volume
   establishments from dominating. Ties (common for this score) are broken by license number, as for
   every ranking in the project.
2. **Logistic regression.** Standardised numeric features + facility one-hots, fitted on all training
   examples (`sklearn` defaults, no tuning). Coefficients are in `results/logistic_coefficients.csv`
   and `charts/fig7_logistic_coefficients.png`.
3. **XGBoost.** Gradient-boosted trees. The configuration is **chosen automatically on training dates
   only** from six candidates (§6). On this snapshot it selects `max_depth=6`, 500 trees, learning
   rate 0.05, `min_child_weight=5`, subsample 0.9, column subsample 0.8.

All three score the same candidate set at every decision date, so the comparison is exact.

## 5. Evaluation design

- **Ranking task per decision date:** score all candidates, take the top 100, measure the share whose
  next canvass failed. Also reported for K = 25, 50, 150, 200, 300.
- **Reference:** the mean failure rate of all candidates at that date (picking 100 at random).
- **Out-of-sample rule:** models and model selection see only 2016–2019 decision dates; 2022–2025 is
  untouched until the final evaluation.

## 6. Results (test period 2022–2025, out-of-sample)

| Approach | Precision@100 (mean over 14 test dates) | × random (24.7%) | vs baseline, per date |
|---|---|---|---|
| Random selection (reference) | 24.7% | 1.0 | |
| **Baseline: historical failure rate** | **44.1%** | 1.8 | |
| **Logistic regression** | **47.8%** | 1.9 | ahead 10, tied 2, behind 2 |
| XGBoost (selected pre-2020) | 43.8% | 1.8 | ahead 6, tied 2, behind 6 |

In-sample (train dates) for contrast: baseline 45.3%, logistic 50.2%, XGBoost **97.1%**.

Precision@K over the test period (`charts/fig3_precision_at_k.png`):

| K | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| Baseline | **53.7%** | **49.1%** | 44.1% | 41.0% | 39.2% | 39.2% |
| Logistic | 49.4% | 47.3% | **47.8%** | **45.8%** | **44.2%** | **43.7%** |
| XGBoost | 47.7% | 45.7% | 43.8% | 41.8% | 41.9% | 41.1% |

**Findings**

- *A simple rule gets far.* The historical-failure-rate baseline roughly **doubles** the failure yield
  of random selection.
- *The transparent model adds about 3.7 points at K = 100* (47.8% vs 44.1%), ahead on 10 of 14 dates
  (`charts/fig2_precision_at_100.png`), and its lead grows for larger shortlists. **For very short
  lists (K ≤ 50) the baseline is better**: the establishments with the worst track record are the
  surest bets, and the model trades some of them for more moderate-risk places.
- *Why XGBoost reaches 97% in-sample.* Because one inspection labels ~3 consecutive rows of the same
  establishment, a deep tree ensemble can memorise labels it has seen in neighbouring rows. That
  inflates in-sample precision, and it also inflates any validation scheme whose fit and validation
  dates are less than 12 months apart.
- *Model selection must be purged.* `model_selection.py` / `results/model_selection.csv` scores every
  configuration on two **purged** schemes (fit 2016–17 → validate 2019; fit 2016 → validate 2018),
  where the last fit date is at least 12 months before the first validation date, so no inspection
  labels both a fit row and a validation row. Two unpurged schemes are shown for reference only:

  | Model | P1 (purged) | P2 (purged) | U1 fit 16–18 → 19 (leaky) | U2 fit 16–17 → 18 (leaky) | Criterion: mean of purged |
  |---|---|---|---|---|---|
  | Logistic regression (reference) | **0.502** | **0.460** | 0.498 | 0.467 | **0.481** |
  | XGB d6-500-mcw5 (selected) | 0.467 | 0.440 | **0.568** | **0.518** | 0.454 |
  | XGB d3-300-mcw50 | 0.472 | 0.428 | 0.488 | 0.473 | 0.450 |
  | XGB d5-400-mcw10 | 0.430 | 0.455 | 0.532 | 0.480 | 0.442 |
  | XGB d2-200-mcw100 | 0.437 | 0.432 | 0.478 | 0.378 | 0.435 |
  | XGB d4-300-mcw100 | 0.438 | 0.422 | 0.520 | 0.400 | 0.430 |
  | XGB d3-150-mcw200-gamma2 | 0.415 | 0.365 | 0.488 | 0.390 | 0.390 |

  The leaky schemes make the deepest XGBoost look clearly best (0.568 vs 0.498 for logistic). Purged,
  **logistic regression beats every XGBoost configuration**, and the XGBoost configurations are
  within a few points of each other. A pre-2020 protocol that compares model families on purged
  validation therefore picks logistic regression, which is also the test-period winner. We keep
  XGBoost in the comparison as the flexible alternative, not because it would be deployed.
- *The shortlists differ sharply.* Baseline and logistic top-100s share on average **27.5** of 100
  establishments, baseline and XGBoost **23.6**, logistic and XGBoost **49.4** (chance overlap: 1.6)
  — `charts/fig4_shortlist_overlap.png`, map in `charts/fig5_shortlist_map.png`. The models pull in
  many establishments the baseline never shortlists, largely through facility type (schools,
  long-term care) and recent violation counts (`fig6_feature_importance.png`,
  `fig7_logistic_coefficients.png`).

## 7. What this establishes — and what it cannot

**Established.** On data the decision-maker could have known at each decision date, ranking by
predicted failure probability finds about twice as many real failures in the next 100 routine
inspections as random selection, and the transparent model beats the natural simple baseline at
K = 100 on 10 of 14 out-of-sample dates over three post-pandemic years.

**Cannot be established (limitations).**

- **Uninspected restaurants are invisible.** Evaluation covers only establishments the city actually
  canvassed within the follow-up window. A deployed system would score everyone, but for uninspected
  places the outcome never exists. Because canvass scheduling correlates with risk, precision on the
  inspected subset may overstate what a re-prioritised queue would find.
- **"Failed inspection" ≠ "unsafe restaurant".** Results depend on inspector strictness and
  establishment type; `Pass w/ Conditions` (counted as a pass) is a middle state.
- **Regime change.** The 2018 framework change and the 2020–21 disruption both shifted behaviour;
  any real deployment needs rolling, purged revalidation.
- **Identity and churn.** Establishments are identified by license number; re-licensing can split or
  merge what is operationally one restaurant.
- **Correlated examples.** An establishment contributes up to 16 training rows and each label is
  reused by ~3 of them, so the effective sample size is much smaller than the row count, and the
  per-date comparisons are not independent.
- **No capacity model.** We optimise a fixed K = 100; we do not model travel, scheduling, re-visit
  timing, or the cost of not inspecting someone else.

## 8. How to run

```bash
pip install -r requirements.txt
python run_project.py                  # ~1-2 min with the cached snapshot and panel
python run_project.py --rebuild-panel  # rebuild data/panel_examples.csv.gz from the raw snapshot
python run_project.py --refresh        # re-download the live dataset first (~170 MB)
python model_selection.py              # optional: print the pre-2020 model-selection table only
```

Fixed random seed (42); a rerun on the same snapshot reproduces every output. Tested with Python 3.12,
pandas 3.0, scikit-learn, xgboost 3.x. Exact XGBoost numbers can move slightly across library versions.

## 9. Files

| Path | Content |
|---|---|
| `run_project.py` | Entry point: runs the whole pipeline |
| `model_selection.py` | Prints the pre-2020 model-selection table on its own |
| `food_inspections/config.py` | Paths, decision dates, features, model configurations, validation schemes |
| `food_inspections/data.py` | Download and cleaning |
| `food_inspections/features.py` | Point-in-time panel construction |
| `food_inspections/models.py` | Design matrix, logistic regression, XGBoost, purged model selection |
| `food_inspections/evaluation.py` | Ranking, Precision@K, shortlists, summary |
| `food_inspections/charts.py` | Figures |
| `food_inspections/pipeline.py` | The end-to-end run |
| `data/raw_canvass_inspections.csv.gz` | Raw snapshot (the submitted dataset) |
| `data/panel_examples.csv.gz` | Analytic dataset: features + labels per (establishment, decision date) |
| `results/metrics_summary.json` | Headline numbers (source of the tables above) |
| `results/metrics_by_date.csv` | Precision@K, failures found, base rate and shortlist overlaps per decision date |
| `results/model_selection.csv` | Validation scores of every candidate model |
| `results/feature_importance.csv` | XGBoost gain importance |
| `results/logistic_coefficients.csv` | Logistic regression standardised coefficients |
| `results/shortlist_latest.csv` | The three top-100 shortlists for the latest test date (2025-06-30) |
| `results/shortlists_all_test_dates.csv.gz` | Shortlists for every test decision date |
| `charts/fig1..fig7.png` | Figures for the presentation |

Figure guide: `fig1` context (inspection volume and failure rate over time) · `fig2` Precision@100 per
decision date · `fig3` Precision@K curve · `fig4` shortlist overlap · `fig5` geography of the
shortlists · `fig6` XGBoost feature importance · `fig7` logistic regression coefficients.
