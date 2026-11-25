# Pre-registration: recent local failure rates + complaint history

Written and committed on 2026-11-25, **before** any model with these features was scored on a date
after 2019. Implemented by `confirm_local.py`; output `results/confirm_local.json`. No existing result
file changes.

## Why

Chicago's own 2014 model found that the inspectors who last visited an area were its strongest
predictor ([Kannan et al. 2019](https://arxiv.org/abs/1910.04906)). The public data has no inspector
ID, but inspectors work geographic territories. So we measure, at each decision date, how often
routine inspections **near** an establishment failed **recently**.

## What was seen before writing this (training period only)

- `experiment_local.py` → `results/experiment_local_validation.json`, purged 2016–19 folds only:
  logistic 48.3% → + local 49.1% → + complaint history 49.3% → **+ both 52.9%** (fold 2019: 50.5 →
  53.3; fold 2018: 46.0 → 52.5). Neural network: 50.6% → 53.0%.
- On 2016–19 training rows, within the same past-failure band, the next-inspection failure rate rises
  steadily across quintiles of the 180-day local failure rate (e.g. 14.7% → 23.7% for places with
  low past failure; 22.1% → 35.5% for high).
- No model with local features has been scored on 2022–2025 or 2025-09-30.

## Fixed in advance

| Item | Choice |
|---|---|
| Local features (13) | `food_inspections/local_features.py`: smoothed failure rate of other establishments' outcome-bearing routine inspections within 1 km over the last 90/180/365 days, their counts, the same rates by ZIP, the city-wide rates, and the 90-minus-365-day trend; all dated ≤ T, the establishment itself excluded |
| Complaint features (15) | As in `docs/preregistration_nonroutine.md` |
| Primary model | Logistic regression, C = 1, on the existing features + facility one-hots + local (13) + complaint (15) |
| Comparators | History baseline; plain logistic (approach B); logistic + complaint history (confirmed earlier) |
| Secondary (reported only) | Neural network 16-8, alpha = 10, with the same features |
| Training rows | All 2016–2019 decision dates |
| Metric | Precision@100 per decision date; ties by license number |
| Uncertainty | 4-quarter moving-block bootstrap over the 14 dates of 2022–25; bootstrap over establishments for the single 2025-09-30 date (2,000 resamples, seed 42) |

## Decision rule

Counts as **confirmed** only if all hold:

- (a) on 2022–2025, the 95% interval for primary model minus plain logistic is above zero;
- (b) on 2022–2025, the primary model's mean is above logistic + complaint history;
- (c) on 2025-09-30, the primary model is not worse than logistic + complaint history.

Also reported, decided now: the failure rate on 2022–2025 by past-failure band × local-rate quintile,
as a description of the effect.

## Limits stated in advance

- 2025-09-30 was scored once before, for the complaint features; it is untouched for this feature set
  but not a never-seen date.
- The local rate may reflect neighbourhood conditions as well as inspector strictness; the data
  cannot separate them.
- The validation result was found by trying feature sets on only eight validation dates.
