# Pre-registration: complaint and re-inspection history features

Written and committed on 2026-11-25, **before** any of the scoring below was run. The confirmation
script `confirm_nonroutine.py` implements exactly this plan; its output goes to
`results/confirm_nonroutine.json` and does not change any existing result file.

## Question

Do history features from **non-routine inspections** (complaint inspections, re-inspections and other
non-Canvass visits) improve the transparent model's Precision@100 over (a) the history baseline and
(b) the same logistic regression without those features?

## Fixed in advance

| Item | Choice |
|---|---|
| Feature set | The 15 features in `experiment_nonroutine.py` (`EXTRA`): `nr_total`, `nr_fail`, `nr_complaint_total`, `nr_complaint_fail`, `nr_12m`, `nr_fail_12m`, `nr_complaint_12m`, `nr_complaint_fail_12m`, `nr_reinspect_12m`, `nr_reinspect_fail_12m`, `nr_days_last`, `nr_days_last_fail`, `nr_days_last_complaint`, `nr_last_failed`, `nr_viol_last`; built only from non-routine records dated on or before the decision date |
| Primary model | Logistic regression, C = 1 (approach B), on the existing 16 features + facility one-hots + the 15 features |
| Comparators | History baseline (approach A); the same logistic regression without the 15 features |
| Secondary model (reported, not used for the decision) | Neural network 16-8, alpha = 10, with and without the features |
| Training rows | All 2016–2019 decision dates, as in the main pipeline |
| Metric | Precision@100 per decision date; ties broken by license number |
| Uncertainty | Paired per-date differences; 95% moving-block bootstrap with 4-quarter blocks over multi-date periods; for a single date, a 95% bootstrap over candidate establishments (2,000 resamples, seed 42) |

## Evidence, in order

1. **Validation gate (pre-2020 only).** Mean Precision@100 over the two purged folds (fit 2016–17 →
   validate 2019; fit 2016 → validate 2018). Already examined in `experiment_nonroutine.py`; rerun
   here for the record.
2. **Untouched holdout: decision date 2025-09-30.** No result for this date has ever been computed.
   Its label window (to 2026-09-30) is complete in the 2026-11-24 snapshot. Non-routine features for
   it need an extract through 2025-09-30, downloaded separately. It is scored **once**.
3. **2022–2025 (14 dates).** Reported with the block bootstrap, but **not untouched**: the
   exploratory script already showed logistic 47.4% → 49.2% on these dates, so this is a
   re-statement, not a confirmation.

## Decision rule

The gain counts as **confirmed** only if all three hold:

- (a) the mean purged-validation gain over plain logistic is above zero;
- (b) on the untouched 2025-09-30 date, logistic + features is not worse than plain logistic;
- (c) on 2022–2025, the 95% block-bootstrap interval for logistic + features minus plain logistic
  excludes zero.

If confirmed, the README, deck and PowerPoint report it as a confirmed improvement, still noting that
one untouched date has low power. If not, they report it as an unconfirmed lead and the headline
results stay unchanged.

## Known limits, stated in advance

- One untouched date is weak evidence: Precision@100 on a single date has a standard error of about
  5 points, larger than the ~2-point gain being tested.
- The non-routine extract is a current snapshot; later corrections to old records cannot be ruled
  out.
- The 2016–2019 folds and the 2022–2025 dates have both been examined during development.
