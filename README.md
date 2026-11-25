# Which restaurants should be inspected first?

**Prospect-selection project — Machine Learning for Business Analytics (Plaksha MS-AI, Monsoon 2026)**
Setting: municipal food-safety inspection prioritisation on public data from the City of Chicago.

A municipal food-safety department can move only **K = 100 establishments** to the front of its
routine-inspection queue. This project builds and evaluates data-driven ways of choosing those 100 so
that the **number of food-safety failures found** is maximised. Everything is reproducible with one
script on public data; no manual steps, no invented outcomes.

---

## 1. The decision problem

| Component | Formulation |
|---|---|
| Organisation (unnamed) | A city food-safety/health department |
| Opportunity | A licensed food establishment in Chicago |
| Limit | K = 100 establishments prioritised per quarterly decision date |
| Value of a good decision | More genuine food-safety failures found per 100 inspections |
| Prediction target | Whether the establishment's **next routine ("Canvass") inspection fails** |
| Evaluation | **Precision@100** = share of the shortlisted 100 whose next routine inspection failed; secondary: violations cited at those 100 inspections |
| Baseline | Prioritise by smoothed historical failure rate |
| Approaches compared | Baseline vs logistic regression (transparent); plus a validation-selected model and flexible challengers (neural network, Extra Trees, XGBoost) |
| Validation | Features use only information dated on or before each decision date; model selection uses only 2016–2019 decision dates; results reported on 2022–2025 |

## 2. Data and source

| Item | Detail |
|---|---|
| Dataset | **Food Inspections**, City of Chicago Data Portal, dataset id `4ijn-s7e5` |
| Page | https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5 |
| API | Socrata Open Data API: `https://data.cityofchicago.org/resource/4ijn-s7e5.csv` (public, no key) |
| Main extract | All rows with `inspection_type = 'Canvass'` (routine inspections), 2010-01-04 → 2026-11-20: `data/raw_canvass_inspections.csv.gz`, 164,361 rows, downloaded **2026-11-24** |
| Experiment extract | Non-routine inspections (complaints, re-inspections, licence, …) up to 2025-06-30: `data/raw_other_inspections_through_2025_06_30.csv.gz`, used only by `experiment_nonroutine.py` (§7) |

Scope: **routine canvass inspections** are the scheduling decision. Complaint, licence and task-force
inspections are triggered by events, so they are not candidates; they appear only as optional history
features in a separate experiment.

Cleaning (`food_inspections/data.py`): rows with missing/zero license numbers are dropped; duplicate
(license, date, type, result) rows are collapsed; `Fail` is the positive outcome, `Pass` and
`Pass w/ Conditions` are negative; `Out of Business` / `Not Ready` / `No Entry` are non-outcomes that
count as activity but are excluded from failure-rate denominators. Result: **163,316 routine
inspections across 36,991 establishments**. Licenses identify establishments; re-licensing can split
or merge what is operationally one restaurant.

## 3. Formulation: point-in-time examples

Each observation is an **(establishment, decision date)** pair at a quarter-end date T. Features use
only inspections dated ≤ T; the label is the result of the first outcome-bearing canvass in
(T, T + 365 days]. An example exists only if the establishment has ≥ 1 prior inspection **and** its next
routine outcome is observed within the window.

| Feature | Meaning |
|---|---|
| `hist_fail_rate_smooth` | Failure rate over prior outcome-bearing canvasses, shrunk toward the city-wide rate as of T with 6 pseudo-observations (the **baseline score**) |
| `n_canvass_prev`, `n_outcome_prev`, `n_fail_prev`, `has_failed_before` | Inspection and failure history |
| `days_since_last_canvass`, `days_since_first_canvass`, `days_since_last_fail` | Recency (last failure capped at 10 years) |
| `n_canvass_24m`, `n_fail_24m`, `fail_rate_24m` | Activity and failures in the last 24 months (rate over outcome-bearing inspections) |
| `viol_last`, `viol_mean_last3`, `viol_mean_24m` | Violations cited on the last canvass, mean of the last three, mean over 24 months |
| `risk_latest` | Latest known risk category (1 high / 2 medium / 3 low) |
| `facility_grouped` (one-hot) | 11 facility groups + "Other", fixed in advance |
| `zip_hist_fail_rate` | ZIP-level canvass failure rate as of T, shrunk toward the city-wide rate (20 pseudo-observations) |

Panel: **30 decision dates** (16 train: 2016Q1–2019Q4; 14 evaluation: 2022Q1–2025Q2), 100,130 train
and 87,797 evaluation examples. 2020–2021 decision dates are excluded (COVID disruption). Caveats:

- 2019 decision dates have label windows reaching into 2020: **12.5% of training labels come from 2020
  inspections**. 2020–2021 inspections also feed the history features of 2022+ dates.
- Decision dates are three months apart but the label window is twelve months, so **one inspection
  labels about 3 consecutive rows** of the same establishment. This matters for validation (§6).
- The panel cache carries a feature version and is rebuilt automatically when it is out of date.

## 4. The approaches

1. **Approach A — baseline: smoothed historical failure rate.** Inspect the places that failed most
   often before. Ties are broken by license number, as in every ranking in the project.
2. **Approach B — logistic regression.** Standardised features + facility one-hots, default
   regularisation (C = 1). Transparent: coefficients in `results/logistic_coefficients.csv` and
   `charts/fig7_logistic_coefficients.png`.
3. **Validation-selected model.** The pipeline also chooses, on 2016–2019 data only, the best of 15
   candidates: the baseline, four logistic regularisation strengths, two spline-logistic models, a
   small neural network (16-8 hidden units, `alpha = 10`), Extra Trees and six XGBoost configurations.
   On this snapshot it picks the **neural network**.
4. **Flexible challengers** — the best XGBoost and Extra Trees configurations — are reported for
   comparison.

All approaches score the same candidates at every decision date, so the comparison is exact. We
present logistic regression as approach B because it is transparent and does as well as the selected
neural network on 2022–2025 (§6); the neural result is reported alongside it.

## 5. Evaluation design

- **What is evaluated: re-ordering the routine queue.** At each decision date the candidates are the
  establishments the city did canvass in the following 12 months. Each approach re-orders that set,
  and we ask whether its first 100 contain more failures. This is the design Chicago's own 2014 pilot
  used ([CDPH/DoIT](https://chicago.github.io/food-inspections-evaluation)): the same inspections, in
  a better order. Which establishments will be inspected is not known at the decision date, so this
  is a retrospective study of the inspected subset, not a deployable queue (§8).
- **Metrics:** Precision@100 (also K = 25 … 300), and violations cited at the top 100's next canvass,
  which is closer to "how much was wrong". Boston's Yelp tournament and Chicago's 2014 model both
  scored violations rather than pass/fail.
- **Reference:** what 100 random candidates would achieve at that date (24.7% on average).
- **Uncertainty:** per-date paired differences against the baseline, with a 95% moving-block
  bootstrap interval (blocks of 4 quarters, because consecutive dates share labelled inspections).
  The sign test is also reported; it assumes independent dates and so overstates the evidence.
- **Honesty about the 2022–2025 period:** model selection never uses it, but it was looked at during
  development, so its results are labelled **exploratory** in the outputs rather than a fresh,
  untouched test.

## 6. Results

**Model selection (2016–2019 only).** `results/model_selection.csv` scores every candidate on two
**purged** schemes (fit 2016–17 → validate 2019; fit 2016 → validate 2018), where the last fit date is
at least 12 months before the first validation date, so no inspection labels both a fit row and a
validation row. Two unpurged schemes are shown as a leakage diagnostic only.

| Candidate (best of each family) | Purged validation, mean Precision@100 | Leaky check, fit 2016–18 → 2019 |
|---|---|---|
| **Neural network 16-8, alpha = 10 (selected)** | **50.6%** | not run |
| Spline logistic, C = 0.01 | 49.5% | 49.8% |
| Logistic, C = 0.01 | 48.8% | 49.5% |
| Logistic, C = 1 (approach B) | 48.2% | 51.0% |
| XGBoost, depth 5, 400 trees | 46.4% | 49.8% |
| Extra Trees, leaf 75 | 45.8% | not run |
| XGBoost, depth 6, 500 trees | 43.5% | **56.0%** |
| Baseline | 42.9% | 37.2% |

The leaky check makes the deepest XGBoost look clearly best; purged, it is near the bottom. With the
gap, the simpler and smoother models win. The neural network's lead over the logistic models is about
1–2 points on only eight validation dates, which does not establish that it is reliably better.

**Results on 2022–2025 (exploratory; 14 decision dates).**

| Approach | Precision@100 | vs baseline: points, 95% interval | Dates ahead / tied / behind | Violations cited in top 100, per date |
|---|---|---|---|---|
| Random selection (reference) | 24.7% | | | 402 |
| **A · baseline** | **44.1%** | | | 662 |
| **B · logistic regression** | **47.4%** | **+3.3 [+1.6, +5.1]** | 11 / 1 / 2 | **734** (+72 [+57, +91]; ahead on all 14) |
| Neural network (validation-selected) | 47.5% | +3.4 [+1.5, +6.4] | 10 / 1 / 3 | 730 |
| Extra Trees | 45.6% | | | 714 |
| XGBoost (best on purged validation) | 44.1% | 0.0 [−1.3, +1.9] | 7 / 2 / 5 | 716 |

In-sample (train dates) for contrast: baseline 45.3%, logistic 50.5%, neural 51.4%, Extra Trees
70.3%, XGBoost 87.9%.

Precision@K over 2022–2025 (`charts/fig3_precision_at_k.png`):

| K | 25 | 50 | 100 | 150 | 200 | 300 |
|---|---|---|---|---|---|---|
| Baseline | **53.7%** | **49.1%** | 44.1% | 41.5% | 40.2% | 39.3% |
| Logistic | 49.1% | 46.9% | 47.4% | 45.7% | 44.3% | 44.1% |
| Neural (selected) | 46.9% | 45.9% | **47.5%** | **46.6%** | 44.5% | **44.2%** |
| Extra Trees | 47.7% | 48.0% | 45.6% | 45.6% | **46.5%** | 44.1% |
| XGBoost | 49.7% | 46.9% | 44.1% | 43.9% | 44.1% | 42.8% |

**Findings**

- *A simple rule gets far.* The history baseline finds 44% failures in the top 100, against 25% at
  random.
- *The transparent model adds about 3 points at K = 100* (about 3 more failures per 100 inspections;
  95% interval +1.6 to +5.1), ahead on 11 of 14 dates, and finds about 72 more cited violations per
  100 inspections, ahead on all 14. The validation-selected neural network performs the same
  (47.5% vs 47.4%). Fourteen dates over three years is a small sample, so read the interval as
  "clearly positive", not as a precise size. **For very short lists (K ≤ 50) the baseline is best**:
  the establishments with the worst track record are the surest bets.
- *Flexible trees memorise.* XGBoost reaches 88% in-sample but only 44.1% out of sample, no better
  than the baseline. One inspection labels ~3 neighbouring rows, so deep trees can memorise labels,
  and any validation without a 12-month gap rewards that.
- *The shortlists differ sharply, and for a clear reason.* Baseline and logistic top-100s share on
  average only **28.4** of 100 establishments (baseline and XGBoost 26.5, logistic and XGBoost 53.3;
  chance overlap 1.6) — `charts/fig4_shortlist_overlap.png`, `charts/fig5_shortlist_map.png`. Who each
  picks (`charts/fig8_shortlist_profiles.png`, `results/shortlist_profiles.csv`; means over the 14
  dates):

  | | Baseline only | Both | Logistic only | All candidates |
  |---|---|---|---|---|
  | Prior routine inspections | 8.9 | 8.9 | 4.8 | 7.7 |
  | Prior failures | 6.0 | 6.4 | 2.3 | 1.6 |
  | Days since last failure | 466 | 393 | 764 | 1,949 |
  | Mean violations, last 3 inspections | 6.7 | 11.8 | 14.6 | 4.4 |
  | Schools / grocery stores (% of group) | 17% / 6% | 9% / 10% | 4% / 15% | 12% / 6% |
  | **Next inspection failed** | **38%** | **59%** | **43%** | 25% |

  The baseline only sees *how often a place has failed*, so its extra picks are **chronic failers
  with long histories**, many of them schools, whose recent inspections are fairly clean. Logistic
  regression also weighs **how bad the recent inspections were**, so it swaps those for places with
  fewer failures on record but **very high recent violation counts**, more often restaurants and
  grocery stores. Those fail more often next time (43% vs 38%). Places strong on both counts are
  picked by both and fail 59% of the time.

## 7. Further experiments (separate scripts; not part of the headline pipeline)

All of these use the same purged 2016–2019 folds for comparison; where a 2022–2025 number is given it
is exploratory.

| Script → output | What was tried | Purged validation | 2022–25 (exploratory) |
|---|---|---|---|
| `experiment_nonroutine.py` → `results/experiment_nonroutine.json` | 15 history features from **non-routine inspections** (complaints, re-inspections) dated ≤ T | logistic 48.2% → **49.3%**; neural 50.6% → **51.3%** | logistic 47.4% → **49.2%**; neural 47.5% → **49.9%** |
| `experiment_history.py`, `experiment_history_test.py` → `results/experiment_history*.json` | 12 richer routine-history features (last outcome, conditional passes, violation trend) | neural 50.6% → 51.6% at a 60-epoch cap; 50.6% when allowed 150 epochs | 47.5% → 47.8% |
| `experiment_violation_topics.py` → `results/experiment_violation_topics.json` | Past violation topics (pests, temperature, hand-washing, …) | logistic 48.8% → 49.3%; neural 50.6% → 49.6% | — |
| `experiment_ranking.py` → `results/experiment_ranking.json` | XGBoost **learning-to-rank** (`rank:ndcg`, grouped by decision date), a violations regressor, and blends | best ranker 47.0%; no blend beats the neural model (50.6%) | — |
| `rolling_experiment.py` → `results/rolling_experiment.json` | Fixed vs expanding vs recent-only training windows on 2023–25 | — | logistic 47.1 / 46.7 / 45.9%; neural 47.4 / 46.7 / 43.6% |
| `benchmark_kumo.py` → `results/kumo_benchmark.json`, `results/kumo_context2048.json` | Kumo Tabular Small (pretrained tabular model), one estimator, sampled context | 45.4% (512 rows), 45.5% (2,048 rows) | — |

A learning-to-rank objective and Kumo Small did not beat the simple models; a larger Kumo run was
still in progress at the time of this commit.

### Pre-registered confirmation: complaint and re-inspection history

Because the non-routine features were found after 2022–2025 had been examined, we confirmed them under
a plan committed **before** scoring ([`docs/preregistration_nonroutine.md`](docs/preregistration_nonroutine.md)):
fixed feature set (15 features), fixed model (approach B's logistic regression plus those features),
a decision rule, and one **untouched decision date, 2025-09-30** (5,598 candidates; label window to
2026-09-30, complete in the snapshot), scored once. `python confirm_nonroutine.py` →
`results/confirm_nonroutine.json`; no other result file changes.

| Evidence | Logistic | Logistic + complaint history | Difference, 95% interval | History baseline |
|---|---|---|---|---|
| (a) Purged 2016–19 validation, mean of 2 folds | 48.2% | 49.3% | +1.0 (fold 2019: +3.3; fold 2018: −1.3) | 42.9% |
| (b) **Untouched 2025-09-30**, one date | 33% | **43%** | **+10 [+1, +16]** (bootstrap over establishments) | 38% |
| (c) 2022–2025, 14 dates (previously examined) | 47.4% | **49.2%** | **+1.9 [+1.3, +3.6]**, ahead 9 / tied 3 / behind 2 | 44.1% |

All three pre-registered conditions hold, so the gain counts as **confirmed under our rule**: adding
complaint and re-inspection history to approach B raises Precision@100 to 49.2% on 2022–2025
(+5.1 points over the baseline, 95% interval +3.4 to +8.4) and to 43% on the untouched date. The
secondary neural model shows the same pattern (47.5% → 49.9% on 2022–2025; 38% → 43% on the
untouched date). Limits: the validation gain is small and negative on one fold; a single untouched
date has a standard error of about 5 points; and on that date **plain logistic regression (33%) fell
below the history baseline (38%)**, continuing the narrowing seen in 2024–2025. The headline results
in §6 are unchanged; this is reported as a confirmed extension.

### Pre-registered confirmation: how strictly the area is being inspected right now

Chicago's own 2014 model found that the inspectors who last visited an area were its strongest
predictor ([Kannan et al. 2019](https://arxiv.org/abs/1910.04906)). The public data has no inspector
ID, but inspectors work geographic territories. So at each decision date we measure how often routine
inspections of **other** establishments **within 1 km** (and in the same ZIP) failed over the **last
90, 180 and 365 days** (`food_inspections/local_features.py`; 13 features, all dated ≤ T). Tried first
on the purged 2016–19 folds only (`experiment_local.py`), then pre-registered
([`docs/preregistration_local.md`](docs/preregistration_local.md)) and scored once
(`python confirm_local.py` → `results/confirm_local.json`).

**The effect, on 2022–2025.** Share of establishments whose next routine inspection failed, by their
own past-failure band (rows) and the recent local failure rate around them (columns, quintile within
each decision date):

| Own past failure ↓ / local recent failure rate → | 1 (lowest) | 2 | 3 | 4 | 5 (highest) |
|---|---|---|---|---|---|
| Low | 15.4% | 16.8% | 18.5% | 19.6% | **22.1%** |
| Middle | **21.2%** | 23.3% | 23.3% | 24.7% | 27.9% |
| High | 28.4% | 29.2% | 29.8% | 31.8% | **35.5%** |

With the same track record, a place where nearby inspections have recently been failing is **about
1.25–1.45× as likely to fail** as one in the most lenient fifth of areas, and a clean-record place in a
strict area (22.1%) fails about as often as a middling one in a lenient area (21.2%). What is going on
nearby matters almost as much as the restaurant's own record. The data cannot separate inspector
strictness from neighbourhood conditions; the recent rates predict better than the long-run ZIP rate,
which points to something that changes over time, such as inspector assignment.

**The model.** Approach B's logistic regression + complaint history + local recent rates:

| Precision@100 | History baseline | Logistic (B) | + complaint history | **+ local recent rates** | Neural, all features |
|---|---|---|---|---|---|
| Purged 2016–19 validation | 42.9% | 48.2% | 49.3% | **52.9%** | 53.0% |
| 2022–2025, 14 dates | 44.1% | 47.4% | 49.2% | **50.6%** | 51.6% |
| 2025-09-30, untouched for these features | 38% | 33% | 43% | **48%** | 48% |

On 2022–2025 the full model beats the baseline by **+6.5 points (95% interval +4.2 to +9.1), ahead on
11 of 14 dates**, and plain logistic by +3.2 (+2.1 to +4.6). That is **about 15% more failures found
with the same 100 inspections** than the history rule, roughly 90 more failures over the 14 quarters.
On the newest quarter, where plain logistic had slipped below the rule, the full model recovers to
48%. All three pre-registered conditions held. Limits: the gain over logistic + complaint history is
+1.4 with an interval that touches zero (−0.1 to +2.0); 2025-09-30 had been scored once before for the
complaint features; and the feature set was chosen on eight validation dates.

### Business impact: failures caught days earlier with the same inspectors

`python days_earlier.py` → `results/days_earlier.json`. Same design as Chicago's 2014 pilot: at each
decision date, take the routine inspections the city actually carried out in the next 90 days (about
1,700 per quarter, 426 of them failures), keep the same inspection days, and hand them out in each
approach's order. Nothing is added: same inspectors, same number of visits.

| 2022–2025, mean of 14 quarters | Actual order | History rule | Logistic (B) | **Final model** |
|---|---|---|---|---|
| Days earlier each failure is caught | 0 | 6.1 | 9.0 | **10.1** [9.3, 10.9] |
| Failures found in the first half of the quarter's visits | 52% | 60% | 65% | **67%** |
| Days that failing places operate before being caught, removed per quarter | — | 2,616 | 3,839 | **4,321** |

The final model catches failures **10 days sooner** than the city's actual order, on all 14 quarters,
and 4.0 days sooner than the history rule (95% interval +3.7 to +4.6, ahead on all 14). That removes
about **4,300 days of failing restaurants operating unchecked every quarter**, 65% more than the rule,
with no extra inspections. On the untouched 2025-09-30 date: 7.5 days earlier, against 4.2 for the
rule. For comparison, Chicago's own pilot reported 7.4 days over a two-month window, with critical
violations found in the first half rising from 55% to 69%. Assumptions: an establishment's result does
not depend on when in the quarter it is visited, and travel and routing are ignored.

### Adoption: what the department would get, and why it is safe to try

`python adoption_pack.py` → `results/shortlist_2026_q4.csv`, `results/adoption_pack.json`.

**1. A list inspectors can check, not a black box.** The final model scores every establishment with a
routine inspection in the last two years that is not recorded as out of business (9,088 as of
2026-09-30), and every one of the top 100 comes with the three facts that put it there, taken from the
department's own records. Examples from the live list (names are in the CSV):

| # | Facility, ZIP | Why it is on the list |
|---|---|---|
| 1 | Grocery store, 60625 | 25 violations at its last routine inspection (typical: 3) · failed its only past routine inspection |
| 2 | Restaurant, 60613 | 28 violations last time, 17 on average over its last 3 · failed 7 complaint or re-inspection visits · failed 5 of 9 routine inspections |
| 4 | Restaurant, 60647 | 23 violations at each of its last 3 inspections · 440 days since its last routine inspection · risk category 1 |

**2. A guardrail that costs nothing.** Left alone, the model sends more visits to a few areas than the
rule does (on average 17.6 of 100 from one ZIP; 47% from the fifth of areas with the highest recent
local failure rate). A simple cap fixed before testing, **at most 5 of the 100 per ZIP code**, spreads
the list and does not lower its hit rate; it raises it:

| 2022–2025, mean per date | Precision@100 | Distinct ZIPs | Most from one ZIP | From the strictest fifth of areas |
|---|---|---|---|---|
| History rule | 44.1% | 32.9 | 11.8 | 29% |
| Final model | 50.6% | 29.9 | 17.6 | 47% |
| **Final model with ZIP cap** | **52.2%** | **35.4** | **5** | **36.5%** |

On the untouched 2025-09-30 date the capped model also scores 48% (the rule: 38%). The cap was chosen
in advance but checked on dates already examined, so treat its gain as exploratory; its point is that
fairness does not have to cost accuracy.

**3. Nothing else changes.** Every establishment still gets the inspection its risk category requires;
the model only changes the **order**. It uses only data the department already publishes, runs in a
few minutes with one command, and has no inspector-level inputs.

**4. A low-risk way to find out.** A one-quarter pilot: split inspection territories at random; half
work in the model's order, half as usual. Compare failures found per 100 visits and days to detection.
No extra staff; stop at the end of the quarter if the model's half does no better. A randomised pilot
also answers what this retrospective study cannot: what happens at places the city would not otherwise
have visited first. Keep about 10% of slots for random visits so the model keeps learning about places
it would not pick, and re-check it every quarter on the purged validation used here.

## 8. What this establishes — and what it cannot

**Established.** On data the decision-maker could have known at each decision date, ranking by
predicted failure probability finds about twice as many real failures in the next 100 routine
inspections as random selection, and the transparent model beats the natural simple baseline at
K = 100 (+3.3 points, 95% interval +1.6 to +5.1; ahead on 11 of 14 dates over three post-pandemic
years), with more cited violations found on every date. Adding complaint and re-inspection history and the
recent local failure rate, each confirmed under a pre-registered plan, lifts it to **50.6%**: +6.5
points over the baseline (95% interval +4.2 to +9.1), about 15% more failures found. Re-ordering the
city's actual quarterly inspections with it would catch failures about 10 days sooner, removing about
4,300 days of failing restaurants operating unchecked per quarter with the same inspectors (§7). It is a claim about **re-ordering the
routine queue** the city actually ran.

**Where the approach is fine, and where it may look fine but not be useful.** It is fine for deciding
the order of a routine queue that will be worked through anyway. It may look fine but not help
(a) for very short lists, where the baseline does as well or better; (b) for establishments with
little or no history, which never enter the candidate set; and (c) for finding problems at places the
city does not visit, which this data cannot test.

**Cannot be established (limitations).**

- **Uninspected establishments are invisible** (selective labels). Because canvass scheduling
  correlates with risk, precision on the inspected subset may overstate what a re-prioritised queue
  would find. A prospective trial would be needed before deployment.
- **"Failed inspection" ≠ "unsafe restaurant".** `Pass w/ Conditions` counts as a pass; the violation
  metric treats a minor and a critical violation alike.
- **Inspector effects.** Who inspects, and when, changes what gets cited
  ([Kannan et al. 2019](https://arxiv.org/abs/1910.04906);
  [Ibanez and Toffel 2020](https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3318)). The public data
  has no inspector ID.
- **Regime change.** On 1 July 2018 the
  [Chicago Food Code](https://www.chicago.gov/content/dam/city/depts/cdph/FoodProtection/ChicagoFoodCodeMajorChangesFinal2018.pdf)
  renamed the violation classes and raised the number of possible violations from 45 to 63; ten of
  the 16 training dates come before it. The 2020–21 disruption shifted behaviour again.
- **Small, correlated evaluation sample.** 14 evaluation dates and 8 validation dates, with labels
  shared across neighbouring quarters; small differences between models are not meaningful.
- **No capacity model.** No travel, scheduling or re-visit timing.

### Related work

- **Chicago's own model (2014).** A 16-feature logistic regression re-ordered two months of canvass
  inspections and found critical violations about 7.4 days earlier on average
  ([CDPH/DoIT](https://chicago.github.io/food-inspections-evaluation)).
- **Its audit.** [Kannan, Shapiro and Bilgic (2019)](https://arxiv.org/abs/1910.04906) found its largest
  coefficient was a dummy for one group of inspectors.
- **History is hard to beat.** Boston's Yelp tournament
  ([Glaeser et al. 2016](https://www.nber.org/papers/w22124)) and a re-analysis of Yelp-based hygiene
  prediction ([Altenburger and Ho 2019](https://dho.stanford.edu/wp-content/uploads/WWW_Final.pdf))
  find that inspection history carries most of the signal, as here.
- **Selective labels.** Outcomes exist only where someone chose to inspect
  ([Lakkaraju et al. 2017](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf)).

## 9. How to run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v   # point-in-time features, cache and selection tests
python run_project.py                     # ~1.5 min with the cached snapshot and panel
python run_project.py --rebuild-panel     # rebuild data/panel_examples.csv.gz from the raw snapshot
python run_project.py --refresh           # re-download the live dataset first (~170 MB)
python model_selection.py                 # print the model-selection table only
python experiment_nonroutine.py           # optional experiments (§7); add --download to refetch
```

Fixed random seed (42); a rerun on the same snapshot reproduces every output in `results/`. Tested
with Python 3.12, pandas 3.0, scikit-learn and xgboost 3.x.

The Kumo benchmark needs NVIDIA's optional `structured-data-models` package and PyTorch
(`python benchmark_kumo.py --help`); on Windows ARM an x64 Python under emulation may be needed for
PyArrow. Model weights download on first use (`HF_HOME` sets the cache).

## 10. Files

| Path | Content |
|---|---|
| `run_project.py` | Entry point: runs the whole pipeline |
| `model_selection.py` | Prints the 2016–2019 model-selection table on its own |
| `food_inspections/` | `config` (paths, dates, candidates, validation schemes) · `data` (download, cleaning) · `features` (point-in-time panel) · `models` and `extra_models` (candidates, purged selection) · `evaluation` (ranking, Precision@K) · `analysis` (shortlist profiles, block bootstrap) · `charts` · `pipeline` |
| `experiment_*.py`, `rolling_experiment.py`, `benchmark_kumo.py` | Separate experiments (§7) |
| `docs/preregistration_nonroutine.md`, `confirm_nonroutine.py` | Pre-registered confirmation of the complaint-history features (§7) |
| `data/raw_other_inspections_through_2025_09_30.csv.gz` | Non-routine inspections through the untouched holdout date |
| `food_inspections/local_features.py`, `experiment_local.py`, `docs/preregistration_local.md`, `confirm_local.py` | Recent local failure rates: features, validation experiment, pre-registration, confirmation (§7) |
| `tests/` | Unit tests |
| `data/raw_canvass_inspections.csv.gz` | Raw routine-inspection snapshot (the submitted dataset) |
| `data/raw_other_inspections_through_2025_06_30.csv.gz` | Non-routine inspections for the §7 experiment |
| `data/panel_examples.csv.gz` | Analytic dataset: features + labels per (establishment, decision date) |
| `results/metrics_summary.json` | Headline numbers (source of the tables above) |
| `results/metrics_by_date.csv` | Precision@K, violations, base rate and overlaps per decision date |
| `results/model_selection.csv` | Validation scores of every candidate |
| `results/shortlist_profiles.csv`, `results/shortlist_facility_mix.csv` | Who the baseline-only, logistic-only and shared picks are |
| `results/feature_importance.csv`, `results/logistic_coefficients.csv` | XGBoost importance and logistic coefficients |
| `results/shortlist_latest.csv`, `results/shortlists_all_test_dates.csv.gz` | Top-100 shortlists |
| `results/experiment_*.json`, `results/rolling_experiment.json`, `results/kumo_*.json` | Experiment outputs |
| `results/confirm_nonroutine.json`, `results/confirm_local.json`, `results/experiment_local_validation.json` | Pre-registered confirmation results |
| `days_earlier.py` → `results/days_earlier.json` | Business impact: days earlier failures are caught when the actual queue is re-ordered |
| `adoption_pack.py` → `results/shortlist_2026_q4.csv`, `results/adoption_pack.json` | Live shortlist with reasons, fairness check and ZIP-cap guardrail |
| `data/raw_other_inspections_through_2026_09_30.csv.gz` | Non-routine inspections for the live shortlist |
| `charts/fig1..fig8.png` | Figures for the presentation |
| `presentation/food_inspections_presentation.pptx` | The 14-slide presentation with speaker notes; rebuild from current results with `python presentation/build_presentation.py . presentation/food_inspections_presentation.pptx` (needs `python-pptx`) |

Figure guide: `fig1` context · `fig2` Precision@100 per decision date · `fig3` Precision@K · `fig4`
shortlist overlap · `fig5` shortlist geography · `fig6` XGBoost feature importance · `fig7` logistic
coefficients · `fig8` profile of who each shortlist picks.
