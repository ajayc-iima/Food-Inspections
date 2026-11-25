"""Figures for the presentation (written to charts/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .evaluation import top_k  # noqa: E402


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def _save(fig, name):
    fig.savefig(C.CHARTS_DIR / name, dpi=160)
    plt.close(fig)


def fig1_context(df: pd.DataFrame):
    counts = df.groupby(df["date"].dt.year).size()
    counts = counts[counts.index <= 2025]
    yr = df[df["is_outcome"]].assign(year=lambda d: d["date"].dt.year)
    rate = yr.groupby("year")["is_fail"].mean()
    rate = rate[(rate.index >= 2010) & (rate.index <= 2025)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.bar(counts.index, counts.values, color="#9ecae1", edgecolor="#3182bd", linewidth=0.5)
    ax.set_title("Routine (Canvass) inspections per year")
    ax.set_xlabel("Year"); ax.set_ylabel("Inspections")
    _style(ax)
    ax = axes[1]
    ax.plot(rate.index, rate * 100, marker="o", color="#d62728", linewidth=1.8)
    ax.axvspan(2020, 2022, color="#f0f0f0", zorder=0)
    ax.text(2021, ax.get_ylim()[1] * 0.97, "COVID", ha="center", va="top", fontsize=9, color="#666")
    ax.set_title("Failure rate of routine inspections (%)")
    ax.set_xlabel("Year"); ax.set_ylabel("Failures / inspected (%)")
    _style(ax)
    fig.suptitle("Chicago food-safety inspections, 2010-2025", fontsize=12, y=1.04)
    _save(fig, "fig1_context.png")


def fig2_precision_dates(metrics: pd.DataFrame):
    m = metrics.sort_values("decision_date")
    x = pd.to_datetime(m["decision_date"])
    tr, te = m["period"] == "train", m["period"] == "test"
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01"), color="#f5f5f5", zorder=0)
    for name in C.APPROACHES:
        col = f"prec{C.K}_{name}"
        ax.plot(x[tr], m.loc[tr, col] * 100, marker="o", markersize=4, linewidth=1.2,
                color=C.COLORS[name], alpha=0.55, label=f"{C.APPROACH_LABELS[name]} (train, in-sample)")
        ax.plot(x[te], m.loc[te, col] * 100, marker="o", markersize=5, linewidth=2.0,
                color=C.COLORS[name],
                label=f"{C.APPROACH_LABELS[name]} (test, mean {m.loc[te, col].mean() * 100:.0f}%)")
    base_te = m.loc[te, "base_rate"].mean() * 100
    ax.axhline(base_te, color="black", linestyle=":", linewidth=1.2)
    ax.text(pd.Timestamp("2022-06-01"), base_te + 1.5,
            f"random selection reference ({base_te:.0f}%)", fontsize=8.5, color="#333")
    top = max(72, float(m.loc[:, [f"prec{C.K}_{a}" for a in C.APPROACHES]].max().max()) * 100 + 5)
    ax.axvline(pd.Timestamp("2020-01-01"), color="#888", linestyle="--", linewidth=1)
    ax.text(pd.Timestamp("2021-01-01"), 0.06, "2020-21 COVID gap\n(excluded)",
            transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#666")
    ax.set_title(f"Precision@{C.K}: share of shortlisted establishments whose next routine inspection failed")
    ax.set_xlabel("Decision date"); ax.set_ylabel(f"Failures found in top {C.K} (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_ylim(0, top)
    _style(ax)
    _save(fig, "fig2_precision_at_100.png")


def fig3_precision_at_k(metrics: pd.DataFrame):
    test = metrics[metrics["period"] == "test"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for name in C.APPROACHES:
        vals = [test[f"prec{k}_{name}"].mean() * 100 for k in C.K_GRID]
        ax.plot(C.K_GRID, vals, marker="o", color=C.COLORS[name], label=C.APPROACH_LABELS[name])
    base = test["base_rate"].mean() * 100
    ax.axhline(base, color="black", linestyle=":", linewidth=1.2)
    ax.text(C.K_GRID[0], base + 0.7, f"random selection ({base:.0f}%)", fontsize=8.5, color="#333")
    ax.set_title("Precision@K averaged over test decision dates")
    ax.set_xlabel("K (number of establishments prioritised)"); ax.set_ylabel("Failures found (%)")
    ax.set_xticks(C.K_GRID)
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_ylim(bottom=0)
    _style(ax)
    _save(fig, "fig3_precision_at_k.png")


def fig4_overlap(metrics: pd.DataFrame):
    test = metrics[metrics["period"] == "test"].sort_values("decision_date")
    chance = (C.K * C.K / test["n_candidates"]).mean()
    x = pd.to_datetime(test["decision_date"])
    pairs = [("baseline_logistic", "Baseline & logistic", "#1f77b4"),
             ("baseline_xgboost", "Baseline & XGBoost", "#d62728"),
             ("logistic_xgboost", "Logistic & XGBoost", "#7b52ab")]
    fig, ax = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    for col, lab, color in pairs:
        vals = test[f"overlap_{col}"]
        ax.plot(x, vals, marker="o", color=color, label=f"{lab} (mean {vals.mean():.0f})")
    ax.axhline(chance, color="black", linestyle=":", linewidth=1.2)
    ax.text(x.iloc[-1], chance + 2, f"expected overlap by chance ({chance:.1f})",
            fontsize=8.5, ha="right", color="#333")
    ax.set_title(f"Establishments shared by two top-{C.K} shortlists, per test decision date")
    ax.set_ylabel(f"Shared establishments (of {C.K})")
    ax.set_ylim(0, C.K)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    _style(ax)
    _save(fig, "fig4_shortlist_overlap.png")


def fig5_map(panel: pd.DataFrame):
    tp = panel[(panel["period"] == "test") & panel["latitude"].notna() & panel["longitude"].notna()]
    backdrop = tp.drop_duplicates(subset="license_")
    ids = {name: set() for name in C.APPROACHES}
    for _, rows in tp.groupby("decision_date"):
        for name in C.APPROACHES:
            ids[name] |= set(top_k(rows, C.SCORE_COLUMNS[name])["license_"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), constrained_layout=True)
    for ax, other in zip(axes, ["logistic", "xgboost"]):
        b, o = ids["baseline"], ids[other]
        ax.scatter(backdrop["longitude"], backdrop["latitude"], s=3, color="#d9d9d9", alpha=0.5,
                   linewidths=0, label="candidate establishments", rasterized=True)
        for sub_ids, color, lab in [(b & o, "#7b52ab", f"in both ({len(b & o)})"),
                                    (b - o, "#7f7f7f", f"baseline only ({len(b - o)})"),
                                    (o - b, C.COLORS[other],
                                     f"{C.APPROACH_LABELS[other]} only ({len(o - b)})")]:
            sub = backdrop[backdrop["license_"].isin(sub_ids)]
            ax.scatter(sub["longitude"], sub["latitude"], s=14, color=color, alpha=0.85,
                       linewidths=0, label=lab)
        ax.set_title(f"Baseline vs {C.APPROACH_LABELS[other]}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Where the top-100 shortlists fall (union over all test decision dates)")
    _save(fig, "fig5_shortlist_map.png")


def fig6_importance(importance: pd.Series):
    imp = importance.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.barh(imp.index, imp.values, color=C.COLORS["xgboost"], alpha=0.85)
    ax.set_title("XGBoost feature importance (gain share)")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=8.5)
    _style(ax)
    _save(fig, "fig6_feature_importance.png")


def fig7_logistic_coefficients(coef: pd.Series):
    top = coef.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.barh(top.index, top.values,
            color=np.where(top.values > 0, C.COLORS["xgboost"], C.COLORS["logistic"]), alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic regression: change in log-odds of failing per 1 SD")
    ax.set_xlabel("Standardised coefficient (red raises, blue lowers failure risk)")
    ax.tick_params(axis="y", labelsize=8.5)
    _style(ax)
    _save(fig, "fig7_logistic_coefficients.png")


def fig8_shortlist_profiles(profile: pd.DataFrame, facility_mix: pd.DataFrame):
    """Who the baseline picks that logistic does not, and vice versa."""
    groups = ["baseline only", "both", "logistic only"]
    colors = {"baseline only": C.COLORS["baseline"], "both": "#7b52ab",
              "logistic only": C.COLORS["logistic"]}
    rows = ["prior routine inspections", "prior failures", "smoothed past failure rate",
            "mean violations, last 3 inspections", "share whose next inspection FAILED"]
    fac = facility_mix.head(6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(rows))
    h = 0.26
    ref = profile.loc[rows, "all candidates"].to_numpy(dtype=float)
    for j, g in enumerate(groups):
        vals = profile.loc[rows, g].to_numpy(dtype=float) / ref
        ax.barh(y + (1 - j) * h, vals, height=h, color=colors[g], label=g)
        for yi, v, raw in zip(y + (1 - j) * h, vals, profile.loc[rows, g]):
            ax.text(v + 0.03, yi, f"{raw:.2f}" if raw < 10 else f"{raw:.0f}",
                    va="center", fontsize=7.5, color="#333")
    ax.axvline(1, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(rows, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Relative to the average candidate (1 = average; labels show raw means)")
    ax.set_title("Track record of the shortlisted establishments")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax)

    ax = axes[1]
    y = np.arange(len(fac))
    for j, g in enumerate(groups + ["all candidates"]):
        color = colors.get(g, "#d9d9d9")
        ax.barh(y + (1.5 - j) * 0.2, fac[g] * 100, height=0.2, color=color, label=g)
    ax.set_yticks(y); ax.set_yticklabels(fac.index, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Share of group (%)")
    ax.set_title("Facility type")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax)
    fig.suptitle("Why the baseline and logistic shortlists differ (pooled over test dates)")
    _save(fig, "fig8_shortlist_profiles.png")


def render_all(df, panel, metrics, importance, coef, profile, facility_mix):
    C.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig1_context(df)
    fig2_precision_dates(metrics)
    fig3_precision_at_k(metrics)
    fig4_overlap(metrics)
    fig5_map(panel)
    fig6_importance(importance)
    fig7_logistic_coefficients(coef)
    fig8_shortlist_profiles(profile, facility_mix)
