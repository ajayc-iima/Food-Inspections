"""Point-in-time panel: one row per (establishment, decision date).

Features use only inspections dated <= T. The label is the result of the
first outcome-bearing canvass in (T, T + LABEL_WINDOW_DAYS].
"""
import numpy as np
import pandas as pd

from . import config as C

# Increment when the meaning or calculation of cached panel features changes.
PANEL_FEATURE_VERSION = 2


def features_at_date(df: pd.DataFrame, T: pd.Timestamp) -> pd.DataFrame:
    hist = df[df["date"] <= T]
    fut = df[(df["date"] > T) & (df["date"] <= T + pd.Timedelta(days=C.LABEL_WINDOW_DAYS))
             & df["is_outcome"]]

    outcomes = hist[hist["is_outcome"]]
    global_rate = float(outcomes["is_fail"].mean()) if len(outcomes) else 0.0

    gb = hist.groupby("license_")
    agg = pd.DataFrame({
        "n_canvass_prev": gb.size(),
        "n_outcome_prev": gb["is_outcome"].sum(),
        "n_fail_prev": gb["is_fail"].sum(),
        "last_date": gb["date"].max(),
        "first_date": gb["date"].min(),
    })
    last_fail_date = hist[hist["is_fail"] == 1].groupby("license_")["date"].max()

    # violation-count statistics (inspection history only)
    v = hist[["license_", "date", "viol_count"]]
    agg["viol_last"] = v.groupby("license_")["viol_count"].last()
    agg["viol_mean_last3"] = v.groupby("license_").tail(3).groupby("license_")["viol_count"].mean()
    recent = T - pd.Timedelta(days=C.HISTORY_WINDOW_DAYS)
    agg["viol_mean_24m"] = v[v["date"] > recent].groupby("license_")["viol_count"].mean()
    c24 = hist[hist["date"] > recent].groupby("license_").agg(
        n_canvass_24m=("is_fail", "size"), n_outcome_24m=("is_outcome", "sum"),
        n_fail_24m=("is_fail", "sum"))
    agg = agg.join(c24)

    # point-in-time zip-level failure rate (smoothed toward the global rate)
    z = outcomes[outcomes["zip"].notna()].groupby("zip")["is_fail"].agg(["sum", "count"])
    zip_rate = (z["sum"] + C.SMOOTH_ZIP * global_rate) / (z["count"] + C.SMOOTH_ZIP)

    lastrow = hist.groupby("license_").tail(1).set_index("license_")
    first_fut = fut.drop_duplicates(subset="license_", keep="first").set_index("license_")

    out = pd.DataFrame(index=agg.index)
    out["n_canvass_prev"] = agg["n_canvass_prev"]
    out["n_outcome_prev"] = agg["n_outcome_prev"]
    out["n_fail_prev"] = agg["n_fail_prev"]
    out["hist_fail_rate_smooth"] = ((agg["n_fail_prev"] + C.SMOOTH_EST * global_rate)
                                    / (agg["n_outcome_prev"] + C.SMOOTH_EST))
    out["days_since_last_canvass"] = (T - agg["last_date"]).dt.days
    out["days_since_first_canvass"] = (T - agg["first_date"]).dt.days
    last_fail = (T - last_fail_date.reindex(out.index)).dt.days
    out["days_since_last_fail"] = last_fail.fillna(3650).clip(upper=3650)
    out["has_failed_before"] = (agg["n_fail_prev"] > 0).astype(int)
    out["n_canvass_24m"] = agg["n_canvass_24m"].fillna(0)
    out["n_fail_24m"] = agg["n_fail_24m"].fillna(0)
    n_outcome_24m = agg["n_outcome_24m"].fillna(0)
    out["fail_rate_24m"] = (out["n_fail_24m"] / n_outcome_24m.replace(0, np.nan)).fillna(0.0)
    out["viol_last"] = agg["viol_last"].fillna(0)
    out["viol_mean_last3"] = agg["viol_mean_last3"].fillna(0)
    out["viol_mean_24m"] = agg["viol_mean_24m"].fillna(0)

    est_zip = lastrow["zip"].reindex(out.index)
    out["zip_hist_fail_rate"] = est_zip.map(zip_rate).fillna(global_rate)
    out["risk_latest"] = lastrow["risk_num"].reindex(out.index).fillna(2.0)
    out["facility_grouped"] = lastrow["facility_grouped"].reindex(out.index).fillna("Other")
    out["dba_name"] = lastrow["dba_name"].reindex(out.index)
    out["zip"] = est_zip
    out["latitude"] = lastrow["latitude"].reindex(out.index)
    out["longitude"] = lastrow["longitude"].reindex(out.index)

    out["label"] = first_fut["is_fail"].reindex(out.index)
    out["days_to_next"] = (first_fut["date"].reindex(out.index) - T).dt.days
    # violations cited at that labelled inspection (secondary outcome, never a feature)
    out["label_viol_count"] = first_fut["viol_count"].reindex(out.index)
    out["decision_date"] = T
    out["global_rate_at_T"] = global_rate

    # candidates: establishments with history whose next routine outcome is observed
    return out[(out["n_canvass_prev"] >= 1) & out["label"].notna()].reset_index()


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for T in C.DECISION_DATES:
        part = features_at_date(df, T)
        parts.append(part)
        print(f"    decision date {T.date()}  candidates={len(part):>6,}  "
              f"fail rate={part['label'].mean():.3f}")
    panel = pd.concat(parts, ignore_index=True)
    panel["period"] = np.where(panel["decision_date"].isin(C.TRAIN_DATES), "train",
                               np.where(panel["decision_date"].isin(C.TEST_DATES), "test", "gap"))
    return panel


def load_panel(df: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Load the cached panel, or build it from the cleaned inspections."""
    panel = None
    if C.PANEL_FILE.exists() and not rebuild:
        print(f"[panel] loading cached panel {C.PANEL_FILE.relative_to(C.ROOT)}")
        panel = pd.read_csv(C.PANEL_FILE, parse_dates=["decision_date"],
                            dtype={"license_": str, "zip": str})
        if ("label_viol_count" not in panel.columns or "panel_feature_version" not in panel.columns
                or panel.empty or not panel["panel_feature_version"].eq(PANEL_FEATURE_VERSION).all()):
            print("[panel] cached panel has an incompatible feature version; rebuilding")
            panel = None
    if panel is None:
        print("[panel] building point-in-time examples ...")
        panel = build_panel(df)
        panel["panel_feature_version"] = PANEL_FEATURE_VERSION
        panel.to_csv(C.PANEL_FILE, index=False, compression="gzip")
    # date of the inspection that supplies the label (used for purging checks)
    panel["label_date"] = panel["decision_date"] + pd.to_timedelta(panel["days_to_next"], unit="D")
    return panel
