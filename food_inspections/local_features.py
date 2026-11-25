"""Recent failure rates around each establishment, as of the decision date.

Chicago's own 2014 model found that the group of inspectors who last visited
an area was its strongest predictor (Kannan et al. 2019). The public data has
no inspector ID, but inspectors work geographic territories, so the recent
failure rate of routine inspections nearby is a point-in-time proxy for how
strictly an area is currently being inspected.

Every feature uses only outcome-bearing routine inspections dated on or before
the decision date T, and excludes the establishment's own inspections.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0
RADIUS_KM = 1.0
WINDOWS = (90, 180, 365)
SMOOTH = 20  # pseudo-observations toward the city-wide rate in the same window

LOCAL_COLUMNS = [f"near_rate_{w}d" for w in WINDOWS] + [f"near_n_{w}d" for w in WINDOWS] + \
                [f"zip_rate_{w}d" for w in WINDOWS] + [f"city_rate_{w}d" for w in WINDOWS] + \
                ["near_rate_trend"]


def local_features_at(df: pd.DataFrame, T: pd.Timestamp, cand: pd.DataFrame) -> pd.DataFrame:
    """cand: one row per candidate with license_, latitude, longitude, zip."""
    longest = max(WINDOWS)
    hist = df[(df["date"] <= T) & (df["date"] > T - pd.Timedelta(days=longest)) & df["is_outcome"]]
    out = pd.DataFrame({"license_": cand["license_"].to_numpy()})
    age = (T - hist["date"]).dt.days.to_numpy()
    fail = hist["is_fail"].to_numpy().astype(float)
    city = {w: (fail[age < w].mean() if (age < w).any() else 0.0) for w in WINDOWS}

    # own inspections inside each window, to remove the establishment itself
    own = {}
    for w in WINDOWS:
        h = hist[age < w]
        g = h.groupby("license_")["is_fail"].agg(["sum", "count"])
        own[w] = g.reindex(out["license_"]).fillna(0)

    geo = hist["latitude"].notna() & hist["longitude"].notna()
    hg = hist[geo]
    ok = cand["latitude"].notna().to_numpy() & cand["longitude"].notna().to_numpy()
    near_sum = {w: np.zeros(len(cand)) for w in WINDOWS}
    near_cnt = {w: np.zeros(len(cand)) for w in WINDOWS}
    if len(hg) and ok.any():
        tree = BallTree(np.radians(hg[["latitude", "longitude"]].to_numpy()), metric="haversine")
        q = np.radians(cand.loc[ok, ["latitude", "longitude"]].to_numpy())
        idx = tree.query_radius(q, r=RADIUS_KM / EARTH_KM)
        lens = np.array([len(a) for a in idx])
        flat = np.concatenate(idx) if lens.sum() else np.array([], dtype=int)
        who = np.repeat(np.flatnonzero(ok), lens)
        hage = (T - hg["date"]).dt.days.to_numpy()[flat]
        hfail = hg["is_fail"].to_numpy().astype(float)[flat]
        hlic = hg["license_"].to_numpy()[flat]
        not_self = hlic != cand["license_"].to_numpy()[who]
        for w in WINDOWS:
            m = (hage < w) & not_self
            near_sum[w] = np.bincount(who[m], weights=hfail[m], minlength=len(cand))
            near_cnt[w] = np.bincount(who[m], minlength=len(cand)).astype(float)

    zip_g = {w: hist[age < w].groupby("zip")["is_fail"].agg(["sum", "count"]) for w in WINDOWS}
    for w in WINDOWS:
        out[f"near_rate_{w}d"] = (near_sum[w] + SMOOTH * city[w]) / (near_cnt[w] + SMOOTH)
        out[f"near_n_{w}d"] = near_cnt[w]
        z = zip_g[w].reindex(cand["zip"].to_numpy())
        zs = z["sum"].fillna(0).to_numpy() - own[w]["sum"].to_numpy()
        zc = z["count"].fillna(0).to_numpy() - own[w]["count"].to_numpy()
        out[f"zip_rate_{w}d"] = (np.clip(zs, 0, None) + SMOOTH * city[w]) / (np.clip(zc, 0, None) + SMOOTH)
        out[f"city_rate_{w}d"] = city[w]
    out["near_rate_trend"] = out["near_rate_90d"] - out["near_rate_365d"]
    return out


def build_local(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for T, rows in panel.groupby("decision_date"):
        f = local_features_at(df, pd.Timestamp(T), rows[["license_", "latitude", "longitude", "zip"]])
        f["decision_date"] = pd.Timestamp(T)
        parts.append(f)
        print(f"    local features {pd.Timestamp(T).date()}: {len(f):,} rows", flush=True)
    return pd.concat(parts, ignore_index=True)
