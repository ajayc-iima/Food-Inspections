"""Download the raw snapshot and clean it to one row per routine inspection."""
import re
import time
import urllib.parse
import urllib.request
from io import StringIO

import pandas as pd

from . import config as C

VIOLATION_RE = re.compile(r"\s*#?\s*\d+\s*\.")
OUTCOME_RESULTS = ["Pass", "Fail", "Pass w/ Conditions"]


# --------------------------------------------------------------------------
# Download / load
# --------------------------------------------------------------------------
def _fetch_page(offset: int, retries: int = 3) -> pd.DataFrame:
    params = {
        "$select": C.SELECT_COLS,
        "$where": "inspection_type = 'Canvass'",
        "$order": "inspection_id",
        "$limit": str(C.PAGE_SIZE),
        "$offset": str(offset),
    }
    url = f"{C.DATASET_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                text = resp.read().decode("utf-8")
            return pd.read_csv(StringIO(text), dtype=str, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - robust download
            print(f"    page at offset {offset}: attempt {attempt} failed ({exc})")
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Return the raw Canvass snapshot, downloading it if missing or asked to."""
    if C.RAW_FILE.exists() and not refresh:
        print(f"[data] loading cached snapshot {C.RAW_FILE.relative_to(C.ROOT)}")
        return pd.read_csv(C.RAW_FILE, dtype=str, low_memory=False)

    print("[data] downloading routine (Canvass) inspections from the Chicago Data Portal ...")
    frames, offset = [], 0
    while True:
        page = _fetch_page(offset)
        frames.append(page)
        print(f"    fetched {len(page):>6} rows (offset {offset})")
        if len(page) < C.PAGE_SIZE:
            break
        offset += C.PAGE_SIZE
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="inspection_id")
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(C.RAW_FILE, index=False, compression="gzip")
    print(f"[data] snapshot saved: {C.RAW_FILE} ({len(raw):,} rows)")
    return raw


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def count_violations(text) -> int:
    """Number of numbered violation entries in the raw violations string.

    Both eras use entries separated by '|' that start with the violation
    number, e.g. '18. NO EVIDENCE OF RODENT ...' or '#1. PERSON IN CHARGE ...'.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    return sum(1 for seg in text.split("|") if VIOLATION_RE.match(seg.strip()))


def group_facility(x) -> str:
    if not isinstance(x, str) or not x.strip():
        return "Other"
    s = " ".join(x.strip().split()).title()
    return s if s in C.TOP_FACILITIES else "Other"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["inspection_date"], errors="coerce").dt.normalize()
    df["license_"] = df["license_"].astype(str).str.strip()
    # notna() matters on pandas >= 3, where astype(str) keeps missing values as NA
    df = df[df["date"].notna() & df["license_"].notna() & (df["license_"] != "")
            & (df["license_"] != "0") & (df["license_"] != "nan")].copy()
    df["results"] = df["results"].fillna("").str.strip()
    df["is_fail"] = (df["results"] == "Fail").astype(int)
    df["is_outcome"] = df["results"].isin(OUTCOME_RESULTS)
    df["viol_count"] = df["violations"].map(count_violations).astype(int)
    df["zip"] = df["zip"].astype(str).str.extract(r"(\d{5})", expand=False)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["risk_num"] = pd.to_numeric(df["risk"].astype(str).str.extract(r"(\d+)", expand=False),
                                   errors="coerce")
    df["facility_grouped"] = df["facility_type"].map(group_facility)
    df = df.sort_values(["license_", "date", "inspection_id"], kind="mergesort")
    df = df.drop_duplicates(subset=["license_", "date", "inspection_type", "results"])
    return df.reset_index(drop=True)
