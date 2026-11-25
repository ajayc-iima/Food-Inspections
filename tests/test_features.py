"""Regression tests for historical rates and panel cache compatibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from food_inspections import features
from food_inspections import config as C


def inspection(license_, date, result, zip_code="60601", violations=1):
    return {
        "license_": license_, "date": pd.Timestamp(date),
        "is_outcome": result in ("Fail", "Pass", "Pass w/ Conditions"),
        "is_fail": int(result == "Fail"), "zip": zip_code,
        "viol_count": violations, "risk_num": 1, "facility_grouped": "Restaurant",
        "dba_name": license_, "latitude": 41.0, "longitude": -87.0,
    }


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2020-12-31")
        self.inspections = pd.DataFrame([
            inspection("a", "2020-01-01", "Fail"),
            inspection("a", "2020-06-01", "No Entry"),
            inspection("a", "2021-01-10", "No Entry"),
            inspection("a", "2021-02-01", "Pass", violations=2),
            inspection("a", "2021-03-01", "Fail", violations=10),
            inspection("b", "2020-03-01", "Pass"),
            inspection("b", "2021-02-01", "Fail"),
            inspection("c", "2020-04-01", "No Entry", zip_code="60602"),
            inspection("c", "2021-03-01", "Pass"),
            inspection("d", "2017-01-01", "Fail", zip_code="60603"),
            inspection("d", "2021-03-01", "Pass"),
        ]).sort_values(["license_", "date"], kind="mergesort")

    def test_rates_use_only_outcome_bearing_inspections(self):
        rows = features.features_at_date(self.inspections, self.date).set_index("license_")
        # A no-entry canvass remains in activity counts but does not count as a pass.
        self.assertEqual(rows.loc["a", "n_canvass_24m"], 2)
        self.assertEqual(rows.loc["a", "n_outcome_prev"], 1)
        self.assertEqual(rows.loc["a", "fail_rate_24m"], 1.0)
        self.assertEqual(rows.loc["c", "fail_rate_24m"], 0.0)
        self.assertEqual(rows.loc["d", "fail_rate_24m"], 0.0)
        global_rate = 2 / 3
        self.assertAlmostEqual(rows.loc["a", "global_rate_at_T"], global_rate)
        self.assertAlmostEqual(rows.loc["a", "hist_fail_rate_smooth"],
                               (1 + C.SMOOTH_EST * global_rate) / (1 + C.SMOOTH_EST))
        self.assertAlmostEqual(rows.loc["a", "zip_hist_fail_rate"],
                               (1 + C.SMOOTH_ZIP * global_rate) / (2 + C.SMOOTH_ZIP))
        # A ZIP with no outcome-bearing history backs off to the global rate.
        self.assertAlmostEqual(rows.loc["c", "zip_hist_fail_rate"], global_rate)

    def test_future_records_cannot_change_history_features(self):
        original = features.features_at_date(self.inspections, self.date).set_index("license_")
        altered = self.inspections.copy()
        future = altered["date"] > self.date
        altered.loc[future, "viol_count"] = 999
        altered.loc[future & altered["is_outcome"], "is_fail"] = 1
        changed = features.features_at_date(altered, self.date).set_index("license_")
        pd.testing.assert_frame_equal(original[C.NUMERIC_FEATURES], changed[C.NUMERIC_FEATURES])
        self.assertEqual(original.loc["a", "label"], 0)
        self.assertEqual(original.loc["a", "label_viol_count"], 2)
        self.assertEqual(original.loc["a", "days_to_next"], 32)

    def test_incompatible_cache_is_rebuilt_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "panel.csv.gz"
            old = pd.DataFrame({"decision_date": [self.date], "days_to_next": [32],
                                "license_": ["a"], "zip": ["60601"],
                                "label_viol_count": [2]})
            old.to_csv(path, index=False, compression="gzip")
            with patch.object(C, "ROOT", root), patch.object(C, "PANEL_FILE", path), \
                    patch.object(features, "build_panel", return_value=old.copy()) as build:
                result = features.load_panel(self.inspections)
                self.assertEqual(result.loc[0, "panel_feature_version"], features.PANEL_FEATURE_VERSION)
                self.assertEqual(build.call_count, 1)
                features.load_panel(self.inspections)
                self.assertEqual(build.call_count, 1)


if __name__ == "__main__":
    unittest.main()
