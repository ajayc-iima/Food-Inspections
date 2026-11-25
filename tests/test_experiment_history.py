"""Point-in-time checks for experimental outcome-history features."""
import unittest

import pandas as pd

from experiment_history import EXTRA_COLUMNS, history_features


class HistoricalExperimentTests(unittest.TestCase):
    def test_future_inspections_do_not_affect_features(self):
        frame = pd.DataFrame([
            {"license_": "a", "date": pd.Timestamp("2019-01-01"), "results": "Fail",
             "is_outcome": True, "is_fail": 1, "viol_count": 4},
            {"license_": "a", "date": pd.Timestamp("2019-06-01"), "results": "Pass w/ Conditions",
             "is_outcome": True, "is_fail": 0, "viol_count": 2},
            {"license_": "a", "date": pd.Timestamp("2019-10-01"), "results": "No Entry",
             "is_outcome": False, "is_fail": 0, "viol_count": 0},
            {"license_": "a", "date": pd.Timestamp("2020-01-01"), "results": "Fail",
             "is_outcome": True, "is_fail": 1, "viol_count": 99},
        ])
        date = pd.Timestamp("2019-12-31")
        original = history_features(frame, date).set_index("license_")
        altered = frame.copy()
        altered.loc[altered["date"] > date, "viol_count"] = 999
        changed = history_features(altered, date).set_index("license_")
        pd.testing.assert_frame_equal(original[EXTRA_COLUMNS], changed[EXTRA_COLUMNS])
        self.assertEqual(original.loc["a", "last_outcome_conditional"], 1)
        self.assertEqual(original.loc["a", "last3_fails"], 1)
        self.assertEqual(original.loc["a", "outcome_viol_delta"], -2)
        self.assertEqual(original.loc["a", "outcomes_12m"], 2)


if __name__ == "__main__":
    unittest.main()
