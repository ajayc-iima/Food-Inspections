"""Point-in-time checks for non-routine inspection features."""
import unittest

import pandas as pd

from experiment_nonroutine import EXTRA, clean_other, history_features


class NonRoutineHistoryTests(unittest.TestCase):
    def test_future_inspections_do_not_change_prior_features(self):
        raw = pd.DataFrame([
            {"inspection_id": "1", "license_": "11", "inspection_date": "2019-01-01",
             "inspection_type": "Complaint", "results": "Fail", "violations": "1. issue"},
            {"inspection_id": "2", "license_": "11", "inspection_date": "2019-10-01",
             "inspection_type": "Canvass Re-Inspection", "results": "Pass", "violations": ""},
            {"inspection_id": "3", "license_": "11", "inspection_date": "2020-01-01",
             "inspection_type": "Complaint", "results": "Fail", "violations": "1. issue"},
        ])
        date = pd.Timestamp("2019-12-31")
        cleaned = clean_other(raw)
        original = history_features(cleaned, date).set_index("license_")
        altered = cleaned.copy()
        altered.loc[altered["date"] > date, "is_fail"] = 0
        changed = history_features(altered, date).set_index("license_")
        pd.testing.assert_frame_equal(original[EXTRA], changed[EXTRA])
        self.assertEqual(original.loc["11", "nr_total"], 2)
        self.assertEqual(original.loc["11", "nr_complaint_fail_12m"], 1)
        self.assertEqual(original.loc["11", "nr_reinspect_12m"], 1)
        self.assertEqual(original.loc["11", "nr_last_failed"], 0)


if __name__ == "__main__":
    unittest.main()
