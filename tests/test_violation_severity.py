"""Checks for exploratory point-in-time comment indicators."""
import unittest

import pandas as pd

from experiment_violation_severity import comment_history, comment_indicators


class CommentIndicatorTests(unittest.TestCase):
    def test_indicators_use_comments_not_headings(self):
        found = comment_indicators("18. HAND WASHING SINK - Comments: OBSERVED MOUSE DROPPINGS")
        self.assertEqual(found["hand_sink"], 0)
        self.assertEqual(found["rodent_evidence"], 1)
        self.assertEqual(comment_indicators(None)["rodent_evidence"], 0)

    def test_future_text_does_not_affect_features(self):
        rows = []
        for date, text in [("2018-01-01", "18. X - Comments: LIVE ROACHES FOUND"),
                           ("2020-01-01", "2. X - Comments: HAND SINK BLOCKED")]:
            rows.append({"license_": "a", "date": pd.Timestamp(date), **comment_indicators(text)})
        frame = pd.DataFrame(rows)
        cutoff = pd.Timestamp("2019-12-31")
        before = comment_history(frame, cutoff)
        frame.loc[frame["date"] > cutoff, "live_pests"] = 99
        pd.testing.assert_frame_equal(before, comment_history(frame, cutoff))
        self.assertEqual(before.loc[0, "comment_last_live_pests"], 1)
        self.assertEqual(before.loc[0, "comment_last_hand_sink"], 0)


if __name__ == "__main__":
    unittest.main()
