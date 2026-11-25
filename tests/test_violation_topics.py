"""Checks for the exploratory point-in-time violation topic features."""
import unittest

import pandas as pd

from experiment_violation_topics import TOPICS, extract_topics, violation_topic_history


class ViolationTopicTests(unittest.TestCase):
    def test_only_numbered_headings_are_matched(self):
        text = ("18. NO EVIDENCE OF RODENT OR INSECT - Comments: blocked hand sink | "
                "22. HAND WASHING FACILITIES - Comments: mice nearby")
        counts = extract_topics(text)
        self.assertEqual(counts["pest"], 1)
        self.assertEqual(counts["hand_hygiene"], 1)
        self.assertEqual(extract_topics("Comments: rodents and hot food"), dict.fromkeys(TOPICS, 0))

    def test_later_inspections_never_affect_historical_features(self):
        rows = []
        for date, text in [("2019-01-01", "18. NO EVIDENCE OF RODENT - Comments: old"),
                           ("2020-01-01", "2. HAND WASHING - Comments: future")]:
            rows.append({"license_": "x", "date": pd.Timestamp(date), **extract_topics(text)})
        frame = pd.DataFrame(rows)
        before = violation_topic_history(frame, pd.Timestamp("2019-12-31")).set_index("license_")
        altered = frame.copy()
        altered.loc[altered["date"] > pd.Timestamp("2019-12-31"), "pest"] = 999
        after = violation_topic_history(altered, pd.Timestamp("2019-12-31")).set_index("license_")
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.loc["x", "topic_last_pest"], 1)
        self.assertEqual(before.loc["x", "topic_last_hand_hygiene"], 0)


if __name__ == "__main__":
    unittest.main()
