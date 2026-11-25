"""Fast tests for Kumo benchmark helpers; no model weights are required."""
import unittest

import numpy as np
import pandas as pd

from benchmark_kumo import positive_probability


class Output:
    def __init__(self, frame):
        self.frame = frame

    def to_pandas(self):
        return self.frame


class KumoBenchmarkTests(unittest.TestCase):
    def test_selects_failure_class_by_name_not_position(self):
        out = Output(pd.DataFrame({"1": [0.8, 0.1], "0": [0.2, 0.9]}))
        np.testing.assert_allclose(positive_probability(out), [0.8, 0.1])

    def test_rejects_missing_or_invalid_failure_probabilities(self):
        with self.assertRaises(ValueError):
            positive_probability(Output(pd.DataFrame({"0": [1.0]})))
        with self.assertRaises(ValueError):
            positive_probability(Output(pd.DataFrame({"1": [float("nan")]})))


if __name__ == "__main__":
    unittest.main()
