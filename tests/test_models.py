"""Tests for time-purged selection and fixed feature schemas."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from food_inspections import config as C
from food_inspections import models
from food_inspections.evaluation import mean_precision_at_k


class FakeModel:
    """A cheap candidate whose supplied scores control the test ranking."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        score = X["mock_score"].to_numpy()
        return np.column_stack((1 - score, score))


class ModelSelectionTests(unittest.TestCase):
    def test_selection_uses_purged_folds_and_can_choose_baseline(self):
        dates = pd.to_datetime(["2016-01-01", "2018-03-31", "2018-12-31", "2019-03-31"])
        # On the purged validation date the baseline ranks the failure first;
        # on the unpurged date the fitted models rank it first instead.
        panel = pd.DataFrame({
            "decision_date": np.repeat(dates, 2),
            "license_": ["1", "2"] * 4,
            "label": [1, 0] * 4,
            "hist_fail_rate_smooth": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.9],
        })
        X = pd.DataFrame({"mock_score": [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1]})
        schemes = {
            "purged": {"fit": [dates[0]], "val": [dates[1]], "purged": True},
            "unpurged": {"fit": [dates[2]], "val": [dates[3]], "purged": False},
        }

        def candidate(name):
            return None if name == models.BASELINE_NAME else FakeModel()

        with patch.object(C, "VALIDATION_SCHEMES", schemes), \
             patch.object(C, "XGB_CONFIGS", {"fake xgb": {}}), \
             patch.object(models, "LOGISTIC_CONFIGS", {"fake logistic": 1.0}), \
             patch.object(models, "NEURAL_CONFIGS", {}), \
             patch.object(models, "EXTRA_TREES_CONFIGS", {}), \
             patch.object(models, "make_candidate", side_effect=candidate), \
             patch.object(models, "mean_precision_at_k",
                          side_effect=lambda p, s: mean_precision_at_k(p, s, k=1)):
            winner, best_xgb, table = models.select_model(panel, X, verbose=False)

        self.assertEqual(winner, "baseline")
        self.assertEqual(best_xgb, "fake xgb")
        self.assertEqual(table.loc["baseline", "purged"], 1.0)
        self.assertEqual(table.loc["baseline", "unpurged"], 0.0)
        self.assertEqual(table.loc["fake xgb", "purged"], 0.0)
        self.assertEqual(table.loc["fake xgb", "unpurged"], 1.0)
        self.assertEqual(table.loc["baseline", "criterion (mean of purged)"], 1.0)
        self.assertTrue(table.loc["baseline", "selected"])
        self.assertTrue(table.loc["fake xgb", "selected_xgboost"])

    def test_design_matrix_uses_fixed_categories_for_each_partition(self):
        def rows(facilities, index):
            panel = pd.DataFrame(0.0, columns=C.NUMERIC_FEATURES, index=index)
            panel[C.CATEGORICAL_FEATURE] = facilities
            return panel

        train = models.design_matrix(rows(["Restaurant", "unknown"], [4, 8]))
        test = models.design_matrix(rows(["Bakery"], [15]))

        self.assertEqual(list(train.columns), list(test.columns))
        self.assertEqual(list(train.index), [4, 8])
        self.assertEqual(train.loc[4, "fac_Restaurant"], 1.0)
        self.assertEqual(train.loc[8, "fac_Other"], 1.0)
        self.assertEqual(test.loc[15, "fac_Bakery"], 1.0)
        self.assertEqual(test.loc[15, "fac_Restaurant"], 0.0)

    def test_spline_logistic_handles_out_of_range_without_refitting(self):
        values = np.arange(12, dtype=float)
        train = pd.DataFrame({
            "viol_mean_last3": values,
            "hist_fail_rate_smooth": values / 20,
            "days_since_last_canvass": values * 10,
            "other": values % 3,
        })
        validation = pd.DataFrame({
            "viol_mean_last3": [-1e6, 1e6],
            "hist_fail_rate_smooth": [-1e6, 1e6],
            "days_since_last_canvass": [-1e6, 1e6],
            "other": [0.0, 1.0],
        })
        model = models.make_spline_logistic(0.1).fit(train, np.arange(12) % 2)
        spline = model.named_steps["features"].named_transformers_["splines"]
        knots = [basis.t.copy() for basis in spline.bsplines_]
        scale_mean = model.named_steps["scale"].mean_.copy()
        probabilities = model.predict_proba(validation)

        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        for name, basis, original_knots in zip(models.SPLINE_FEATURES, spline.bsplines_, knots):
            # The boundary knots reflect the fit data, not the extreme validation values.
            self.assertAlmostEqual(original_knots[basis.k], train[name].min())
            self.assertAlmostEqual(original_knots[-basis.k - 1], train[name].max())
            np.testing.assert_array_equal(basis.t, original_knots)
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, scale_mean)

    def test_neural_pipeline_fits_scaler_only_on_training_data(self):
        values = np.arange(24, dtype=float)
        train = pd.DataFrame({"feature_a": values, "feature_b": values % 5})
        validation = pd.DataFrame({"feature_a": [-1000.0, 1000.0],
                                   "feature_b": [-2000.0, 2000.0]})
        # A tiny synthetic sample need not converge within the production limit.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = models.make_neural_network(
                {"hidden_layer_sizes": (4, 2), "alpha": 10.0}
            ).fit(train, np.arange(len(train)) % 2)

        scaler = model.named_steps["scale"]
        mean = scaler.mean_.copy()
        scale = scaler.scale_.copy()
        np.testing.assert_allclose(mean, train.mean().to_numpy())
        probabilities = model.predict_proba(validation)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(scaler.mean_, mean)
        np.testing.assert_array_equal(scaler.scale_, scale)
        self.assertFalse(model.named_steps["clf"].early_stopping)


if __name__ == "__main__":
    unittest.main()
