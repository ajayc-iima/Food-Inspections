"""A regularized bagged-tree candidate for temporal model comparison.

This candidate differs from the boosted-tree models: each tree makes
randomized splits, and predictions are averaged across trees. Repeated panel
rows can otherwise encourage memorization, so leaves have a minimum size.
"""

from sklearn.ensemble import ExtraTreesClassifier

from . import config as C


EXTRA_TREES_CONFIGS = {
    "extra trees leaf=75": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 75,
        "max_features": 0.8,
    },
}


def make_extra_trees(params: dict) -> ExtraTreesClassifier:
    """Return a deterministic, unfitted ensemble of randomized trees."""
    return ExtraTreesClassifier(random_state=C.SEED, n_jobs=4, **params)
