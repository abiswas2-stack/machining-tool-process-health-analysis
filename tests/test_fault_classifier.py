"""Tests for the fault classification module."""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from fault_classifier import train_and_evaluate


def test_train_and_evaluate_separates_clearly_distinct_classes():
    """
    Builds a synthetic dataset with two obviously separable classes
    and checks the classifier achieves high accuracy on it.
    """
    rng = np.random.default_rng(seed=0)

    # Class "A": features clustered around low values
    class_a = rng.normal(loc=0.0, scale=0.1, size=(100, 8))
    # Class "B": features clustered around high values, clearly separated
    class_b = rng.normal(loc=5.0, scale=0.1, size=(100, 8))

    X = np.vstack([class_a, class_b])
    y = np.array(["A"] * 100 + ["B"] * 100)

    results = train_and_evaluate(X, y, test_size=0.3, random_state=0)

    # With such clearly separated classes, accuracy should be very high
    assert results["accuracy"] > 0.95


def test_train_and_evaluate_returns_expected_keys():
    """Checks the results dictionary has all expected fields."""
    rng = np.random.default_rng(seed=1)
    X = rng.normal(size=(60, 8))
    y = np.array(["A"] * 30 + ["B"] * 30)

    results = train_and_evaluate(X, y, test_size=0.3, random_state=1)

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert "labels_order" in results
    assert "report" in results
    assert "model" in results