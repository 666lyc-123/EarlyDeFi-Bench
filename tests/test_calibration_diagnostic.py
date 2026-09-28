from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_calibration_diagnostic import expected_calibration_error


def test_expected_calibration_error_uses_equal_width_bins() -> None:
    y_true = np.array([0, 1, 1, 1])
    y_probability = np.array([0.1, 0.2, 0.8, 0.9])

    ece, mce, bins = expected_calibration_error(y_true, y_probability, n_bins=2)

    assert isinstance(bins, pd.DataFrame)
    assert bins["count"].tolist() == [2, 2]
    assert bins["mean_probability"].round(3).tolist() == [0.15, 0.85]
    assert bins["empirical_positive_rate"].tolist() == [0.5, 1.0]
    assert round(ece, 3) == 0.25
    assert round(mce, 3) == 0.35


def test_expected_calibration_error_rejects_invalid_probabilities() -> None:
    y_true = np.array([0, 1])
    y_probability = np.array([0.2, 1.2])

    try:
        expected_calibration_error(y_true, y_probability)
    except ValueError as exc:
        assert "probabilities" in str(exc)
    else:
        raise AssertionError("invalid probabilities should raise ValueError")
