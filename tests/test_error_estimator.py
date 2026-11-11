"""
Pruebas de caja blanca para el estimador de error.
"""
from unittest.mock import Mock

import numpy as np

from app.services.error_estimator import ErrorEstimator


def test_predict_error_raises_if_not_loaded():
    estimator = ErrorEstimator()
    try:
        estimator.predict_error(np.zeros((1, 3)), np.zeros(1))
        assert False, "Debería haber lanzado RuntimeError"
    except RuntimeError:
        pass


def test_predict_error_returns_array():
    estimator = ErrorEstimator()
    estimator.model = Mock()
    estimator.model.predict.return_value = np.array([0.1, 0.2])

    result = estimator.predict_error(np.zeros((2, 3)), np.zeros(2))

    assert isinstance(result, np.ndarray)
    assert len(result) == 2
