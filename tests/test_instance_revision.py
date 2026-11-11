"""
Pruebas de caja blanca para la revisión post-hoc.
Viven en app/tests, no en ensamble/test.
"""
from unittest.mock import Mock

import numpy as np

from app.services.error_estimator import ErrorEstimator
from app.services.instance_revision import InstanceRevision


def test_revision_only_when_error_high():
    estimator = Mock(spec=ErrorEstimator)
    estimator.predict_error.return_value = np.array([0.05, 0.5])

    revision = InstanceRevision(estimator, threshold=0.1)
    y_pred = np.array([10.0, 20.0])
    X = np.zeros((2, 3))
    historical_X = np.zeros((10, 3))
    historical_y = np.full(10, 15.0)

    revised = revision.revise(X, y_pred, historical_X, historical_y)

    assert revised[0] == 10.0
    assert revised[1] != 20.0


def test_revision_with_no_high_error_returns_same():
    estimator = Mock(spec=ErrorEstimator)
    estimator.predict_error.return_value = np.array([0.01, 0.02])

    revision = InstanceRevision(estimator, threshold=0.1)
    y_pred = np.array([10.0, 20.0])
    revised = revision.revise(
        np.zeros((2, 3)), y_pred, np.zeros((10, 3)), np.zeros(10)
    )

    assert np.array_equal(revised, y_pred)
