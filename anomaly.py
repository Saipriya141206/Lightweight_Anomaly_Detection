"""
anomaly.py
----------
Reconstruction-error calculation and percentile-based anomaly detection.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_reconstruction_error(model, data: np.ndarray) -> np.ndarray:
    """
    Run the autoencoder on `data` and return per-sample MSE.

    Parameters
    ----------
    model : trained Keras autoencoder
    data  : np.ndarray, shape (n, features)  — already scaled to [0,1]

    Returns
    -------
    mse : np.ndarray, shape (n,)
    """
    reconstructed = model.predict(data, verbose=0, batch_size=512)
    mse = np.mean(np.square(data - reconstructed), axis=1)
    logger.info(
        "Reconstruction error — min=%.6f  mean=%.6f  p95=%.6f  max=%.6f",
        mse.min(), mse.mean(), np.percentile(mse, 95), mse.max(),
    )
    return mse.astype(np.float32)


def detect_anomalies(
    mse: np.ndarray,
    threshold: float | None = None,
    percentile: float = 95,
) -> tuple[np.ndarray, float]:
    """
    Label each sample as anomaly (True) or normal (False).

    If `threshold` is None the threshold is set to the `percentile`-th
    percentile of the MSE distribution (dynamic, data-driven).

    Returns
    -------
    anomaly_mask : bool np.ndarray, shape (n,)
    threshold    : float — the cut-off value used
    """
    if threshold is None:
        threshold = float(np.percentile(mse, percentile))

    anomaly_mask = mse > threshold

    n_anomalies = anomaly_mask.sum()
    logger.info(
        "Anomaly detection — threshold=%.6f  anomalies=%d / %d (%.1f%%)",
        threshold, n_anomalies, len(mse), 100 * n_anomalies / len(mse),
    )
    return anomaly_mask, threshold
