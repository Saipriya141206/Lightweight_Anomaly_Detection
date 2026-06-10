"""
feature_engineering.py
-----------------------
Derives higher-level health signals from the cleaned time-series:

  stress_score    – HR / HRV  (↑ HR + ↓ HRV = more stress)
  sedentary_index – 1 if steps in window < 10, else 0
  recovery_index  – Active energy / (HR + 1)  (lower HR + effort = recovery)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stress = HeartRate / HeartRateVariabilitySDNN.
    HRV=0 is replaced by 1 to avoid division-by-zero.
    Result is clipped to [0, 20] to remove extreme spikes.
    """
    hr = df["HeartRate"].clip(lower=0)
    hrv = df["HeartRateVariabilitySDNN"].replace(0, 1).clip(lower=1)
    df["stress_score"] = (hr / hrv).clip(0, 20).round(4)
    logger.debug("stress_score stats: %s", df["stress_score"].describe().to_dict())
    return df


def compute_sedentary_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 if the 1-minute step count is below threshold (very inactive), else 0.
    """
    SEDENTARY_THRESHOLD = 10  # steps per minute
    df["sedentary_index"] = (df["StepCount"] < SEDENTARY_THRESHOLD).astype(int)
    return df


def compute_recovery_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recovery = ActiveEnergyBurned / (HeartRate + 1).
    Higher values mean the body is expending energy at a lower heart rate —
    a proxy for efficient, recovery-friendly activity.
    Clipped and normalised to [0, 1].
    """
    energy = df["ActiveEnergyBurned"].clip(lower=0)
    hr = df["HeartRate"].clip(lower=0)
    raw = energy / (hr + 1)
    # Normalise to [0, 1] using 99th percentile as cap
    cap = raw.quantile(0.99) if raw.quantile(0.99) > 0 else 1
    df["recovery_index"] = (raw / cap).clip(0, 1).round(4)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling-mean versions of key metrics to help the autoencoder
    learn smoother temporal patterns.
    """
    for col in ("HeartRate", "StepCount", "ActiveEnergyBurned"):
        if col in df.columns:
            df[f"{col}_roll{window}"] = (
                df[col].rolling(window, min_periods=1).mean().round(4)
            )
    return df


def feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature-engineering steps in sequence."""
    df = compute_stress_score(df)
    df = compute_sedentary_index(df)
    df = compute_recovery_index(df)
    df = add_rolling_features(df)

    logger.info(
        "Feature engineering done. Columns: %s", [c for c in df.columns if c != "startdate"]
    )
    return df
