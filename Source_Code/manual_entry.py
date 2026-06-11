"""
manual_entry.py
---------------
Handles manual health data entry for dates with no Apple Watch data.

Flow:
  1. User fills in the 9 raw measurable attributes + sleep hours
  2. Derived features (stress_score, sedentary_index, recovery_index,
     rolling averages) are computed automatically
  3. Autoencoder runs on the single row to get reconstruction_error + anomaly
  4. Result saved to data/manual_log.csv and shown as a full dashboard day
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANUAL_LOG_PATH = "data/manual_log.csv"

# All columns that final_data.csv contains (must match exactly)
MANUAL_LOG_COLS = [
    "date",
    "HeartRate",
    "HeartRateVariabilitySDNN",
    "RestingHeartRate",
    "StepCount",
    "ActiveEnergyBurned",
    "BasalEnergyBurned",
    "PhysicalEffort",
    "WalkingSpeed",
    "RespiratoryRate",
    "sleep_hours",
    "stress_score",
    "sedentary_index",
    "recovery_index",
    "HeartRate_roll5",
    "StepCount_roll5",
    "ActiveEnergyBurned_roll5",
    "reconstruction_error",
    "anomaly",
    "is_manual",          # flag so dashboard can show indicator
]

# Typical healthy reference ranges — shown as placeholder hints in the form
FIELD_HINTS = {
    "HeartRate":                  ("Heart Rate",             "bpm",              40,  200,  72.0,  "Resting: 60–100 bpm"),
    "HeartRateVariabilitySDNN":   ("HRV (SDNN)",             "ms",                1,  200,  45.0,  "Healthy: 20–100 ms"),
    "RestingHeartRate":           ("Resting Heart Rate",     "bpm",              30,  120,  65.0,  "Normal: 60–100 bpm"),
    "StepCount":                  ("Step Count",             "steps",             0, 50000, 8000.0,"Target: 8,000+"),
    "ActiveEnergyBurned":         ("Active Energy Burned",   "kcal",              0,  2000,  300.0, "Typical: 200–600 kcal"),
    "BasalEnergyBurned":          ("Basal Energy Burned",    "kcal",            800,  3000, 1800.0,"Typical: 1400–2000 kcal"),
    "PhysicalEffort":             ("Physical Effort",        "0–1 scale",       0.0,   1.0,   0.3,  "0 = rest, 1 = max effort"),
    "WalkingSpeed":               ("Walking Speed",          "km/h",            0.0,  10.0,   4.5,  "Normal: 3–6 km/h"),
    "RespiratoryRate":            ("Respiratory Rate",       "breaths/min",       8,   40,   15.0,  "Normal: 12–20 br/min"),
    "sleep_hours":                ("Sleep Duration",         "hours",           0.0,  14.0,   7.0,  "Recommended: 7–9 hrs"),
}


def load_manual_log() -> pd.DataFrame:
    """Load existing manual log entries."""
    if os.path.exists(MANUAL_LOG_PATH):
        df = pd.read_csv(MANUAL_LOG_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df
    return pd.DataFrame(columns=MANUAL_LOG_COLS)


def save_manual_log(df: pd.DataFrame):
    df.to_csv(MANUAL_LOG_PATH, index=False)


def get_manual_entry(date) -> pd.Series | None:
    """Return the manual log row for a date, or None if not found."""
    log = load_manual_log()
    row = log[log["date"] == date]
    return row.iloc[0] if not row.empty else None


def compute_derived_features(raw: dict) -> dict:
    """
    Given raw user inputs, compute all derived columns.
    Mirrors exactly what feature_engineering.py does.
    """
    hr  = max(float(raw["HeartRate"]), 1)
    hrv = max(float(raw["HeartRateVariabilitySDNN"]), 1)
    steps  = float(raw["StepCount"])
    energy = float(raw["ActiveEnergyBurned"])

    stress_score    = round(min(hr / hrv, 20), 4)
    sedentary_index = int(steps < 10)

    cap = max(energy / (hr + 1), 1e-6)     # single-row normalisation
    recovery_index  = round(min(cap / cap, 1.0), 4)   # always 1.0 for single row

    # Rolling features: single entry — use the value itself as the "roll"
    hr_roll5     = hr
    steps_roll5  = steps
    energy_roll5 = energy

    derived = {
        "stress_score":              stress_score,
        "sedentary_index":           sedentary_index,
        "recovery_index":            recovery_index,
        "HeartRate_roll5":           hr_roll5,
        "StepCount_roll5":           steps_roll5,
        "ActiveEnergyBurned_roll5":  energy_roll5,
    }
    return derived


def run_autoencoder_on_row(raw: dict, derived: dict, model, scaler, feature_cols: list) -> tuple:
    """
    Build a single-row feature vector, scale it, run the autoencoder,
    and return (reconstruction_error, is_anomaly).

    Uses the same scaler and feature_cols from the pipeline run so the
    reconstruction error is comparable to watch-data days.
    """
    # Build row in exact feature_cols order
    row_dict = {**raw, **derived}
    row_vals = []
    for col in feature_cols:
        row_vals.append(float(row_dict.get(col, 0.0)))

    row_arr = np.array(row_vals, dtype=np.float32).reshape(1, -1)

    # Scale using the trained scaler (clip to [0,1] for out-of-range inputs)
    row_scaled = scaler.transform(row_arr)
    row_scaled = np.clip(row_scaled, 0.0, 1.0)

    # Reconstruct
    reconstructed = model.predict(row_scaled, verbose=0)
    mse = float(np.mean(np.square(row_scaled - reconstructed)))

    # Load threshold from pipeline output if available
    threshold_path = "data/anomaly_threshold.txt"
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.01   # conservative fallback

    is_anomaly = int(mse > threshold)

    logger.info(
        "Manual row — MSE=%.6f  threshold=%.6f  anomaly=%d",
        mse, threshold, is_anomaly,
    )
    return mse, is_anomaly


def save_entry(date, raw: dict, derived: dict, mse: float, is_anomaly: int):
    """Persist the complete manual entry row to manual_log.csv."""
    log = load_manual_log()
    log = log[log["date"] != date]   # remove any previous entry for this date

    row = {"date": date, "is_manual": 1, "reconstruction_error": mse, "anomaly": is_anomaly}
    row.update(raw)
    row.update(derived)

    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    save_manual_log(log)
    logger.info("Manual entry saved for %s", date)
