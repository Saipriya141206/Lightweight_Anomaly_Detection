"""
preprocessing.py
----------------
Loads, cleans, and pivots raw Apple Watch CSV data into a
time-series dataframe ready for feature engineering.

KEY FIX (v3):
  StepCount, ActiveEnergyBurned, BasalEnergyBurned, PhysicalEffort
  are COUNT metrics -- Apple Health stores them as per-interval totals.
  They must be aggregated with SUM, not mean.
  Using mean() was the root cause of steps/energy not matching the Apple Health app.

  HeartRate, HRV, RespiratoryRate etc. are RATE/LEVEL metrics and
  should use mean().
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

REQUIRED_TYPES = [
    "HeartRate",
    "HeartRateVariabilitySDNN",
    "RestingHeartRate",
    "StepCount",
    "PhysicalEffort",
    "WalkingSpeed",
    "ActiveEnergyBurned",
    "BasalEnergyBurned",
    "RespiratoryRate",
]

SLEEP_TYPE = "SleepAnalysis"

# Metrics that represent cumulative counts in each recording interval.
# These MUST be summed when binning, not averaged.
SUM_METRICS = {
    "StepCount",
    "ActiveEnergyBurned",
    "BasalEnergyBurned",
    "PhysicalEffort",
}

# Metrics that represent instantaneous rates or levels.
# Averaging across a time bin is correct for these.
MEAN_METRICS = {
    "HeartRate",
    "HeartRateVariabilitySDNN",
    "RestingHeartRate",
    "WalkingSpeed",
    "RespiratoryRate",
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load and lightly clean the raw Apple Watch CSV export."""
    df = pd.read_csv(
        file_path,
        usecols=["Type", "Value", "StartDate", "EndDate"],
        low_memory=False,
    )
    df.columns = ["type", "value", "startdate", "enddate"]

    # Strip Apple prefix (handles both HKQuantityTypeIdentifier and HKCategoryTypeIdentifier)
    df["type"] = df["type"].str.replace(
        r"HKQuantityTypeIdentifier|HKCategoryTypeIdentifier",
        "",
        regex=True,
    )

    # Remove timezone suffix (+0530) before datetime parsing
    for col in ("startdate", "enddate"):
        df[col] = df[col].str.replace(r"\s+[+-]\d{4}$", "", regex=True)
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric value for most rows; raw string kept for SleepAnalysis
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")

    df.dropna(subset=["startdate"], inplace=True)

    logger.info("Loaded %d rows from %s", len(df), file_path)
    return df


def filter_health_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the numeric health metrics we need."""
    mask = df["type"].isin(REQUIRED_TYPES) & df["value_num"].notna()
    result = df.loc[mask, ["type", "value_num", "startdate"]].copy()
    result.rename(columns={"value_num": "value"}, inplace=True)
    logger.info(
        "Filtered to health metrics: %d rows, types: %s",
        len(result),
        result["type"].unique().tolist(),
    )
    return result


def extract_sleep_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SleepAnalysis rows.
    Returns a DataFrame with [startdate, enddate, duration_hrs, date].
    """
    sleep_mask = df["type"].str.contains(SLEEP_TYPE, na=False)
    value_mask = df["value"].str.contains("Asleep", na=False)
    sleep_df = df.loc[sleep_mask & value_mask, ["startdate", "enddate"]].copy()

    sleep_df["duration_hrs"] = (
        (sleep_df["enddate"] - sleep_df["startdate"]).dt.total_seconds() / 3600
    )
    sleep_df = sleep_df[
        (sleep_df["duration_hrs"] > 0) & (sleep_df["duration_hrs"] < 14)
    ]
    sleep_df["date"] = sleep_df["startdate"].dt.date

    logger.info("Extracted %d sleep segments.", len(sleep_df))
    return sleep_df


def pivot_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample all health metrics to 1-minute bins, then pivot so each
    metric becomes a column.

    CRITICAL: uses the correct aggregation per metric type.
      SUM  -> StepCount, ActiveEnergyBurned, BasalEnergyBurned, PhysicalEffort
      MEAN -> HeartRate, HRV, RestingHeartRate, WalkingSpeed, RespiratoryRate

    The previous code used aggfunc='mean' for ALL metrics, which caused
    StepCount totals to be ~60x lower than the Apple Health app shows,
    because per-minute counts were averaged instead of summed.
    """
    df = df.sort_values("startdate")
    df["startdate"] = df["startdate"].dt.floor("1min")

    frames = []
    for metric_type, group in df.groupby("type"):
        series = group.set_index("startdate")["value"]
        agg_fn = "sum" if metric_type in SUM_METRICS else "mean"
        resampled = series.resample("1min").agg(agg_fn).rename(metric_type)
        frames.append(resampled)

    if not frames:
        return pd.DataFrame(columns=["startdate"] + REQUIRED_TYPES)

    pivot = pd.concat(frames, axis=1)
    pivot.reset_index(inplace=True)
    pivot.columns.name = None

    sum_cols = [c for c in pivot.columns if c in SUM_METRICS]
    logger.info(
        "Pivoted: %d rows x %d cols | SUM cols: %s",
        len(pivot), len(pivot.columns), sum_cols,
    )
    return pivot


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps and ensure all expected columns exist.

    SUM metrics (steps, energy): NaN after resampling means no activity
    recorded in that minute -- fill with 0, NOT forward-fill.
    Filling steps with the last known value would fabricate movement.

    MEAN metrics (HR, HRV): forward-fill is appropriate since heart rate
    doesn't drop to zero between readings.
    """
    df = df.sort_values("startdate").reset_index(drop=True)

    # Ensure all expected columns exist
    for col in REQUIRED_TYPES:
        if col not in df.columns:
            df[col] = float("nan")

    # SUM metrics: missing = zero activity
    for col in SUM_METRICS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # MEAN metrics: forward-fill then back-fill, then median for any remainder
    mean_cols = [c for c in MEAN_METRICS if c in df.columns]
    df[mean_cols] = df[mean_cols].ffill().bfill()
    for col in mean_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    df.dropna(subset=[c for c in REQUIRED_TYPES if c in df.columns], how="all", inplace=True)

    logger.info("Cleaned time-series: %d rows remain.", len(df))
    return df


def preprocess_pipeline(file_path: str) -> tuple:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df_health : pd.DataFrame  -- cleaned, pivoted health metrics
    df_sleep  : pd.DataFrame  -- sleep segments with duration_hrs and date
    """
    raw = load_data(file_path)
    df_health = filter_health_metrics(raw)
    df_health = pivot_to_timeseries(df_health)
    df_health = clean_timeseries(df_health)
    df_sleep = extract_sleep_data(raw)
    return df_health, df_sleep
