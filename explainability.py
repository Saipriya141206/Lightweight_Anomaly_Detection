"""
app/explainability.py
---------------------
Computes which input features contributed most to a high reconstruction
error, and converts them into plain-English explanations.

No changes to the autoencoder or anomaly detection logic —
this only post-processes the reconstruction error per feature.
"""

import numpy as np


# ── Human-readable descriptions for each feature ─────────────────────────────
FEATURE_DESCRIPTIONS = {
    "HeartRate": {
        "high": "Heart rate is higher than your normal pattern — could indicate stress, exertion, or illness.",
        "low":  "Heart rate is lower than usual — may indicate deep rest or bradycardia.",
        "label": "Heart Rate",
        "icon": "❤️",
    },
    "HeartRateVariabilitySDNN": {
        "high": "HRV is unusually high — your nervous system may be in a very relaxed or irregular state.",
        "low":  "Low HRV indicates your body is under stress or hasn't recovered well.",
        "label": "Heart Rate Variability",
        "icon": "📈",
    },
    "RestingHeartRate": {
        "high": "Resting heart rate is elevated — a sign of fatigue, stress, or early illness.",
        "low":  "Resting heart rate is unusually low.",
        "label": "Resting Heart Rate",
        "icon": "💤",
    },
    "StepCount": {
        "high": "Much more physical activity than your normal pattern.",
        "low":  "Low activity level — significantly less movement than usual.",
        "label": "Step Count",
        "icon": "🚶",
    },
    "ActiveEnergyBurned": {
        "high": "Active energy burned is unusually high — intense activity detected.",
        "low":  "Very low active energy — minimal physical effort today.",
        "label": "Active Energy",
        "icon": "🔥",
    },
    "BasalEnergyBurned": {
        "high": "Basal metabolism is running higher than normal.",
        "low":  "Basal energy is lower than your typical baseline.",
        "label": "Basal Energy",
        "icon": "⚡",
    },
    "PhysicalEffort": {
        "high": "Physical effort is unusually high compared to your normal days.",
        "low":  "Very low physical effort recorded.",
        "label": "Physical Effort",
        "icon": "💪",
    },
    "WalkingSpeed": {
        "high": "Walking speed is faster than your normal pace.",
        "low":  "Walking speed is slower than usual — fatigue or inactivity.",
        "label": "Walking Speed",
        "icon": "🏃",
    },
    "RespiratoryRate": {
        "high": "Respiratory rate is elevated — possible stress, exertion, or respiratory issue.",
        "low":  "Breathing rate is lower than normal.",
        "label": "Respiratory Rate",
        "icon": "🌬️",
    },
    "stress_score": {
        "high": "Stress score is significantly above your baseline — high HR relative to HRV.",
        "low":  "Stress score is unusually low — very relaxed state.",
        "label": "Stress Score",
        "icon": "🧠",
    },
    "recovery_index": {
        "high": "Recovery index is unusually high.",
        "low":  "Recovery index is low — body may not be recovering efficiently.",
        "label": "Recovery Index",
        "icon": "🔄",
    },
    "HeartRate_roll5": {
        "high": "Sustained high heart rate over recent period.",
        "low":  "Sustained low heart rate over recent period.",
        "label": "HR Trend",
        "icon": "📊",
    },
    "StepCount_roll5": {
        "high": "Activity has been consistently high recently.",
        "low":  "Activity has been consistently low recently.",
        "label": "Activity Trend",
        "icon": "📊",
    },
    "ActiveEnergyBurned_roll5": {
        "high": "Energy burn trend is higher than normal.",
        "low":  "Energy burn trend is lower than normal.",
        "label": "Energy Trend",
        "icon": "📊",
    },
}


def get_feature_contributions(
    original_scaled: np.ndarray,
    reconstructed_scaled: np.ndarray,
    feature_cols: list[str],
    scaler,
    top_n: int = 4,
) -> list[dict]:
    """
    Compute per-feature squared reconstruction error and return the
    top_n features that contributed most to the anomaly score.

    Returns a list of dicts:
      {feature, label, icon, contribution_pct, direction, explanation}
    """
    # Per-feature squared error
    per_feature_error = np.square(original_scaled - reconstructed_scaled).flatten()

    # Rank by error (highest = most anomalous feature)
    ranked_idx = np.argsort(per_feature_error)[::-1][:top_n]

    total_error = per_feature_error.sum() + 1e-12

    results = []
    for idx in ranked_idx:
        if idx >= len(feature_cols):
            continue
        feat  = feature_cols[idx]
        error = per_feature_error[idx]
        pct   = round(100 * error / total_error, 1)

        # Direction: is the original value higher or lower than reconstruction?
        direction = "high" if original_scaled[0, idx] > reconstructed_scaled[0, idx] else "low"

        desc  = FEATURE_DESCRIPTIONS.get(feat, {})
        label = desc.get("label", feat)
        icon  = desc.get("icon", "📌")
        expl  = desc.get(direction, f"{label} differs from your normal pattern.")

        results.append({
            "feature":          feat,
            "label":            label,
            "icon":             icon,
            "contribution_pct": pct,
            "direction":        direction,
            "explanation":      expl,
        })

    return results


def compute_risk_level(mse: float, threshold: float) -> tuple[str, str]:
    """
    Convert reconstruction error into a risk level.

    Returns (level: str, color: str)
    """
    if mse <= threshold:
        return "Low", "#22c55e"
    ratio = mse / threshold
    if ratio < 1.5:
        return "Medium", "#f59e0b"
    return "High", "#ef4444"
