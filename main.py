"""
main.py — End-to-end pipeline runner.
After training, saves the scaler + model + threshold so the dashboard
can run the autoencoder on manual entries without retraining.

Usage:  python main.py --input data/Apple_watch_data.csv
"""

import argparse, logging, os, pickle
from app.model import set_global_seeds
set_global_seeds(42)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.preprocessing    import preprocess_pipeline
from app.feature_engineering import feature_pipeline
from app.model            import normalize_data, build_autoencoder, train_autoencoder
from app.anomaly          import calculate_reconstruction_error, detect_anomalies

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
OUTPUT_DIR = "data"


def run_pipeline(raw_data_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Preprocess
    logger.info("=== Step 1: Preprocessing ===")
    df, sleep_df = preprocess_pipeline(raw_data_path)

    daily_sleep = (sleep_df.groupby("date")["duration_hrs"].sum()
                   .reset_index().rename(columns={"duration_hrs": "sleep_hours"})
                   if not sleep_df.empty
                   else pd.DataFrame(columns=["date","sleep_hours"]))
    daily_sleep.to_csv(f"{OUTPUT_DIR}/sleep_data.csv", index=False)

    # 2. Feature engineering
    logger.info("=== Step 2: Feature Engineering ===")
    df = feature_pipeline(df)
    df.to_csv(f"{OUTPUT_DIR}/processed_data.csv", index=False)

    # 3. Normalise
    logger.info("=== Step 3: Normalisation ===")
    scaled_data, scaler, feature_cols = normalize_data(df)

    # 4-5. Build + train
    logger.info("=== Step 4-5: Build & Train Autoencoder ===")
    model   = build_autoencoder(scaled_data.shape[1])
    history = train_autoencoder(model, scaled_data, epochs=20)

    # Save training plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("Autoencoder Training Loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/training_loss.png", dpi=150); plt.close(fig)

    # 6-7. Reconstruction error + anomaly detection
    logger.info("=== Step 6-7: Reconstruction Error & Anomaly Detection ===")
    mse                   = calculate_reconstruction_error(model, scaled_data)
    anomaly_mask, threshold = detect_anomalies(mse, percentile=95)

    df["reconstruction_error"] = mse.astype(float)
    df["anomaly"]              = anomaly_mask.astype(int)
    df.to_csv(f"{OUTPUT_DIR}/final_data.csv", index=False)

    # 8. Persist artefacts for manual-entry reuse
    logger.info("=== Step 8: Saving pipeline artefacts ===")

    model.save(f"{OUTPUT_DIR}/autoencoder_model.keras")

    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{OUTPUT_DIR}/feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    with open(f"{OUTPUT_DIR}/anomaly_threshold.txt", "w") as f:
        f.write(str(threshold))

    print("\nPipeline complete!")
    print(f"  Rows        : {len(df):,}")
    print(f"  Anomalies   : {int(anomaly_mask.sum()):,}  ({100*anomaly_mask.mean():.1f}%)")
    print(f"  Threshold   : {threshold:.6f}")
    print(f"  Outputs     : ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/Apple_watch_data.csv")
    run_pipeline(parser.parse_args().input)
