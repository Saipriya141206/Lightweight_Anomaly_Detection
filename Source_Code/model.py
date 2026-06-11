"""
model.py
--------
Normalisation + Autoencoder training.

SEED STRATEGY (fixes changing anomaly count each run):
  Every source of randomness is seeded with RANDOM_SEED=42:
    - Python random
    - NumPy random
    - TensorFlow / Keras global seed
    - np.random.choice for sampling
    - model.fit shuffle
  With all seeds fixed the autoencoder produces identical weights,
  identical reconstruction errors, and therefore identical anomaly
  counts every single run on the same dataset.
"""

import logging
import random
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Global seed — change this value if you want a different but stable result
RANDOM_SEED = 42

NON_FEATURE_COLS = {"startdate", "date", "anomaly", "reconstruction_error"}


def set_global_seeds(seed: int = RANDOM_SEED):
    """
    Seed every RNG that could affect training outcome.
    Must be called BEFORE importing tensorflow/keras layers.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Force single-threaded ops for full determinism on CPU
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception as e:
        logger.warning("Could not seed TensorFlow: %s", e)

    logger.info("All random seeds set to %d.", seed)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return numeric columns used as model features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS and df[c].dtype != object]


def normalize_data(df: pd.DataFrame):
    """
    Scale all feature columns to [0, 1].

    Returns
    -------
    scaled_array : np.ndarray  shape (n, features)
    scaler       : fitted MinMaxScaler
    feature_cols : list of column names in order
    """
    feature_cols = get_feature_columns(df)
    data = df[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Replace any residual NaN/Inf
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)

    logger.info("Normalised %d rows x %d features.", *scaled.shape)
    return scaled, scaler, feature_cols


def build_autoencoder(input_dim: int):
    """
    Lightweight symmetric autoencoder.
    Seeds must be set before calling this (set_global_seeds()).
    Bottleneck = input_dim // 2 (minimum 4).
    """
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras.regularizers import l2
    except ImportError:
        from keras.models import Model
        from keras.layers import Input, Dense, Dropout
        from keras.regularizers import l2

    bottleneck = max(input_dim // 2, 4)
    mid        = max(bottleneck * 2, 8)

    inp = Input(shape=(input_dim,))

    # Encoder
    x = Dense(mid,        activation="relu", kernel_regularizer=l2(1e-4))(inp)
    x = Dropout(0.1)(x)
    x = Dense(bottleneck, activation="relu")(x)

    # Decoder
    x   = Dense(mid,       activation="relu")(x)
    out = Dense(input_dim, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")

    logger.info(
        "Autoencoder built: input=%d -> bottleneck=%d -> output=%d",
        input_dim, bottleneck, input_dim,
    )
    return model


def train_autoencoder(model, data: np.ndarray, epochs: int = 20, batch_size: int = 256):
    """
    Train the autoencoder with full determinism.

    Key changes vs original:
      - np.random.seed set before sampling
      - sample uses a fixed seed via np.random.choice after seeding
      - shuffle=False  (data already randomly ordered by seeded sampling;
                        shuffling inside fit adds non-determinism)
    """
    n = len(data)

    # Fixed-seed sampling for datasets larger than 50k rows
    sample_size = min(n, 50_000)
    if sample_size < n:
        np.random.seed(RANDOM_SEED)          # re-seed just before sampling
        idx        = np.random.choice(n, sample_size, replace=False)
        train_data = data[idx]
    else:
        # For smaller datasets, shuffle once with fixed seed
        np.random.seed(RANDOM_SEED)
        idx        = np.random.permutation(n)
        train_data = data[idx]

    effective_batch = min(batch_size, max(len(train_data) // 4, 1))

    logger.info(
        "Training autoencoder on %d rows, batch=%d, epochs=%d, seed=%d",
        len(train_data), effective_batch, epochs, RANDOM_SEED,
    )

    history = model.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=effective_batch,
        validation_split=0.1,
        shuffle=False,       # deterministic — data already shuffled above
        verbose=0,
    )

    final_loss = history.history["loss"][-1]
    logger.info("Training complete. Final loss: %.6f", final_loss)

    return history
