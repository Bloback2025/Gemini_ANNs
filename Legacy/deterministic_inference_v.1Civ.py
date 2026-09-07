#!/usr/bin/env python3
"""
deterministic_inference_v.1Civ.py

Robust inference script for HO ANNs:
- Loads scaler JSON with BOM-tolerant encoding
- Honors scaler["target_type"] == "denorm" or CLI --no-inverse to skip inverse transform
- Applies feature scaling matched to scaler.feature_cols mu/sigma
- Writes explicit pred_denorm column and metrics
- Sanity check to detect denorm mismatches early
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras

# -------------------------
# Utilities
# -------------------------
def setup_logger(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("deterministic_inference")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def safe_make_outdir(path: Optional[str]) -> str:
    if not path or not isinstance(path, str):
        return os.getcwd()
    p = os.path.abspath(os.path.expanduser(path.strip()))
    os.makedirs(p, exist_ok=True)
    return p

def load_json_safe(path: str) -> Dict:
    """
    Load JSON tolerant of a UTF-8 BOM and return a dict.
    Raises FileNotFoundError if missing and ValueError on parse errors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler JSON not found: {path}")
    # 'utf-8-sig' will strip BOM if present
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise KeyError("Input CSV must contain 'Close' column.")
    df = df.copy()
    df["PrevClose"] = df["Close"].shift(1).bfill()
    df["Close_3d_mean"] = df["Close"].rolling(window=3, min_periods=1).mean()
    df["Close_7d_mean"] = df["Close"].rolling(window=7, min_periods=1).mean()
    return df

def map_scaler_features_to_csv(scaler: Dict, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    csv_map = {c.lower(): c for c in df.columns}
    feat_cols = scaler.get("feature_cols", []) if isinstance(scaler, dict) else []
    mapped = []
    missing = []
    for f in feat_cols:
        key = f.lower()
        if key in csv_map:
            mapped.append(csv_map[key])
        else:
            missing.append(f)
    return mapped, missing

def select_features_for_model(scaler: Dict, df: pd.DataFrame, model: keras.Model, logger: logging.Logger) -> Tuple[np.ndarray, List[str]]:
    feat_list = None
    if isinstance(scaler, dict) and "feature_cols" in scaler:
        mapped, missing = map_scaler_features_to_csv(scaler, df)
        if missing:
            logger.debug("Some scaler features missing from CSV: %s", missing)
        feat_list = mapped
    if not feat_list:
        feat_list = [c for c in df.columns if c not in ("Date", "Close") and pd.api.types.is_numeric_dtype(df[c])]
        logger.debug("scaler.json missing feature_cols; falling back to CSV numeric columns.")
    try:
        N_expected = model.input_shape[1] if isinstance(model.input_shape, tuple) else model.inputs[0].shape[1]
        N_expected = int(N_expected)
    except Exception:
        raise RuntimeError("Unable to determine model input shape.")
    if len(feat_list) < N_expected:
        raise RuntimeError(f"Feature list length {len(feat_list)} is less than model expected {N_expected}. Update scaler or model.")
    selected = feat_list[:N_expected]
    logger.info("Selecting features for inference: %s", selected)
    X = df[selected].astype(float).values
    return X, selected

def apply_feature_scaling(X: np.ndarray, selected_feats: List[str], scaler: Dict, logger: logging.Logger) -> np.ndarray:
    if not isinstance(scaler, dict):
        return X
    if "mu" in scaler and "sigma" in scaler and "feature_cols" in scaler:
        scaler_map = {name.lower(): (float(mu), float(sig)) for name, mu, sig in zip(scaler["feature_cols"], scaler["mu"], scaler["sigma"])}
        mu_list = []
        sigma_list = []
        for f in selected_feats:
            key = f.lower()
            if key in scaler_map:
                mu_list.append(scaler_map[key][0])
                sigma_list.append(scaler_map[key][1])
            else:
                mu_list.append(0.0)
                sigma_list.append(1.0)
        mu = np.array(mu_list, dtype=float)
        sigma = np.array(sigma_list, dtype=float)
        logger.debug("Applied scaler mu/sigma for selected features (first 3 mu): %s", mu[:3].tolist())
        return (X - mu) / (sigma + 1e-12)
    else:
        logger.debug("Scaler missing mu/sigma or feature_cols; skipping feature scaling.")
        return X

def inverse_transform_preds(preds_std: np.ndarray, scaler: Dict, logger: logging.Logger) -> np.ndarray:
    if isinstance(scaler, dict) and "target_mu" in scaler and "target_sigma" in scaler:
        tmu = float(scaler["target_mu"])
        tsig = float(scaler["target_sigma"])
        logger.debug("Applying target inverse transform with target_mu=%s target_sigma=%s", tmu, tsig)
        return preds_std * tsig + tmu
    if isinstance(scaler, dict) and "feature_cols" in scaler and "mu" in scaler and "sigma" in scaler:
        try:
            idx = [c.lower() for c in scaler["feature_cols"]].index("close")
            mu_close = float(scaler["mu"][idx])
            sigma_close = float(scaler["sigma"][idx])
            logger.debug("Falling back to Close mu/sigma for inverse transform: mu=%s sigma=%s", mu_close, sigma_close)
            return preds_std * sigma_close + mu_close
        except ValueError:
            logger.debug("Close not found in scaler.feature_cols; cannot inverse-transform preds automatically.")
    logger.debug("Scaler missing target_mu/target_sigma and Close mu/sigma; returning standardized preds.")
    return preds_std

def compute_metrics_and_save(preds: np.ndarray, df: pd.DataFrame, outdir: str, logger: logging.Logger) -> None:
    y_true = df["Close"].values[1: len(preds) + 1]
    preds_aligned = preds[: len(y_true)]
    naive = df["Close"].shift(1).bfill().values[1: len(preds) + 1]

    mae_ann = float(np.mean(np.abs(preds_aligned - y_true)))
    rmse_ann = float(np.sqrt(np.mean((preds_aligned - y_true) ** 2)))
    mae_naive = float(np.mean(np.abs(naive - y_true)))
    rmse_naive = float(np.sqrt(np.mean((naive - y_true) ** 2)))

    logger.info("ANN MAE: %.6f, RMSE: %.6f", mae_ann, rmse_ann)
    logger.info("Naive MAE: %.6f, Naive RMSE: %.6f", mae_naive, rmse_naive)

    preds_df = pd.DataFrame({
        "Date": df["Date"].values[1: len(preds) + 1],
        "y_true": y_true,
        "pred": preds_aligned,
        "naive": naive
    })
    preds_df["pred_denorm"] = preds_aligned

    preds_csv = os.path.join(outdir, "preds.csv")
    preds_df.to_csv(preds_csv, index=False)
    logger.info("Saved predictions to %s", preds_csv)

    metrics_txt = os.path.join(outdir, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as fh:
        fh.write(f"ANN_MAE: {mae_ann:.6f}\nANN_RMSE: {rmse_ann:.6f}\n")
        fh.write(f"Naive_MAE: {mae_naive:.6f}\nNaive_RMSE: {rmse_naive:.6f}\n")
    logger.info("Saved metrics to %s", metrics_txt)

# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic inference for HO ANNs")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for preds and metrics")
    p.add_argument("--test_csv", type=str, required=True, help="Test CSV path (must include Date and Close)")
    p.add_argument("--model", type=str, required=True, help="Trained Keras model path (.keras/.h5)")
    p.add_argument("--scaler", type=str, required=True, help="Scaler JSON path")
    p.add_argument("--batch_size", type=int, default=1024, help="Batch size for predict")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--no-inverse", action="store_true", help="Skip inverse transform of model outputs (treat preds as denorm)")
    return p.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(args.debug)

    outdir = safe_make_outdir(args.outdir)
    logger.debug("Using outdir: %s", outdir)

    if not os.path.exists(args.test_csv):
        logger.error("Test CSV not found: %s", args.test_csv)
        sys.exit(2)
    if not os.path.exists(args.model):
        logger.error("Model file not found: %s", args.model)
        sys.exit(2)
    if not os.path.exists(args.scaler):
        logger.error("Scaler JSON not found: %s", args.scaler)
        sys.exit(2)

    try:
        model = keras.models.load_model(args.model)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        sys.exit(3)

    try:
        scaler = load_json_safe(args.scaler)
    except Exception as e:
        logger.exception("Failed to load scaler.json: %s", e)
        sys.exit(3)

    try:
        df = pd.read_csv(args.test_csv, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    except Exception as e:
        logger.exception("Failed to read test CSV: %s", e)
        sys.exit(3)

    try:
        df = compute_engineered_features(df)
    except Exception as e:
        logger.exception("Failed to compute engineered features: %s", e)
        sys.exit(3)

    try:
        X, selected_feats = select_features_for_model(scaler, df, model, logger)
    except Exception as e:
        logger.exception("Feature selection error: %s", e)
        sys.exit(3)

    try:
        Xn = apply_feature_scaling(X, selected_feats, scaler, logger)
    except Exception as e:
        logger.exception("Feature scaling error: %s", e)
        sys.exit(3)

    try:
        if getattr(Xn, "ndim", None) != 2:
            raise RuntimeError(f"Xn must be 2D array, got ndim={getattr(Xn, 'ndim', None)}")
        N_actual = Xn.shape[1]
        N_expected = model.input_shape[1] if isinstance(model.input_shape, tuple) else model.inputs[0].shape[1]
        N_expected = int(N_expected)
        if N_actual != N_expected:
            feat_info = scaler.get("feature_cols") if isinstance(scaler, dict) else None
            logger.error("Model expects %d features but inference input has %d.", N_expected, N_actual)
            logger.error("Feature list from scaler.json: %s", feat_info)
            logger.error("Selected features used for inference: %s", selected_feats)
            logger.error("Possible fixes: compute the engineered features, or select the first N_expected features.")
            sys.exit(4)
    except Exception as e:
        logger.exception("Preflight validation failed: %s", e)
        sys.exit(4)

    try:
        n_rows = Xn.shape[0]
        logger.info("Running predict on %d rows with batch_size=%d", n_rows, args.batch_size)
        preds_raw = model.predict(Xn, batch_size=args.batch_size).reshape(-1)
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        sys.exit(5)

    try:
        # Decide whether to skip inverse: CLI flag takes precedence, then scaler marker
        skip_inverse = bool(args.no_inverse) or (isinstance(scaler, dict) and scaler.get("target_type") == "denorm")
        if skip_inverse:
            logger.info("Skipping inverse transform; using model raw outputs as pred_denorm (no scaling applied).")
            preds_denorm = preds_raw
        else:
            logger.info("Applying inverse transform to model outputs using scaler.")
            preds_denorm = inverse_transform_preds(preds_raw, scaler, logger)
        preds_denorm = np.asarray(preds_denorm, dtype=float)
    except Exception as e:
        logger.exception("Inverse transform failed: %s", e)
        sys.exit(6)

    try:
        y_sample = df["Close"].values[1: len(preds_denorm) + 1]
        if len(y_sample) > 0:
            mean_diff = abs(preds_denorm[:len(y_sample)].mean() - y_sample.mean())
            std_y = float(np.std(y_sample))
            if std_y > 0 and mean_diff > 5 * std_y:
                logger.error("Denorm mismatch: preds mean differs from y mean by >5 sigma (%.3f).", mean_diff)
    except Exception:
        logger.debug("Sanity assertion skipped due to insufficient data.")

    try:
        compute_metrics_and_save(preds_denorm, df, outdir, logger)
    except Exception as e:
        logger.exception("Failed to compute metrics or save outputs: %s", e)
        sys.exit(7)

    logger.info("Inference completed successfully.")

if __name__ == "__main__":
    main()


