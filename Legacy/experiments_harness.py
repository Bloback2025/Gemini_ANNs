#!/usr/bin/env python3
"""
experiments_harness.py

End-to-end experiments: permutation test, TimeSeriesSplit CV, baselines, ablation,
residual diagnostics, ensemble and bootstrap uncertainty for next-day close forecasting.
"""

import argparse
import json
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tempfile
from datetime import datetime

# ML libs
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import statsmodels.api as sm

# Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses

# Optional stats
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False

# ----------------------------
# Helpers
# ----------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def build_ann(input_dim, hidden_layers=(64,32), lr=1e-3, dropout=0.0):
    tf.keras.backend.clear_session()
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for h in hidden_layers:
        x = layers.Dense(h, activation="relu")(x)
        if dropout and dropout>0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE safe
    mask = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100) if mask.sum()>0 else None
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape}

def bootstrap_pvalue(diffs, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.mean(diffs)
    n = len(diffs)
    boots = rng.choice(diffs, size=(n_boot, n), replace=True)
    means = boots.mean(axis=1)
    p = np.mean(np.abs(means) >= np.abs(obs))
    return float(p)

# ----------------------------
# Data utilities
# ----------------------------
def load_and_prepare(csv_path, time_col=None, target_col="Actual_Close", feature_cols=None, dropna=True):
    df = pd.read_csv(csv_path)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    if feature_cols is None:
        # default: use first 5 numeric columns (OHLC + 5th)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric[:5]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    return df, X, y, feature_cols

# ----------------------------
# Baselines
# ----------------------------
def naive_one_step(df, target_col="Actual_Close", group_col=None):
    s = df[target_col]
    if group_col and group_col in df.columns:
        naive = df.groupby(group_col)[target_col].shift(1)
    else:
        naive = s.shift(1)
    return naive

def arima_forecast(series, steps=1):
    # Fit a simple ARIMA(1,0,0) or fallback to mean if fails
    try:
        model = sm.tsa.ARIMA(series.dropna(), order=(1,0,0))
        res = model.fit()
        f = res.forecast(steps=steps)
        return f.iloc[-1]
    except Exception:
        return series.dropna().iloc[-1]  # last value fallback

# ----------------------------
# Experiments
# ----------------------------
def time_series_cv_ann(X, y, n_splits=5, ann_kwargs=None, fit_kwargs=None, scaler=None):
    tss = TimeSeriesSplit(n_splits=n_splits)
    ann_scores = []
    naive_scores = []
    fold_preds = []
    fold_trues = []
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # scale
        sc = StandardScaler() if scaler is None else scaler
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
        model = build_ann(X_train_s.shape[1], **(ann_kwargs or {}))
        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train_s, y_train.values, validation_split=0.1, epochs=fit_kwargs.get("epochs",50),
                  batch_size=fit_kwargs.get("batch_size",32), callbacks=[es], verbose=0)
        y_pred = model.predict(X_test_s).ravel()
        # naive baseline
        df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        naive = naive_one_step(df_test, target_col=y_test.name)
        # align: drop first if naive is NaN
        mask = ~naive.isna()
        y_test_al = y_test.reset_index(drop=True)[mask.values]
        y_pred_al = y_pred[mask.values]
        naive_al = naive[mask].values
        ann_scores.append(evaluate_preds(y_test_al.values, y_pred_al))
        naive_scores.append(evaluate_preds(y_test_al.values, naive_al))
        fold_preds.append(y_pred_al)
        fold_trues.append(y_test_al.values)
    return ann_scores, naive_scores, fold_preds, fold_trues

def permutation_test_ann(X, y, n_perm=30, ann_kwargs=None, fit_kwargs=None, seed=0):
    rng = np.random.default_rng(seed)
    perm_maes = []
    # train on true once to get baseline
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    model_true = build_ann(Xs.shape[1], **(ann_kwargs or {}))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model_true.fit(Xs, y.values, validation_split=0.1, epochs=fit_kwargs.get("epochs",50),
                   batch_size=fit_kwargs.get("batch_size",32), callbacks=[es], verbose=0)
    preds_true = model_true.predict(Xs).ravel()
    true_mae = mean_absolute_error(y.values, preds_true)
    for i in range(n_perm):
        y_perm = shuffle(y.values, random_state=int(seed+i))
        model = build_ann(Xs.shape[1], **(ann_kwargs or {}))
        model.fit(Xs, y_perm, validation_split=0.1, epochs=fit_kwargs.get("epochs",50),
                  batch_size=fit_kwargs.get("batch_size",32), callbacks=[es], verbose=0)
        preds = model.predict(Xs).ravel()
        perm_maes.append(mean_absolute_error(y.values, preds))
    perm_maes = np.array(perm_maes)
    p_emp = (np.sum(perm_maes <= true_mae) + 1) / (len(perm_maes) + 1)
    return {"true_mae": float(true_mae), "perm_maes": perm_maes.tolist(), "p_empirical": float(p_emp)}

def train_linear_baseline(X_train, y_train, X_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr.predict(X_test)

def train_ridge_baseline(X_train, y_train, X_test, alpha=1.0):
    r = Ridge(alpha=alpha)
    r.fit(X_train, y_train)
    return r.predict(X_test)

# ----------------------------
# Residual diagnostics
# ----------------------------
def residual_diagnostics(y_true, y_pred, outdir, prefix="resid"):
    resid = y_true - y_pred
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))
    sns.histplot(resid, kde=True)
    plt.title("Residual distribution")
    plt.savefig(Path(outdir)/f"{prefix}_hist.png")
    plt.close()

    plt.figure(figsize=(10,4))
    pd.Series(resid).plot()
    plt.title("Residuals over time")
    plt.savefig(Path(outdir)/f"{prefix}_timeseries.png")
    plt.close()

    # ACF
    try:
        fig = plt.figure(figsize=(10,4))
        sm.graphics.tsa.plot_acf(resid, lags=40, ax=fig.add_subplot(111))
        plt.title("Residual ACF")
        plt.savefig(Path(outdir)/f"{prefix}_acf.png")
        plt.close()
    except Exception:
        pass

    if STATS_AVAILABLE:
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        Path(outdir)/f"{prefix}_ljungbox.json"
        (Path(outdir)/f"{prefix}_ljungbox.json").write_text(lb.to_json())
        return {"ljungbox": lb.to_dict()}
    return {}

# ----------------------------
# Main CLI
# ----------------------------
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--time-col", default="Date")
    parser.add_argument("--target-col", default="Actual_Close")
    parser.add_argument("--n-perm", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--outdir", default="experiments_out")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[128,64])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data
    df_train, X_train, y_train, feat_cols = load_and_prepare(args.train_csv, time_col=args.time_col, target_col=args.target_col)
    df_test, X_test, y_test, _ = load_and_prepare(args.test_csv, time_col=args.time_col, target_col=args.target_col, feature_cols=feat_cols)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Naive baseline on test
    naive = naive_one_step(pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1), target_col=args.target_col)
    mask = ~naive.isna()
    y_test_al = y_test.reset_index(drop=True)[mask.values].values
    naive_al = naive[mask].values

    # Train ANN on full training set and evaluate on test
    ann_kwargs = {"hidden_layers": tuple(args.hidden_layers), "lr": args.lr, "dropout": args.dropout}
    fit_kwargs = {"epochs": args.epochs, "batch_size": args.batch_size}
    model = build_ann(X_train_s.shape[1], **ann_kwargs)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    logging.info("Training ANN on full training set...")
    model.fit(X_train_s, y_train.values, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, callbacks=[es], verbose=1)
    preds_test = model.predict(X_test_s).ravel()

    # Align preds with naive mask
    preds_test_al = preds_test[mask.values]

    # Evaluate
    ann_metrics = evaluate_preds(y_test_al, preds_test_al)
    naive_metrics = evaluate_preds(y_test_al, naive_al)
    linear_preds = train_linear_baseline(X_train, y_train, X_test)
    linear_preds_al = linear_preds[mask.values]
    linear_metrics = evaluate_preds(y_test_al, linear_preds_al)
    ridge_preds = train_ridge_baseline(X_train, y_train, X_test, alpha=1.0)
    ridge_metrics = evaluate_preds(y_test_al, ridge_preds[mask.values])

    # ARIMA baseline per series fallback (if single series)
    try:
        arima_preds = []
        for i in range(len(y_test)):
            # naive ARIMA using history from train + prior test values
            hist = pd.concat([y_train, y_test.iloc[:i]]).reset_index(drop=True)
            arima_preds.append(arima_forecast(hist, steps=1))
        arima_preds = np.array(arima_preds)
        arima_al = arima_preds[mask.values]
        arima_metrics = evaluate_preds(y_test_al, arima_al)
    except Exception:
        arima_metrics = None

    # Ensemble (simple average of ANN + linear + naive)
    ensemble = (preds_test + linear_preds + np.concatenate([np.array([np.nan]), y_test.values[:-1]])) / 3.0
    ensemble_al = ensemble[mask.values]
    ensemble_metrics = evaluate_preds(y_test_al, ensemble_al)

    # TimeSeries CV
    logging.info("Running TimeSeriesSplit CV for ANN vs naive...")
    ann_cv, naive_cv, fold_preds, fold_trues = time_series_cv_ann(X_train, y_train, n_splits=args.cv_folds,
                                                                  ann_kwargs=ann_kwargs, fit_kwargs=fit_kwargs)
    # summarize CV
    def summarize_cv(scores):
        return {"mae_mean": float(np.mean([s["mae"] for s in scores])),
                "mae_std": float(np.std([s["mae"] for s in scores]))}
    ann_cv_summary = summarize_cv(ann_cv)
    naive_cv_summary = summarize_cv(naive_cv)

    # Permutation test (on training set)
    logging.info("Running permutation test (this will retrain ANN multiple times)...")
    perm_res = permutation_test_ann(X_train, y_train, n_perm=args.n_perm, ann_kwargs=ann_kwargs, fit_kwargs=fit_kwargs, seed=args.seed)

    # Residual diagnostics on test set
    logging.info("Running residual diagnostics...")
    resid_info = residual_diagnostics(y_test_al, preds_test_al, outdir=outdir/"residuals", prefix="test")

    # Feature ablation: OHLC only vs 5th column only vs all
    logging.info("Running feature ablation...")
    # assume first 4 are OHLC and 5th is special
    if len(feat_cols) >= 5:
        ohlc_cols = feat_cols[:4]
        fifth = [feat_cols[4]]
    else:
        ohlc_cols = feat_cols
        fifth = feat_cols[:1]
    ablation_results = {}
    for name, cols in [("OHLC", ohlc_cols), ("FIFTH", fifth), ("ALL", feat_cols)]:
        _, Xtr, ytr, _ = load_and_prepare(args.train_csv, time_col=args.time_col, target_col=args.target_col, feature_cols=cols)
        _, Xte, yte, _ = load_and_prepare(args.test_csv, time_col=args.time_col, target_col=args.target_col, feature_cols=cols)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        m = build_ann(Xtr_s.shape[1], **ann_kwargs)
        m.fit(Xtr_s, ytr.values, validation_split=0.1, epochs=10, batch_size=args.batch_size, verbose=0)
        p = m.predict(Xte_s).ravel()
        # align naive
        df_te = pd.concat([Xte.reset_index(drop=True), yte.reset_index(drop=True)], axis=1)
        naive_te = naive_one_step(df_te, target_col=args.target_col)
        mask_te = ~naive_te.isna()
        ablation_results[name] = {
            "ann": evaluate_preds(yte.reset_index(drop=True)[mask_te.values].values, p[mask_te.values]),
            "naive": evaluate_preds(yte.reset_index(drop=True)[mask_te.values].values, naive_te[mask_te.values].values)
        }

    # Save results
    results = {
        "timestamp": now_iso(),
        "ann_test_metrics": ann_metrics,
        "naive_test_metrics": naive_metrics,
        "linear_test_metrics": linear_metrics,
        "ridge_test_metrics": ridge_metrics,
        "arima_test_metrics": arima_metrics,
        "ensemble_test_metrics": ensemble_metrics,
        "cv_ann_summary": ann_cv_summary,
        "cv_naive_summary": naive_cv_summary,
        "permutation_test": perm_res,
        "ablation": ablation_results,
        "residuals": resid_info,
        "n_test_samples": int(len(y_test_al))
    }
    (outdir/"results.json").write_text(json.dumps(results, indent=2))
    logging.info("Saved results to %s", outdir/"results.json")

    # Print concise summary
    print("=== Quick summary ===")
    print(f"Test samples (aligned): {len(y_test_al)}")
    print(f"ANN MAE: {ann_metrics['mae']:.6f}  Naive MAE: {naive_metrics['mae']:.6f}")
    print(f"Linear MAE: {linear_metrics['mae']:.6f}  Ridge MAE: {ridge_metrics['mae']:.6f}")
    if arima_metrics:
        print(f"ARIMA MAE: {arima_metrics['mae']:.6f}")
    print(f"Ensemble MAE: {ensemble_metrics['mae']:.6f}")
    print(f"CV ANN MAE mean: {ann_cv_summary['mae_mean']:.6f}  CV Naive MAE mean: {naive_cv_summary['mae_mean']:.6f}")
    print(f"Permutation test true_mae: {perm_res['true_mae']:.6f}  empirical p: {perm_res['p_empirical']:.4f}")
    print("Ablation (ANN vs naive):")
    for k,v in ablation_results.items():
        print(f"  {k}: ANN MAE {v['ann']['mae']:.4f}  Naive MAE {v['naive']['mae']:.4f}")

if __name__ == "__main__":
    main(sys.argv[1:])
