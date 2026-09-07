#!/usr/bin/env python3
"""
hardened_eval.py
Robust end-to-end runner + evaluator for ANN vs naive one-step-lag baseline.
"""

import sys
import os
import json
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional imports
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ----------------------------
# Helpers
# ----------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def run_subprocess(cmd, timeout, log_path):
    logging.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    (log_path.parent).mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}\n")
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}). See {log_path}")
    return res

def bootstrap_pvalue(diffs, n_boot=10000, seed=0):
    """Two-sided bootstrap test for mean(diffs) != 0 where diffs = abs_err_naive - abs_err_model.
       Returns p-value."""
    rng = np.random.default_rng(seed)
    observed = np.mean(diffs)
    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    # two-sided
    p = np.mean(np.abs(boot_means) >= np.abs(observed))
    return float(p)

# ----------------------------
# Main
# ----------------------------
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", required=True)
    parser.add_argument("--infer-script", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scaler-path", required=True)
    parser.add_argument("--preds-path", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-improvement", type=float, default=0.0,
                        help="Require ANN MAE to be at least this fraction lower than naive (e.g. 0.01 = 1%)")
    parser.add_argument("--fail-on-no-improve", action="store_true",
                        help="Exit with nonzero code if ANN does not beat naive baseline by min_improvement")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args(argv)

    # Setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PY = sys.executable
    artifacts_dir = Path(args.model_path).parent
    outdir = Path(args.outdir) if args.outdir else artifacts_dir / "inference_rebuild"
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": now_iso(),
        "train_script": str(Path(args.train_script).resolve()),
        "infer_script": str(Path(args.infer_script).resolve()),
        "test_csv": str(Path(args.test_csv).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "scaler_path": str(Path(args.scaler_path).resolve()),
        "preds_path": str(Path(args.preds_path).resolve()),
        "python": PY,
        "seed": 0
    }

    # 1) TRAIN
    try:
        train_log = logs_dir / "train.log"
        run_subprocess([PY, args.train_script], timeout=args.timeout, log_path=train_log)
        manifest["train_succeeded"] = True
    except Exception as e:
        manifest["train_succeeded"] = False
        manifest["train_error"] = str(e)
        Path(outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logging.exception("Training failed.")
        raise

    # 2) INFERENCE
    try:
        infer_log = logs_dir / "infer.log"
        infer_cmd = [
            PY, args.infer_script,
            "--test_csv", args.test_csv,
            "--model", args.model_path,
            "--scaler", args.scaler_path,
            "--outdir", str(outdir)
        ]
        run_subprocess(infer_cmd, timeout=args.timeout, log_path=infer_log)
        manifest["infer_succeeded"] = True
    except Exception as e:
        manifest["infer_succeeded"] = False
        manifest["infer_error"] = str(e)
        Path(outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logging.exception("Inference failed.")
        raise

    # 3) LOAD AND VALIDATE PREDICTIONS
    preds_path = Path(args.preds_path)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    try:
        df = pd.read_json(preds_path)
    except ValueError as e:
        raise ValueError(f"Could not parse JSON predictions at {preds_path}: {e}")

    required_cols = {"Pred_Close", "Actual_Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Predictions JSON missing required columns: {required_cols - set(df.columns)}")

    # If there is a timestamp/date column, prefer it for ordering
    time_cols = [c for c in ("Timestamp", "timestamp", "Date", "date", "Datetime", "datetime") if c in df.columns]
    if time_cols:
        time_col = time_cols[0]
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)
            manifest["time_column_used"] = time_col
        except Exception:
            logging.warning("Found time column but could not parse it; proceeding without sorting.")
    else:
        logging.info("No timestamp column found; assuming rows are already time-ordered.")

    # Ensure numeric
    for col in ("Pred_Close", "Actual_Close"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing actual or pred
    before = len(df)
    df = df.dropna(subset=["Pred_Close", "Actual_Close"]).reset_index(drop=True)
    after = len(df)
    manifest["rows_before_dropna"] = int(before)
    manifest["rows_after_dropna"] = int(after)

    if len(df) < 10:
        raise ValueError("Too few valid prediction rows to evaluate reliably.")

    # 4) BUILD NAIVE BASELINE (one-step lag of Actual_Close)
    df = df.copy()
    df["Naive"] = df["Actual_Close"].shift(1)
    # If there is a grouping key (e.g., symbol), we should shift per group. Try to detect 'Symbol' or 'Ticker'
    group_cols = [c for c in ("Symbol", "symbol", "Ticker", "ticker") if c in df.columns]
    if group_cols:
        grp = group_cols[0]
        df = df.sort_values([grp] + ([manifest.get("time_column_used")] if manifest.get("time_column_used") else [])).reset_index(drop=True)
        df["Naive"] = df.groupby(grp)["Actual_Close"].shift(1)
        manifest["naive_grouped_by"] = grp

    df = df.dropna(subset=["Naive"]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Too few rows after constructing naive baseline (check ordering/grouping).")

    # 5) METRICS
    y_true = df["Actual_Close"].to_numpy(dtype=float)
    y_pred = df["Pred_Close"].to_numpy(dtype=float)
    y_naive = df["Naive"].to_numpy(dtype=float)

    ann_mae = float(mean_absolute_error(y_true, y_pred))
    naive_mae = float(mean_absolute_error(y_true, y_naive))
    ann_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, y_naive)))
    # MAPE: avoid division by zero
    nonzero_mask = np.abs(y_true) > 1e-12
    if nonzero_mask.sum() == 0:
        mape_ann = None
        mape_naive = None
    else:
        mape_ann = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
        mape_naive = float(np.mean(np.abs((y_true[nonzero_mask] - y_naive[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)

    # 6) STATISTICAL TEST on absolute errors
    abs_err_model = np.abs(y_true - y_pred)
    abs_err_naive = np.abs(y_true - y_naive)
    diffs = abs_err_naive - abs_err_model  # positive => model better

    if SCIPY_AVAILABLE:
        tstat, pval = stats.ttest_rel(abs_err_naive, abs_err_model, nan_policy="omit")
        test_name = "paired_ttest"
        pval = float(pval)
    else:
        pval = bootstrap_pvalue(diffs, n_boot=args.bootstrap_iterations, seed=manifest["seed"])
        test_name = f"bootstrap_{args.bootstrap_iterations}_iters"

    # 7) Save metrics and manifest
    metrics = {
        "ann_mae": ann_mae,
        "naive_mae": naive_mae,
        "ann_rmse": ann_rmse,
        "naive_rmse": naive_rmse,
        "mape_ann": mape_ann,
        "mape_naive": mape_naive,
        "n_samples": int(len(df)),
        "test_name": test_name,
        "p_value": pval,
        "alpha": args.alpha,
        "mean_diff_abs_err": float(np.mean(diffs)),
        "median_diff_abs_err": float(np.median(diffs)),
        "rows": {
            "before": manifest["rows_before_dropna"],
            "after": manifest["rows_after_dropna"],
            "final": int(len(df))
        },
        "timestamp": now_iso()
    }

    manifest.update(metrics)
    Path(outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    Path(outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 8) Print summary
    print("=== Evaluation summary ===")
    print(f"Samples: {metrics['n_samples']}")
    print(f"ANN MAE: {ann_mae:.6f}")
    print(f"Naive MAE: {naive_mae:.6f}")
    print(f"ANN RMSE: {ann_rmse:.6f}")
    print(f"Naive RMSE: {naive_rmse:.6f}")
    if mape_ann is not None:
        print(f"ANN MAPE: {mape_ann:.3f}%")
        print(f"Naive MAPE: {mape_naive:.3f}%")
    print(f"Stat test: {test_name}, p-value = {pval:.6g} (alpha={args.alpha})")
    print(f"Mean(abs_err_naive - abs_err_model) = {metrics['mean_diff_abs_err']:.6g}")

    # 9) Decide pass/fail
    rel_improvement = (naive_mae - ann_mae) / naive_mae if naive_mae != 0 else float("inf")
    passed = (ann_mae < naive_mae) and (rel_improvement >= args.min_improvement) and (pval < args.alpha)

    print(f"Relative improvement (naive->ANN): {rel_improvement:.4%}")
    print("PASS" if passed else "FAIL")

    # Optionally fail CI
    if args.fail_on_no_improve and not passed:
        logging.error("Model did not pass the improvement/significance criteria.")
        sys.exit(2)

    # Otherwise exit 0
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]) or 0)
    except Exception as e:
        logging.exception("Evaluation failed.")
        sys.exit(1)
