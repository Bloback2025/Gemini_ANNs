"""
Date of Creation: September 6, 2026, 7:15 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\apply_preprocessing_and_validate_gem.py
Original Legacy Source: C:\Users\loweb\AI_Financial_Sims\HO\HO_train_phase_ANNs\apply_preprocessing_and_validate.py

Annotations:
- This is a clean rewrite of the preprocessing validation and test evaluation script, adapted for the Gemini_ANNs project root directory structure.
- Automatically loads trained Keras models, audits input shapes, imports preprocessing steps dynamically, applies scaling adjustments from configuration files, and generates evaluation or prediction reports.
"""

import json, sys, traceback
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow import keras

ROOT = Path.cwd()
ART = ROOT / "artifacts" / "ho_ann_run"
MODEL_PATH = ART / "model_final.keras"
SCALER_PATH = ART / "scaler.json"
RUNLOG_PATH = ART / "RUNLOG.json"
TEST_CSV = ROOT.parent / "data" / "test.csv"
OUT_REPORT = ART / "validation_report.json"
OUT_PROCESSED = ART / "processed_test.csv"
OUT_PREDS = ART / "preds_main.csv"

report = {"actions": [], "errors": []}
try:
    report["actions"].append(f"cwd: {ROOT}")
    report["actions"].append(f"loading model: {MODEL_PATH}")
    model = keras.models.load_model(str(MODEL_PATH))
    report["model_input_shape"] = model.input_shape

    # load test csv
    report["actions"].append(f"reading test csv: {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test csv not found: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV, parse_dates=["Date"], dayfirst=False)
    report["test_csv_path"] = str(TEST_CSV)
    report["test_rows"] = int(df.shape[0])
    report["original_test_columns"] = list(df.columns)

    # attempt to import preprocess from training script if available
    train_script = ROOT / "ho_train_ann_gem.py"
    preprocess_fn = None
    if train_script.exists():
        try:
            import importlib.util, types
            spec = importlib.util.spec_from_file_location("ho_train_ann_mod", str(train_script))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            preprocess_fn = getattr(mod, "preprocess", None)
            if callable(preprocess_fn):
                report["actions"].append("found callable preprocess(df) in training script")
        except Exception as e:
            report["actions"].append("could not import training script: " + str(e))

    # If no preprocess function, heuristically extract features: drop Date/Target
    if preprocess_fn is None:
        report["actions"].append("no preprocess function; using candidate columns (drop Date/Target)")
        X = df.drop(columns=["Date", "Target"], errors="ignore").copy()
        feature_names = list(X.columns)
        report["actions"].append(f"candidate features: {feature_names}")
    else:
        report["actions"].append("running preprocess(df) from training script")
        X = preprocess_fn(df.copy())
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            X = pd.DataFrame(X)
            feature_names = list(X.columns)

    # ensure correct feature count
    expected = model.input_shape[1] if model.input_shape and len(model.input_shape) > 1 else None
    report["expected_feature_count"] = int(expected) if expected is not None else None
    report["processed_feature_names"] = feature_names
    report["processed_feature_count"] = len(feature_names)

    if expected is not None and len(feature_names) < expected:
        missing = expected - len(feature_names)
        dummy_names = [f"dummy_{i+1}" for i in range(missing)]
        for n in dummy_names:
            X[n] = 0
        feature_names += dummy_names
        report["actions"].append(f"added dummy columns: {dummy_names}")
        report["processed_feature_names"] = feature_names
        report["processed_feature_count"] = len(feature_names)

    # apply scaler if usable
    scaler_applied = False
    if SCALER_PATH.exists():
        try:
            s = json.loads(SCALER_PATH.read_text())
            mean = s.get("mean") or s.get("mu")
            scale = s.get("scale") or s.get("sigma")
            if isinstance(mean, list) and isinstance(scale, list) and len(mean) == len(feature_names) and len(scale) == len(feature_names):
                arr = X.values.astype(float)
                arr = (arr - np.array(mean)) / np.array(scale)
                X = pd.DataFrame(arr, columns=feature_names)
                scaler_applied = True
                report["actions"].append("applied scaler configuration mean/scale")
            else:
                report["actions"].append("scaler file present but fields not usable or length mismatch")
        except Exception as e:
            report["actions"].append("error reading scaler file: " + str(e))
    else:
        report["actions"].append("scaler file not found")

    report["scaler_applied"] = scaler_applied
    # save processed features
    X.to_csv(OUT_PROCESSED, index=False)
    report["processed_test_csv"] = str(OUT_PROCESSED)

    # run predict or evaluate
    if "Target" in df.columns:
        y = df["Target"].values
        loss, mae = model.evaluate(X.values, y, verbose=0)
        report["eval_loss"] = float(loss)
        report["eval_mae"] = float(mae)
        report["actions"].append("ran model.evaluate")
    else:
        preds = model.predict(X.values)
        pd.DataFrame(preds, columns=["pred"]).to_csv(OUT_PREDS, index=False)
        report["prediction_file"] = str(OUT_PREDS)
        report["prediction_shape"] = list(preds.shape)
        report["actions"].append("ran model.predict and wrote preds_main.csv")

    # write report
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print("Wrote report to", OUT_REPORT)
except Exception as e:
    tb = traceback.format_exc()
    report["errors"].append(str(e))
    report["traceback"] = tb
    try:
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        OUT_REPORT.write_text(json.dumps(report, indent=2))
        print("Error occurred; wrote partial report to", OUT_REPORT)
    except Exception as e2:
        print("Error and failed to write report:", e2)
    sys.exit(1)