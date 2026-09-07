#!/usr/bin/env python3
"""
Date of Creation: September 6, 2026, 8:42 PM
Path: C:/Users/loweb/AI_Financial_Sims/Gemini_ANNs/ablation_simulate_gem.py

Annotations:
- Modernized Feature Ablation Simulation Suite (Gemini version).
- Sequentially neutralizes each input feature to its training mean (mu).
- Measures MAE degradation to isolate feature importance (e.g., PrevClose).
- Adheres to the _gem project file naming convention.
"""

import json
import os
import tempfile
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def denorm_array(preds_arr, scaler):
    # Inference script already denormalizes outputs; return as-is to prevent double-scaling
    if not isinstance(preds_arr, np.ndarray):
        preds_arr = np.asarray(preds_arr, dtype=float)
    return preds_arr

def main():
    print(f"[{datetime.utcnow().isoformat()}Z] Initializing Ablation Simulation Suite...")
    
    run_dir = Path("artifacts/ho_ann_run_retest1")
    scaler_path = run_dir / "scaler.json"
    model_path = run_dir / "model_final.keras"
    test_csv_path = Path("hoxnc_testing_retest1.csv")
    inference_script = "ho_infer_ann_retest1_gem.py"
    
    if not scaler_path.exists() or not model_path.exists():
        print(f"Warning: Artifacts not found at {run_dir.resolve()}. Please verify paths.")
        return

    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    features = scaler.get("feature_cols", [])
    mus = scaler.get("mu", [])
    
    results = {}
    if not test_csv_path.exists():
        print(f"Error: Test CSV not found at {test_csv_path.resolve()}")
        return

    for i, f in enumerate(features):
        print(f"\n--- Neutralizing feature: {f} (mu = {mus[i]}) ---")
        df = pd.read_csv(test_csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        
        # Engineer PrevClose if it's part of the feature set but missing from the raw CSV
        if "PrevClose" not in df.columns and "Close" in df.columns:
            df["PrevClose"] = df["Close"].shift(1)
        
        col = next((c for c in df.columns if c.lower() == f.lower()), None)
        if col is None:
            results[f] = {"error": "column_not_found"}
            continue
            
        df[col] = float(mus[i])
        
        tmp_test = tempfile.mktemp(suffix=".csv")
        df.to_csv(tmp_test, index=False)
        
        outdir = run_dir / f"ablation_sim_{f}"
        outdir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", inference_script,
            "--input_csv", tmp_test,
            "--model", str(model_path),
            "--scaler", str(scaler_path),
            "--out_csv", str(outdir / "preds.json")
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results[f] = {"error": "inference_failed", "stderr": proc.stderr}
            try:
                os.remove(tmp_test)
            except Exception:
                pass
            continue
            
        try:
            preds_data = json.loads((outdir / "preds.json").read_text(encoding="utf-8"))
            preds = np.array(preds_data.get("preds", []), dtype=float)
            denorm_preds = denorm_array(preds, scaler)
            
            y = df["Close"].values[1:len(denorm_preds) + 1]
            mae = float(np.mean(np.abs(denorm_preds[:len(y)] - y)))
            results[f] = {"ablation_mae": mae}
            print(f"Result for dropping {f}: MAE = {mae:.4f}")
        except Exception as e:
            results[f] = {"error": str(e)}
            
        try:
            os.remove(tmp_test)
        except Exception:
            pass

    out_summary = run_dir / "ablation_simulation_results.json"
    out_summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAblation simulation complete. Saved to: {out_summary.resolve()}")

if __name__ == "__main__":
    main()