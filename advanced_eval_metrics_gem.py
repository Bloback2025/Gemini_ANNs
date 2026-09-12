#!/usr/bin/env python3
r"""
Date of Creation: September 6, 2026, 8:02 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\advanced_eval_metrics_gem.py

Annotations:
- Advanced Robustness & Directional Metrics Suite (Gemini version).
- Computes directional hit rates, Spearman rank correlation (IC), and turning point precision/recall.
- Includes timestamp tracking, file path/name logging, and automated analytical commentary.
- Adheres to the _gem project file naming convention.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime

def main():
    # Define explicit file path and file name for traceability
    target_dir = Path("artifacts/ho_ann_run_retest1/inference_rebuild")
    file_name = "preds_model_retest1.json"
    preds_path = target_dir / file_name

    if not preds_path.exists():
        print(f"Error: Predictions file not found at {preds_path.resolve()}")
        print("Please ensure you have run the inference pipeline first.")
        return

    print(f"[{datetime.utcnow().isoformat()}Z] Loading predictions...")
    print(f"  - Script Name: advanced_eval_metrics_gem.py")
    print(f"  - Target File Name: {file_name}")
    print(f"  - Target File Path: {preds_path.resolve()}")
    
    df = pd.read_json(preds_path)
    
    # Ensure chronological sorting
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Compute actual vs predicted changes (Delta from previous lag)
    df["Actual_Close_Lag"] = df["Actual_Close"].shift(1)
    df["Actual_Change"] = df["Actual_Close"] - df["Actual_Close_Lag"]
    df["Pred_Change"] = df["Pred_Close"] - df["Actual_Close_Lag"]
    
    valid = df.dropna(subset=["Actual_Change", "Pred_Change"]).copy()
    
    # 1. Directional Hit Rate (Sign Accuracy)
    hit_rate = np.mean(np.sign(valid["Actual_Change"]) == np.sign(valid["Pred_Change"]))
    
    # 2. Spearman Rank Information Coefficient (IC)
    ic, p_val = stats.spearmanr(valid["Pred_Close"], valid["Actual_Close"])
    
    # 3. Turning Point (Inflection) Detection Analysis
    valid["Actual_Sign"] = np.sign(valid["Actual_Change"])
    valid["Pred_Sign"] = np.sign(valid["Pred_Change"])
    
    # Turning point defined as a sign reversal from previous period
    valid["Actual_TP"] = (valid["Actual_Sign"] != valid["Actual_Sign"].shift(1)).astype(int)
    valid["Pred_TP"] = (valid["Pred_Sign"] != valid["Pred_Sign"].shift(1)).astype(int)
    
    tp_match = np.sum((valid["Actual_TP"] == 1) & (valid["Pred_TP"] == 1))
    tp_total_actual = np.sum(valid["Actual_TP"] == 1)
    tp_total_pred = np.sum(valid["Pred_TP"] == 1)
    
    precision = tp_match / tp_total_pred if tp_total_pred > 0 else 0
    recall = tp_match / tp_total_actual if tp_total_actual > 0 else 0
    
    # Generate analytical commentary based on empirical results
    commentary = []
    if hit_rate > 0.50:
        commentary.append(f"Directional Hit Rate ({hit_rate:.2%}) demonstrates a positive directional edge over random chance (50%).")
    else:
        commentary.append(f"Directional Hit Rate ({hit_rate:.2%}) indicates standard directional coin-flip performance on this slice.")
        
    if ic > 0.5:
        commentary.append(f"Spearman Rank IC ({ic:.4f}, p={p_val:.4g}) reveals a strong macro-structural rank ordering maintained by the ANN.")
        
    if precision > 0.5:
        commentary.append(f"Turning Point Precision ({precision:.2%}) confirms that more than half of the model's signaled reversals coincide with real market inflections.")
    else:
        commentary.append(f"Turning Point Precision ({precision:.2%}) highlights high false-positive rates during choppy sideways regimes.")

    # Compile final results package including explicit source path details
    results = {
        "script_name": "advanced_eval_metrics_gem.py",
        "source_file_name": file_name,
        "source_file_path": str(preds_path.resolve()),
        "execution_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "sample_count": int(len(valid)),
        "metrics": {
            "directional_hit_rate": float(hit_rate),
            "information_coefficient_spearman": float(ic),
            "ic_p_value": float(p_val),
            "turning_point_precision": float(precision),
            "turning_point_recall": float(recall)
        },
        "analytical_commentary": commentary
    }
    
    # Save output report
    out_path = preds_path.parent / "advanced_metrics_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    
    # Print formatted output to console
    print("\n" + "="*60)
    print(" ADVANCED ROBUSTNESS & DIRECTIONAL REPORT")
    print("="*60)
    print(f"Script Name: advanced_eval_metrics_gem.py")
    print(f"File Name: {file_name}")
    print(f"File Path: {preds_path.resolve()}")
    print(f"Timestamp: {results['execution_timestamp_utc']}")
    print(f"Evaluated Samples: {results['sample_count']}")
    print("\nQuantitative Metrics:")
    for k, v in results["metrics"].items():
        print(f"  - {k}: {v:.4f}")
    print("\nAnalytical Commentary:")
    for comment in results["analytical_commentary"]:
        print(f"  * {comment}")
    print("="*60)
    print(f"Saved complete report to: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()