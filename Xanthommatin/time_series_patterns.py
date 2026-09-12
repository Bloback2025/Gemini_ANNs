#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: time_series_patterns.py
Date of Creation: September 10, 2026
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\Xanthommatin\time_series_patterns.py

Annotations:
- Isolated core architectural patterns extracted from the canonical sf_infer_ann_gem.py pipeline[cite: 5].
- Designed as a reusable reference module and adapter library for future domain-specific or generalized pipelines.
- Pattern 1: Defensive Header Hygiene (preemptive whitespace stripping to prevent silent KeyErrors)[cite: 5].
- Pattern 2: Explicit Scaler Handshaking (sidecar JSON parameter mapping for deterministic normalization)[cite: 5].
- Pattern 3: Temporal Alignment Safeguard (t -> t+1 chronological slicing to completely eliminate lookahead and lag bias)[cite: 5].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# PATTERN 1: Defensive Header Hygiene
# -------------------------------------------------------------------------
def defensive_header_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preemptively strips hidden leading or trailing whitespace from all DataFrame column headers.
    
    Why it matters:
    Raw CSV file exports frequently inject invisible whitespace into headers (e.g., "Close " instead 
    of "Close"), which causes silent, infuriating `KeyError` crashes downstream. This preflight guard 
    normalizes headers instantly upon ingestion.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


# -------------------------------------------------------------------------
# PATTERN 2: Explicit Scaler Handshaking
# -------------------------------------------------------------------------
def load_and_validate_scaler(scaler_path: str | Path) -> Dict[str, Any]:
    """
    Explicitly loads sidecar JSON normalization parameters (mu, sigma, target bounds) 
    rather than relying on black-box preprocessing states.

    Why it matters:
    Ensures absolute mathematical parity between training normalization space and live inference 
    space[cite: 5]. Pulling parameters explicitly allows external auditing and manifest verification.
    """
    sp = Path(scaler_path)
    if not sp.exists():
        raise FileNotFoundError(f"Critical: Scaler parameter sidecar missing at {sp}")

    with sp.open("r", encoding="utf-8-sig") as f:
        scaler = json.load(f)

    required_keys = ["feature_cols", "mu", "sigma", "target_mu", "target_sigma"]
    for k in required_keys:
        if k not in scaler:
            raise RuntimeError(f"Scaler configuration sidecar is missing mandatory key: '{k}'")

    return scaler


def apply_vectorized_scaling(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Applies vectorized Z-score standardization using training parameters with a numerical epsilon 
    to prevent division-by-zero errors[cite: 5].
    """
    return (X - mu) / (sigma + 1e-12)


# -------------------------------------------------------------------------
# PATTERN 3: Temporal Alignment Safeguard (t -> t+1)
# -------------------------------------------------------------------------
def align_temporal_predictions(
    df: pd.DataFrame, 
    preds_denormalized: np.ndarray, 
    date_col: str = "Date", 
    target_col: str = "Close"
) -> pd.DataFrame:
    """
    Enforces strict temporal alignment to map features observed at index t to target outcomes at t+1.

    Why it matters:
    In sequential time-series modeling, failing to offset predictions results in lookahead and lag bias, 
    creating an illusion of predictive perfection. 
    
    The mechanism:
    Slicing `preds[:-1]` and mapping them against actual values from `iloc[1:]`[cite: 5] 
    mathematically guarantees that row index `i` evaluates what the model *foresaw* for the next step, 
    maintaining honest chronological validation.
    """
    if date_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"DataFrame must contain specified date ('{date_col}') and target ('{target_col}') columns.")

    # Ensure chronological sorting before slicing
    df_sorted = df.sort_values(date_col, ascending=True).reset_index(drop=True)

    aligned_df = pd.DataFrame({
        date_col: df_sorted[date_col].iloc[1:].values,
        f"Actual_{target_col}": df_sorted[target_col].iloc[1:].values,
        f"Pred_{target_col}": preds_denormalized[:-1]
    })

    return aligned_df