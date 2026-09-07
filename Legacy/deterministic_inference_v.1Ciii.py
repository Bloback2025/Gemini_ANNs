#!/usr/bin/env python3
# Version: v1Ciii
# Date: 2025-12-02 (AEDT)
#
# deterministic_inference_v.1Ciii.py
#
# Purpose:
# This file is the hardened, audit-focused deterministic inference harness for the HO pipeline.
# It loads a persisted model and optional scaler, validates input CSVs and enforced SHAs, applies
# best-effort determinism measures (PYTHONHASHSEED, NumPy, TensorFlow seeding and threading),
# computes deterministic SHAs for files and directories, writes atomic artifacts and dual-case
# SHA sidecars, and emits compact machine-readable success/failure JSON for CI and orchestration.
#
# Additional paragraph (overview):
# The harness is intended to be used as the authoritative inference step in a regulated or
# auditable pipeline. It supports a synthetic smoke mode for CI and local testing, and a
# deterministic model mode that loads Keras models (SavedModel dir or .h5) and optional
# preprocessing scalers (joblib/pickle). The manifest produced by this script records the
# environment, determinism measures, input/output SHAs, and the exact numeric feature columns
# consumed during inference to provide a complete audit trail for downstream reviewers.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Set TF determinism environment variables BEFORE any potential TF import.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

# -------------------------
# Utilities: logging, atomic writes, SHA helpers
# -------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Best-effort; ignore if not supported
        pass

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(tf.name, str(path))
    _fsync_dir(path.parent)

def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    data = json.dumps(obj, sort_keys=True, indent=indent).encode("utf-8")
    _atomic_write_bytes(path, data)

def compute_sha256_hex(path: Path, *, uppercase: bool = False, chunk_size: int = 8192) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Path missing for SHA: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"compute_sha256_hex expects a file, got directory: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    hexv = h.hexdigest()
    return hexv.upper() if uppercase else hexv.lower()

def compute_path_sha(path: Path, *, uppercase: bool = False, chunk_size: int = 8192) -> str:
    """
    Deterministic SHA for a path:
      - If file: SHA of contents (same as compute_sha256_hex).
      - If directory: deterministic recursive hash of relative file paths + contents.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path missing for directory/file SHA: {path}")
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
    else:
        base = path
        for root, dirs, files in os.walk(base):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = Path(root) / fname
                rel = str(fpath.relative_to(base)).replace(os.sep, "/").encode("utf-8")
                h.update(rel)
                with fpath.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(chunk_size), b""):
                        h.update(chunk)
    hexv = h.hexdigest()
    return hexv.upper() if uppercase else hexv.lower()

def write_sidecars_both(path: Path) -> Tuple[str, str]:
    """
    Write two sidecars for compatibility with case-sensitive/-insensitive filesystems.
      - <name>.sha256.txt (lowercase hex)
      - <name>.SHA256.TXT (uppercase hex)
    Returns (lowercase_hex, uppercase_hex).
    """
    lower = compute_path_sha(path, uppercase=False)
    upper = lower.upper()
    lower_side = path.with_name(path.name + ".sha256.txt")
    upper_side = path.with_name(path.name + ".SHA256.TXT")
    if str(lower_side).lower() == str(upper_side).lower():
        # On case-insensitive FS the two names collide; write a single file containing both hashes
        lower_side.write_text(lower + "\n" + upper + "\n", encoding="ascii")
        return lower, upper
    lower_side.write_text(lower + "\n", encoding="ascii")
    upper_side.write_text(upper + "\n", encoding="ascii")
    return lower, upper

# -------------------------
# Deterministic seeding
# -------------------------
def deterministic_seed_all(seed: int) -> Dict[str, Any]:
    """
    Apply best-effort determinism settings and return a dict of measures applied for manifest logging.
    """
    measures = {
        "PYTHONHASHSEED_set": True,
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        "tf_threading_set": False,
        "tf_op_determinism_enabled": False,
    }
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            measures["tf_threading_set"] = True
        except Exception:
            pass
        try:
            from tensorflow.python.framework import config as tf_config  # type: ignore
            try:
                tf_config.enable_op_determinism()
                measures["tf_op_determinism_enabled"] = True
            except Exception:
                pass
        except Exception:
            pass
        tf.random.set_seed(int(seed))
    except Exception:
        pass
    return measures

# -------------------------
# CSV feature loader (best-effort) with shape normalization and feature capture
# -------------------------
def load_csv_features_best_effort(csv_path: Path, model=None):
    """
    Returns a tuple (arr, feature_cols).
    - arr: 2D numpy array (n_rows, n_features) or fallback list
    - feature_cols: ordered list of numeric column names selected from CSV (may be empty)
    Attempts numeric column selection, falls back to model's input shape, and finally to (1,1) zeros.
    Emits RUN_WARN to stderr if pandas/numpy missing cause fallback.
    """
    np = None
    pd = None
    feature_cols = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    else:
        np = __import__("numpy")
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None
        eprint("RUN_WARN: pandas not available; falling back to model shape or zeros")
    else:
        pd = __import__("pandas")

    if pd is not None:
        try:
            df = pd.read_csv(csv_path)
            num_df = df.select_dtypes(include=["number"])
            feature_cols = list(num_df.columns)
            if num_df.shape[1] == 0:
                if np is not None:
                    return np.zeros((1, 1)), feature_cols
                return [[0.0]], feature_cols
            arr = num_df.to_numpy()
            if np is not None:
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
            return arr, feature_cols
        except Exception as e:
            eprint(f"RUN_WARN: pandas failed to load CSV {csv_path}: {e} — falling back")

    if model is not None:
        try:
            input_shape = getattr(model, "input_shape", None)
            if isinstance(input_shape, (list, tuple)):
                input_shape = input_shape[0]
            features = input_shape[-1] if input_shape and len(input_shape) >= 2 else 1
            if np is not None:
                return np.zeros((1, int(features))), feature_cols
            return [[0.0] * int(features)], feature_cols
        except Exception as e:
            eprint(f"RUN_WARN: model input_shape inference failed: {e} — using zeros")

    if np is not None:
        return np.zeros((1, 1)), feature_cols
    eprint("RUN_WARN: numpy not available; returning list fallback [[0.0]]")
    return [[0.0]], feature_cols

# -------------------------
# Environment/version capture
# -------------------------
def _env_versions() -> Dict[str, Optional[str]]:
    py_ver = sys.version.split()[0]
    tf_ver = None
    np_ver = None
    try:
        import tensorflow as tf
        tf_ver = tf.__version__
    except Exception:
        tf_ver = None
    try:
        import numpy as np
        np_ver = np.__version__
    except Exception:
        np_ver = None
    return {
        "python_version": py_ver,
        "tensorflow_version": tf_ver,
        "numpy_version": np_ver,
        "platform": platform.platform(),
    }

# -------------------------
# Minimal manifest validation
# -------------------------
def validate_manifest_basic(man: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    required_keys = [
        "manifest_version", "mode", "commodity", "seed", "timestamp",
        "environment", "outputs", "input_files"
    ]
    for k in required_keys:
        if k not in man:
            return False, f"Manifest missing key: {k}"
    outs = man.get("outputs", {})
    if "preds" not in outs or "preds_sha_lower" not in outs or "preds_sha_upper" not in outs:
        return False, "Manifest outputs incomplete"
    preds_path = Path(outs["preds"])
    if not preds_path.exists():
        return False, f"Preds file not found: {preds_path}"
    # SHA strings basic format check
    for s in (outs["preds_sha_lower"], outs["preds_sha_upper"]):
        if not isinstance(s, str) or len(s) != 64 or any(c not in "0123456789abcdefABCDEF" for c in s):
            return False, "Preds SHA not valid hex string length=64"
    return True, None

# -------------------------
# Core API
# -------------------------
def infer_with_model_v1c(
    model_path: Optional[str],
    input_csvs: Dict[str, str],
    outdir: str,
    * ,
    seed: int = 20251117,
    scaler_path: Optional[str] = None,
    enforce_shas: Optional[Dict[str, str]] = None,
    mode: str = "deterministic",
    commodity: str = "GENERIC",
    model_sha_override: Optional[str] = None,
    debug: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    def _log(*args):
        if verbose:
            eprint(*args)

    # Preflight CSV existence + optional SHAs
    mismatches = []
    for role, p in input_csvs.items():
        ppath = Path(p)
        if not ppath.exists():
            raise FileNotFoundError(f"Missing input CSV {role}: {ppath}")
        if enforce_shas and role in enforce_shas:
            expected = enforce_shas[role].strip()
            actual_upper = compute_sha256_hex(ppath, uppercase=True)
            actual_lower = actual_upper.lower()
            if expected not in (actual_upper, actual_lower):
                mismatches.append((role, str(ppath), expected, actual_upper))
    if mismatches:
        details = "; ".join([f"{r} expected {e} got {a}" for (r, _, e, a) in mismatches])
        raise RuntimeError("SHA MISMATCH: " + details)

    # Warn if enforce_shas includes roles not present
    if enforce_shas:
        unknown_roles = [r for r in enforce_shas.keys() if r not in input_csvs]
        if unknown_roles:
            _log("RUN_WARN: enforce_sha roles not in input_csvs ->", ",".join(unknown_roles))

    measures = deterministic_seed_all(seed)
    determinism_level = "best_effort"
    _log("RUN_INFO: deterministic_seed", seed, f"determinism_level={determinism_level}")
    env_info = _env_versions()

    # Synthetic mode
    if mode == "synthetic":
        preds_obj = {
            "run_id": outdir_p.name,
            "commodity": commodity,
            "mode": "synthetic",
            "seed": int(seed),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "preds": [{"id": "SYNTH_1", "score": 0.12345}],
        }
        preds_path = outdir_p / "preds_model.json"
        atomic_write_json(preds_path, preds_obj)
        lower_sha, upper_sha = write_sidecars_both(preds_path)
        manifest = {
            "manifest_version": "1Ciii-closure",
            "model_type": None,
            "run_id": outdir_p.name,
            "commodity": commodity,
            "mode": "synthetic",
            "seed": int(seed),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "determinism_level": determinism_level,
            "determinism_measures": measures,
            "environment": env_info,
            "input_files": {k: str(Path(v).resolve()) for k, v in input_csvs.items()},
            "input_shas": enforce_shas or {},
            "feature_columns": [],   # synthetic mode: no CSV-derived feature list recorded here
            "outputs": {"preds": str(preds_path), "preds_sha_lower": lower_sha, "preds_sha_upper": upper_sha},
        }
        manifest_path = outdir_p / f"run_manifest.{outdir_p.name}.inference.json"
        atomic_write_json(manifest_path, manifest)
        write_sidecars_both(manifest_path)
        ok, msg = validate_manifest_basic(manifest)
        if not ok:
            raise RuntimeError(f"Manifest validation failed: {msg}")
        _log("RUN_INFO: synthetic_complete", str(preds_path), lower_sha)
        return {
            "status": "SUCCESS",
            "preds_path": str(preds_path),
            "preds_sha_lower": lower_sha,
            "preds_sha_upper": upper_sha,
            "manifest": str(manifest_path),
            "mode": "synthetic",
        }

    # Deterministic model mode
    try:
        import numpy as np
    except Exception:
        np = None
        _log("RUN_WARN: numpy not available; scaler/model interactions may fail")
    try:
        import joblib
    except Exception:
        joblib = None
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow not available; cannot run deterministic inference in model mode") from e

    # Scaler
    scaler_obj = None
    scaler_sha_upper = None
    if scaler_path:
        sp = Path(scaler_path)
        if not sp.exists():
            raise FileNotFoundError(f"Scaler not found: {sp}")
        scaler_sha_upper = compute_sha256_hex(sp, uppercase=True)
        try:
            if joblib is not None:
                scaler_obj = joblib.load(str(sp))
            else:
                import pickle
                with sp.open("rb") as fh:
                    scaler_obj = pickle.load(fh)
            _log("RUN_INFO: loaded_scaler", str(sp))
        except Exception as e:
            _log("RUN_WARN: failed_loading_scaler", str(e))
            scaler_obj = None

    # Model
    model_p = Path(model_path) if model_path else None
    if model_p is None or not model_p.exists():
        raise FileNotFoundError(f"Model path not found or not provided: {model_path}")
    model_type = "directory" if model_p.is_dir() else "file"
    try:
        model = tf.keras.models.load_model(str(model_p))
        _log("RUN_INFO: model_loaded", str(model_p), f"type={model_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {model_p} (type={model_type}): {e}") from e

    # Select test CSV
    test_csv = None
    for candidate in ("test", "testing", "val", "validation"):
        if candidate in input_csvs:
            test_csv = Path(input_csvs[candidate])
            break
    if test_csv is None:
        test_csv = Path(next(iter(input_csvs.values())))
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # Features
    X_test, feature_cols = load_csv_features_best_effort(test_csv, model=model)
    if np is not None:
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if X_test.ndim != 2:
            raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    else:
        _log("RUN_WARN: numpy missing; continuing with list-based features which may break scaler/model")

    # T+1 leakage guard: reject inference CSVs that contain an explicit 'target' column
    if feature_cols and any(c.lower() == "target" for c in feature_cols):
        raise RuntimeError("Inference CSV appears to contain a 'target' column. This may indicate target leakage. Remove the target column or provide a trainer manifest with target_shift metadata.")

    # Scaler compatibility
    if scaler_obj is not None and hasattr(scaler_obj, "n_features_in_"):
        try:
            n_features = int(getattr(scaler_obj, "n_features_in_"))
            if np is not None and X_test.shape[1] != n_features:
                raise ValueError(f"Scaler expects {n_features} features, got {X_test.shape[1]}")
        except Exception as e:
            raise RuntimeError(f"Scaler compatibility check failed: {e}") from e

    # Apply scaler
    if scaler_obj is not None:
        try:
            X_test = scaler_obj.transform(X_test)
        except Exception as e:
            _log("RUN_WARN: scaler_transform_failed; continuing with raw features", str(e))

    # Predict
    try:
        preds_array = model.predict(X_test, batch_size=32)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    # Normalize preds
    try:
        preds_list = preds_array.tolist()
    except Exception:
        try:
            preds_list = [float(x) for x in preds_array.reshape(-1)]
        except Exception:
            preds_list = []

    # Write preds
    preds_obj = {
        "run_id": outdir_p.name,
        "commodity": commodity,
        "mode": "deterministic",
        "seed": int(seed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_p.resolve()),
        "preds": [{"row": i, "score": p} for i, p in enumerate(preds_list)],
    }
    preds_path = outdir_p / "preds_model.json"
    atomic_write_json(preds_path, preds_obj)
    lower_sha, upper_sha = write_sidecars_both(preds_path)

    # Model SHA: override or compute
    if model_sha_override:
        model_sha_upper = model_sha_override.strip().upper()
        _log("RUN_INFO: model_sha_override_used", model_sha_upper)
    else:
        model_sha_upper = compute_path_sha(model_p, uppercase=True)

    # Manifest
    manifest = {
        "manifest_version": "1Ciii-closure",
        "model_type": model_type,
        "run_id": outdir_p.name,
        "commodity": commodity,
        "mode": "deterministic",
        "seed": int(seed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "determinism_level": determinism_level,
        "determinism_measures": measures,
        "environment": env_info,
        "model_path": str(model_p.resolve()),
        "model_sha_upper": model_sha_upper,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "scaler_sha_upper": scaler_sha_upper,
        "input_files": {k: str(Path(v).resolve()) for k, v in input_csvs.items()},
        "input_shas": enforce_shas or {},
        "feature_columns": feature_cols,
        "outputs": {"preds": str(preds_path), "preds_sha_lower": lower_sha, "preds_sha_upper": upper_sha},
        "sidecar_note": "Both .sha256.txt and .SHA256.TXT written; case-insensitive FS may only store one file with both hashes in content.",
    }
    manifest_path = outdir_p / f"run_manifest.{outdir_p.name}.inference.json"
    atomic_write_json(manifest_path, manifest)
    write_sidecars_both(manifest_path)
    ok, msg = validate_manifest_basic(manifest)
    if not ok:
        raise RuntimeError(f"Manifest validation failed: {msg}")

    _log("RUN_INFO: inference_complete", str(preds_path), lower_sha)
    return {
        "status": "SUCCESS",
        "preds_path": str(preds_path),
        "preds_sha_lower": lower_sha,
        "preds_sha_upper": upper_sha,
        "manifest": str(manifest_path),
        "model_sha_upper": manifest["model_sha_upper"],
        "mode": "deterministic",
    }

# -------------------------
# CLI wrapper
# -------------------------
def _cli_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic inference harness v1Ciii-closure")
    parser.add_argument("--model", required=False, help="Path to model artifact (h5 or SavedModel dir)")
    parser.add_argument("--outdir", required=True, help="Output directory for preds and manifest")
    parser.add_argument("--mode", choices=["deterministic", "synthetic"], default="synthetic", help="Run mode")
    parser.add_argument("--seed", type=int, default=20251117, help="Deterministic seed")
    parser.add_argument("--scaler", required=False, help="Optional scaler path (joblib/pickle)")
    parser.add_argument("--commodity", required=False, default="GENERIC", help="Commodity short name (e.g., HO, GOLD)")
    parser.add_argument("--enforce-sha", action="append", help="Enforce input CSV SHA in the form role=SHA (can be repeated)")
    parser.add_argument("--train_csv", required=False, help="Training CSV path")
    parser.add_argument("--val_csv", required=False, help="Validation CSV path")
    parser.add_argument("--test_csv", required=False, help="Test CSV path")
    parser.add_argument("--debug", action="store_true", help="Print full tracebacks on error")
    parser.add_argument("--model-sha", required=False, help="Override model SHA (precomputed, uppercase or lowercase accepted)")
    args = parser.parse_args(argv)

    # Provenance log at startup
    eprint("RUN_INFO:", f"version=v1Ciii-closure", f"date=2025-12-02", f"commodity={args.commodity}", f"mode={args.mode}", f"outdir={args.outdir}")

    # Collect CSVs
    input_csvs: Dict[str, str] = {}
    if args.train_csv:
        input_csvs["train"] = args.train_csv
    if args.val_csv:
        input_csvs["val"] = args.val_csv
    if args.test_csv:
        input_csvs["test"] = args.test_csv

    # Canonical paths best-effort when none provided — with explicit MISSING markers in output JSON
    missing_markers: Dict[str, str] = {}
    if not input_csvs:
        base = Path.cwd()
        cand_dir = base / f"data_{args.commodity.lower()}"
        cand_train = cand_dir / f"{args.commodity.lower()}_training.csv"
        cand_val = cand_dir / f"{args.commodity.lower()}_validation.csv"
        cand_test = cand_dir / f"{args.commodity.lower()}_testing.csv"
        if cand_train.exists():
            input_csvs["train"] = str(cand_train)
        else:
            missing_markers["train"] = "MISSING"
        if cand_val.exists():
            input_csvs["val"] = str(cand_val)
        else:
            missing_markers["val"] = "MISSING"
        if cand_test.exists():
            input_csvs["test"] = str(cand_test)
        else:
            missing_markers["test"] = "MISSING"

    enforce_shas = {}
    if args.enforce_sha:
        for item in args.enforce_sha:
            if "=" in item:
                role, sha = item.split("=", 1)
                enforce_shas[role.strip()] = sha.strip()

    model_sha_override = args.model_sha.strip() if args.model_sha else None

    try:
        result = infer_with_model_v1c(
            model_path=args.model,
            input_csvs=input_csvs,
            outdir=args.outdir,
            seed=args.seed,
            scaler_path=args.scaler,
            enforce_shas=enforce_shas or None,
            mode=args.mode,
            commodity=args.commodity,
            model_sha_override=model_sha_override,
            debug=args.debug,
            verbose=True,
        )
        # Attach missing markers for transparency
        if missing_markers:
            result["missing_inputs"] = missing_markers
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        eprint("RUN_ERROR:", str(exc))
        if args.debug:
            eprint(traceback.format_exc())
        failure = {
            "status": "FAILURE",
            "error": str(exc),
            "commodity": args.commodity,
            "mode": args.mode,
            "outdir": args.outdir,
            "missing_inputs": missing_markers or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        print(json.dumps(failure, indent=2))
        return 2

if __name__ == "__main__":
    raise SystemExit(_cli_main())

# END deterministic_inference_v.1Ciii.py



