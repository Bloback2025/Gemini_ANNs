#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: harness.py
Date of Creation: September 9, 2026
Timestamp: 2026-09-09T11:35:00Z
Path: Xanthommatin/core/harness.py

Annotations:
- The immutable Deterministic Core Shell (DCS) for Xanthommatin.
- Hardened Methodological Standards:
  1. Byte-Level Determinism: 
     All JSON serializations pass through strict byte-level standardization 
     (`sort_keys=True`, compact separators `("," , ":")`, explicit UTF-8 encoding) 
     to eliminate cross-platform key ordering or whitespace variance.
  2. Atomic Persistence & Concurrency Safety: 
     File writes avoid partial or corrupt states by utilizing low-level 
     transactional primitives: `tempfile.mkstemp()` inside the target directory, 
     explicit buffer flushing (`f.flush()`), OS-level descriptor synchronization 
     (`os.fsync()`), and atomic pointer swapping (`os.replace()`).
  3. Cryptographic Provenance: 
     Recursive and single-file SHA-256 computation paired with automated 
     dual-case sidecar generation (`.sha256.txt` and `.SHA256.TXT`) ensures 
     unbreakable audit trails and tamper-evident outputs.
  4. Environmental Seeding & Locking: 
     Hardcodes `TF_DETERMINISTIC_OPS` and seeds Python random, NumPy, and 
     TensorFlow to enforce bit-for-bit reproducibility across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Enforce deterministic environment flags early before any potential framework import
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

# Lazy library detection
_tf_available = True
try:
    import tensorflow as tf  # type: ignore
except Exception:
    _tf_available = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _deterministic_json_bytes(obj: Any) -> bytes:
    """
    Return deterministic UTF-8 bytes for JSON serialization.
    Uses sort_keys and compact separators to ensure stable byte representation.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Atomically write raw bytes to disk using a temp file, fsync, and replace.
    Ensures zero partial writes or concurrency corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dirpath = path.parent
    fd, tmp = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def atomic_write_json(path: Path, obj: Any) -> None:
    """
    Serialize a Python object deterministically into JSON bytes and write atomically.
    """
    data = _deterministic_json_bytes(obj)
    _atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text strings to disk.
    """
    _atomic_write_bytes(path, text.encode(encoding))


def compute_sha256_hex(path: Path, *, uppercase: bool = False, chunk_size: int = 8192) -> str:
    """
    Compute the SHA-256 hash of a single file with chunked memory efficiency.
    """
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
    Compute recursive SHA-256 hash across a file or a sorted directory structure.
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
    Generate dual-case (lowercase and uppercase) SHA-256 sidecar files atomically.
    """
    lower = compute_path_sha(path, uppercase=False)
    upper = lower.upper()
    lower_side = path.with_name(path.name + ".sha256.txt")
    upper_side = path.with_name(path.name + ".SHA256.TXT")
    
    if str(lower_side).lower() == str(upper_side).lower():
        atomic_write_text(lower_side, lower + "\n" + upper + "\n", encoding="ascii")
        return lower, upper
        
    atomic_write_text(lower_side, lower + "\n", encoding="ascii")
    atomic_write_text(upper_side, upper + "\n", encoding="ascii")
    return lower, upper


def deterministic_seed_all(seed: int) -> Dict[str, Any]:
    """
    Lock environment variables, Python random, NumPy, and TensorFlow random states.
    """
    measures = {
        "PYTHONHASHSEED_set": None,
        "numpy_seed_set": None,
        "tf_seed_set": None,
        "tf_threading_set": None,
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
    }
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
        measures["PYTHONHASHSEED_set"] = str(int(seed))
    except Exception:
        pass
    try:
        random.seed(int(seed))
        if np is not None:
            np.random.seed(int(seed))
            measures["numpy_seed_set"] = int(seed)
    except Exception:
        pass
    if _tf_available:
        try:
            import tensorflow as tf  # type: ignore

            tf.random.set_seed(int(seed))
            measures["tf_seed_set"] = int(seed)
            try:
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                measures["tf_threading_set"] = True
            except Exception:
                measures["tf_threading_set"] = False
        except Exception:
            pass
    return measures


def validate_manifest_basic(man: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Perform structural validation on a universal execution manifest.
    """
    required_keys = [
        "manifest_version",
        "mode",
        "seed",
        "timestamp",
        "environment",
        "outputs",
        "input_files",
    ]
    for k in required_keys:
        if k not in man:
            return False, f"Manifest missing key: {k}"
            
    outs = man.get("outputs", {})
    required_output_keys = ["artifact", "artifact_sha_lower", "artifact_sha_upper"]
    for k in required_output_keys:
        if k not in outs:
            return False, f"Manifest outputs missing required key: {k}"
            
    artifact_path = Path(outs["artifact"])
    if not artifact_path.exists():
        return False, f"Artifact file not found: {artifact_path}"
        
    for s in (outs["artifact_sha_lower"], outs["artifact_sha_upper"]):
        if not isinstance(s, str) or len(s) != 64 or any(c not in "0123456789abcdefABCDEF" for c in s):
            return False, "Artifact SHA not valid hex string length=64"
            
    return True, None