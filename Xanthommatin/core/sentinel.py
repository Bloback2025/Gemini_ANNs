#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: sentinel.py
Date of Creation: September 9, 2026
Timestamp: 2026-09-09T11:05:00Z
Path: Xanthommatin/core/sentinel.py

Annotations:
- The DI (Deterministic Core Shell) Alignment Sentinel.
- Implements deterministic byte-level serialization, atomic writes, and robust sidecar verification.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from core.harness import write_sidecars_both, compute_sha256_hex

DI_CORE_AXIOMS = {
    "system_name": "Deterministic Inference / Deterministic Core Shell (DI)",
    "primary_mandate": "Universal architectural harness, memory structure, and execution control plane for broad project applications.",
    "architectural_rules": [
        "Domain-agnostic: DI governs any arbitrary workload, not restricted to finance or prediction.",
        "Absolute Determinism: Strict environment seeding, environmental locking, and bit-for-bit reproducibility.",
        "Cryptographic Provenance: Recursive SHA-256 sidecars and signed manifests for all states.",
        "Zero Drift: The harness controls the payload; the payload never corrupts the core harness."
    ],
    "version": "1.0.0-anchor"
}

ANCHOR_FILENAME = "di_alignment_anchor.json"
_SIDE_CAR_SUFFIX = ".sha256.txt"
_SIGNATURE_SUFFIX = ".sig"  # Placeholder for future asymmetric signature hooks


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


def _sidecar_path_for(anchor_path: Path) -> Path:
    return anchor_path.with_name(anchor_path.name + _SIDE_CAR_SUFFIX)


def generate_di_anchor(workspace_path: Path) -> Path:
    """
    Create or refresh the DI anchor file with deterministic JSON serialization bytes,
    write atomically to disk, then generate cryptographic sidecars. Returns the anchor path.
    """
    anchor_path = workspace_path / ANCHOR_FILENAME
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "axioms": DI_CORE_AXIOMS
    }
    data = _deterministic_json_bytes(payload)
    _atomic_write_bytes(anchor_path, data)
    write_sidecars_both(anchor_path)
    return anchor_path


def verify_di_alignment(workspace_path: Path) -> Tuple[bool, str]:
    """
    Verify the anchor exists and its SHA-256 sidecar matches the computed hash.
    Returns (True, message) on success, (False, diagnostic) on failure.
    """
    anchor_path = workspace_path / ANCHOR_FILENAME
    sidecar_path = _sidecar_path_for(anchor_path)

    if not anchor_path.exists() or not anchor_path.is_file():
        return False, f"DI Alignment Anchor missing: {anchor_path}"

    if not sidecar_path.exists() or not sidecar_path.is_file():
        return False, f"DI Alignment sidecar missing: {sidecar_path}"

    try:
        current_hash = compute_sha256_hex(anchor_path)
    except Exception as exc:
        return False, f"Failed to compute hash for {anchor_path}: {exc}"

    try:
        content = sidecar_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return False, f"Failed to read sidecar {sidecar_path}: {exc}"

    if not content:
        return False, f"Sidecar {sidecar_path} is empty."

    try:
        first_line = content.splitlines()[0].strip()
        stored_hash = first_line.split()[0].lower()
    except (IndexError, AttributeError) as exc:
        return False, f"Malformed sidecar {sidecar_path}: {exc}"

    if len(stored_hash) != 64 or any(c not in "0123456789abcdef" for c in stored_hash):
        return False, f"Sidecar {sidecar_path} does not contain a valid SHA-256 hex string."

    if current_hash.lower() != stored_hash:
        return False, f"Mismatch computed={current_hash} stored={stored_hash}"

    # Optional signature verification hook if present
    sig_path = anchor_path.with_name(anchor_path.name + _SIGNATURE_SUFFIX)
    if sig_path.exists():
        # Reserved for future public-key signature verification
        pass

    return True, "DI Alignment Verified: Universal harness integrity intact."