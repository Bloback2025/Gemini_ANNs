#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: test_di.py
Date of Creation: September 9, 2026
Timestamp: 2026-09-09T12:05:00Z
Path: Xanthommatin/tests/test_di.py

Annotations:
- Automated regression test suite for the Deterministic Core Shell (DI).
- Validates byte-level determinism, atomic persistence, cryptographic sidecar integrity,
  and aggressive defensive parsing against corruption or tampering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from core.harness import (
    _deterministic_json_bytes,
    _atomic_write_bytes,
    compute_sha256_hex,
    write_sidecars_both,
)
from core.sentinel import (
    ANCHOR_FILENAME,
    generate_di_anchor,
    verify_di_alignment,
)


def test_deterministic_json_serialization():
    """Verify that JSON serialization is strictly deterministic across re-ordering."""
    obj_a = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
    obj_b = {"a": 2, "z": 1, "m": {"a": 4, "b": 3}}
    
    bytes_a = _deterministic_json_bytes(obj_a)
    bytes_b = _deterministic_json_bytes(obj_b)
    
    assert bytes_a == bytes_b, "Deterministic JSON bytes failed to normalize key ordering."
    assert bytes_a == b'{"a":2,"m":{"a":4,"b":3},"z":1}', "JSON bytes do not match expected compact format."


def test_atomic_write_persistence(tmp_path: Path):
    """Verify atomic write mechanics and cleanup of temporary artifacts."""
    target = tmp_path / "test_atomic.bin"
    payload = b"Xanthommatin DI Core Shell Test Payload"
    
    _atomic_write_bytes(target, payload)
    
    assert target.exists(), "Target file was not written."
    assert target.read_bytes() == payload, "Written payload does not match source bytes."
    
    # Ensure no stray tempfiles are left behind in the directory
    dir_contents = list(tmp_path.iterdir())
    assert len(dir_contents) == 1, f"Stray files left in directory after atomic write: {dir_contents}"


def test_sentinel_anchor_generation_and_verification(tmp_path: Path):
    """Verify successful anchor generation and positive validation check."""
    anchor_path = generate_di_anchor(tmp_path)
    
    assert anchor_path.exists()
    assert (tmp_path / ANCHOR_FILENAME).exists()
    assert (tmp_path / f"{ANCHOR_FILENAME}.sha256.txt").exists()
    
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is True, f"Valid alignment check failed: {msg}"


def test_sentinel_catches_missing_anchor(tmp_path: Path):
    """Verify that a missing anchor file is caught immediately."""
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is False
    assert "anchor missing" in msg.lower()


def test_sentinel_catches_missing_sidecar(tmp_path: Path):
    """Verify that a missing sidecar file triggers a validation failure."""
    generate_di_anchor(tmp_path)
    sidecar = tmp_path / f"{ANCHOR_FILENAME}.sha256.txt"
    sidecar.unlink()
    
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is False
    assert "sidecar missing" in msg.lower()


def test_sentinel_catches_content_tampering(tmp_path: Path):
    """Verify that mutating the anchor file content triggers a hash mismatch failure."""
    generate_di_anchor(tmp_path)
    anchor_path = tmp_path / ANCHOR_FILENAME
    
    # Tamper with the anchor file content
    current_content = anchor_path.read_text(encoding="utf-8")
    tampered_content = current_content.replace("Deterministic", "Corrupted")
    anchor_path.write_text(tampered_content, encoding="utf-8")
    
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is False
    assert "mismatch" in msg.lower()


def test_sentinel_defends_against_malformed_sidecar(tmp_path: Path):
    """Verify that a malformed or empty sidecar fails defensively rather than throwing unhandled exceptions."""
    generate_di_anchor(tmp_path)
    sidecar = tmp_path / f"{ANCHOR_FILENAME}.sha256.txt"
    
    # Test empty sidecar
    sidecar.write_text("", encoding="ascii")
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is False
    assert "empty" in msg.lower()
    
    # Test invalid hash length / garbage text in sidecar
    sidecar.write_text("not_a_hex_hash_string\n", encoding="ascii")
    is_aligned, msg = verify_di_alignment(tmp_path)
    assert is_aligned is False
    assert ("not contain a valid" in msg.lower() or "malformed" in msg.lower())