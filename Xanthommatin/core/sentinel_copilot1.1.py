#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: sentinel.py
Date of Creation: September 9, 2026
Timestamp: 2026-09-09T11:05:00Z
Path: Xanthommatin/core/sentinel_copilot1.1.py

Purpose:
- Verify run manifests produced by the DI harness.
- Recompute canonical signature_hash, verify Ed25519 signature (local or KMS hook),
  enforce version monotonicity and previous_hash continuity against an append-only ledger,
  append verified entries, emit signed verification artifacts, and write quarantine markers on failure.

Notes:
- In production replace the local key verification with a KMS/HSM verification call.
- Do NOT store private keys in the repo. For local dev, place public keys in the pubkeys directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional dependency: cryptography for Ed25519 verification
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
except Exception:
    Ed25519PublicKey = None  # type: ignore

# -------------------------
# Configuration defaults
# -------------------------
DEFAULT_LEDGER = "core/ledger.log"
DEFAULT_PUBKEYS_DIR = "core/pubkeys"
DEFAULT_VERIFY_OUT = "core/verify_out"
QUARANTINE_MARKER = "quarantine.marker.json"

# -------------------------
# Atomic write helpers
# -------------------------
def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tf:
        tf.write(data)
        tf.flush()
        try:
            os.fsync(tf.fileno())
        except Exception:
            pass
    os.replace(tf.name, str(path))
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


# -------------------------
# Canonicalization and hashing
# -------------------------
def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().lower()


def compute_state_signature_hash(current_state: Dict[str, Any], version: int, previous_hash: str) -> str:
    payload = {
        "version": int(version),
        "previous_hash": previous_hash or "",
        "current_state": current_state,
    }
    return compute_sha256_hex_bytes(canonical_bytes(payload))


# -------------------------
# Ledger helpers (simple append-only ledger)
# -------------------------
def ledger_tail(ledger_path: Path) -> Optional[Dict[str, Any]]:
    if not ledger_path.exists():
        return None
    try:
        with ledger_path.open("r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if not lines:
            return None
        last = json.loads(lines[-1])
        return last
    except Exception:
        return None


def append_ledger_entry(ledger_path: Path, entry: Dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    # Append with flush and fsync for durability
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    _fsync_dir(ledger_path.parent)


# -------------------------
# Signature verification
# -------------------------
def load_public_key_from_file(pubkey_path: Path) -> Optional[Ed25519PublicKey]:
    if Ed25519PublicKey is None:
        raise RuntimeError("cryptography library not available for Ed25519 verification")
    if not pubkey_path.exists():
        return None
    data = pubkey_path.read_bytes()
    try:
        # Accept PEM or raw public key bytes
        try:
            key = serialization.load_pem_public_key(data)
            if isinstance(key, Ed25519PublicKey):
                return key
        except Exception:
            # try raw 32-byte key
            if len(data) == 32:
                return Ed25519PublicKey.from_public_bytes(data)
    except Exception:
        return None
    return None


def verify_ed25519_signature(pubkey: Ed25519PublicKey, signature_b64: str, message_hex: str) -> bool:
    try:
        sig = base64.b64decode(signature_b64)
        msg = bytes.fromhex(message_hex)
        pubkey.verify(sig, msg)
        return True
    except Exception:
        return False


# Placeholder for KMS/HSM verification hook
def verify_signature_via_kms(signer_id: str, signature_b64: str, message_hex: str) -> bool:
    """
    Replace this with a real KMS/HSM verification call in production.
    Return True if signature verifies under signer_id.
    """
    # Example: call AWS KMS Verify or Cloud KMS asymmetric verify
    return False


# -------------------------
# Manifest schema validation
# -------------------------
def validate_manifest_schema(man: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    required = ["version", "previous_hash", "current_state", "signature_hash", "signer", "signature", "timestamp"]
    for k in required:
        if k not in man:
            return False, f"Manifest missing required key: {k}"
    try:
        int(man["version"])
    except Exception:
        return False, "version must be integer"
    if not isinstance(man["current_state"], dict):
        return False, "current_state must be an object"
    if not isinstance(man["signature_hash"], str) or len(man["signature_hash"]) != 64:
        return False, "signature_hash must be 64-hex string"
    if not isinstance(man["signature"], str):
        return False, "signature must be base64 string"
    return True, None


# -------------------------
# Verification artifact helpers
# -------------------------
def make_verification_artifact(manifest_path: Path, result: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_path": str(manifest_path.resolve()),
        "result": result,
        "details": details,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


# -------------------------
# Quarantine marker
# -------------------------
def write_quarantine_marker(marker_path: Path, reason: str, manifest_path: Path) -> None:
    marker = {
        "quarantine": True,
        "reason": reason,
        "manifest": str(manifest_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(marker_path, marker)


# -------------------------
# Main verification flow
# -------------------------
def verify_manifest(
    manifest_path: Path,
    ledger_path: Path,
    pubkeys_dir: Path,
    verification_outdir: Path,
    allow_kms: bool = True,
) -> int:
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        man = json.loads(raw)
    except Exception as e:
        print(f"FAIL: cannot parse manifest JSON: {e}", file=sys.stderr)
        write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, f"manifest_parse_error: {e}", manifest_path)
        return 2

    ok, msg = validate_manifest_schema(man)
    if not ok:
        print(f"FAIL: manifest schema invalid: {msg}", file=sys.stderr)
        write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, f"schema_invalid: {msg}", manifest_path)
        return 2

    # Recompute signature_hash
    version = int(man["version"])
    previous_hash = man.get("previous_hash", "") or ""
    current_state = man["current_state"]
    computed_hash = compute_state_signature_hash(current_state, version, previous_hash)
    if computed_hash != man["signature_hash"].lower():
        reason = f"signature_hash_mismatch computed={computed_hash} recorded={man['signature_hash']}"
        print(f"FAIL: {reason}", file=sys.stderr)
        write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, reason, manifest_path)
        return 2

    # Verify signature: try KMS first (if allowed), then local pubkey
    signer = man["signer"]
    signature_b64 = man["signature"]
    message_hex = computed_hash  # hex string

    verified = False
    if allow_kms:
        try:
            if verify_signature_via_kms(signer, signature_b64, message_hex):
                verified = True
        except Exception:
            verified = False

    if not verified:
        # local pubkey lookup: signer may be key-id or filename
        pubkey_path = pubkeys_dir / f"{signer}.pub"
        if not pubkey_path.exists():
            pubkey_path = pubkeys_dir / signer
        try:
            pubkey = load_public_key_from_file(pubkey_path)
            if pubkey is not None:
                verified = verify_ed25519_signature(pubkey, signature_b64, message_hex)
        except Exception:
            verified = False

    if not verified:
        reason = "signature_verification_failed"
        print(f"FAIL: {reason}", file=sys.stderr)
        write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, reason, manifest_path)
        return 2

    # Ledger continuity and version monotonicity
    tail = ledger_tail(ledger_path)
    if tail:
        tail_version = int(tail.get("version", -1))
        tail_hash = tail.get("signature_hash", "")
        if version <= tail_version:
            reason = f"version_not_monotonic current={version} tail={tail_version}"
            print(f"FAIL: {reason}", file=sys.stderr)
            write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, reason, manifest_path)
            return 2
        if previous_hash and previous_hash != tail_hash:
            reason = f"previous_hash_mismatch previous={previous_hash} tail_hash={tail_hash}"
            print(f"FAIL: {reason}", file=sys.stderr)
            write_quarantine_marker(verification_outdir / QUARANTINE_MARKER, reason, manifest_path)
            return 2

    # All checks passed: append to ledger and write verification artifact
    ledger_entry = {
        "version": version,
        "signature_hash": computed_hash,
        "signer": signer,
        "manifest": str(manifest_path.resolve()),
        "timestamp": man.get("timestamp"),
    }
    append_ledger_entry(ledger_path, ledger_entry)

    verification_artifact = make_verification_artifact(manifest_path, "PASS", {"version": version, "signature_hash": computed_hash, "signer": signer})
    # Optionally sign the verification artifact with sentinel's local key (omitted here; placeholder)
    out_name = f"verification.{version}.{computed_hash[:8]}.json"
    atomic_write_json(verification_outdir / out_name, verification_artifact)
    print("PASS: manifest verified", file=sys.stdout)
    return 0


# -------------------------
# CLI
# -------------------------
def _cli_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Sentinel Copilot 1.1 - manifest verifier")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON to verify")
    parser.add_argument("--ledger", required=False, default=DEFAULT_LEDGER, help="Path to append-only ledger file")
    parser.add_argument("--pubkeys", required=False, default=DEFAULT_PUBKEYS_DIR, help="Directory containing public keys (signer.pub)")
    parser.add_argument("--outdir", required=False, default=DEFAULT_VERIFY_OUT, help="Directory to write verification artifacts and quarantine markers")
    parser.add_argument("--no-kms", action="store_true", help="Disable KMS/HSM verification hook")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    ledger_path = Path(args.ledger)
    pubkeys_dir = Path(args.pubkeys)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rc = verify_manifest(manifest_path, ledger_path, pubkeys_dir, outdir, allow_kms=not args.no_kms)
    return rc


if __name__ == "__main__":
    raise SystemExit(_cli_main())
