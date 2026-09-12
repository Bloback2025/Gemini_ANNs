#!/usr/bin/env python3
r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: deterministic_inference_v0.1_beta.py
Date of Creation: September 10, 2026 2:37 PM
Path: C:\\Users\\loweb\\AI_Financial_Sims\\Gemini_ANNs\\Xanthommatin/deterministic_inference_v0.1_beta.py

Annotations:
- DI-Core Beta v0.1: Universal, domain-agnostic resource orchestration and governance harness.
- Inherits paranoid cryptographic provenance (dual-case SHA-256 sidecars, atomic writes) and strict preflight boundary/schema firewall validation from the Project DI architecture.
- Provides multi-branch execution capabilities and strict deterministic environment seeding to eliminate non-determinism and drift during resource aggregation tasks.
"""

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
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Filesystem & Atomic Helpers
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
    data = json.dumps(obj, sort_keys=True, indent=indent).encode("utf-8")
    _atomic_write_bytes(path, data)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


# -------------------------
# Cryptographic Provenance
# -------------------------
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
    lower = compute_path_sha(path, uppercase=False)
    upper = lower.upper()
    lower_side = path.with_name(path.name + ".sha256.txt")
    upper_side = path.with_name(path.name + ".SHA256.TXT")
    if str(lower_side).lower() == str(upper_side).lower():
        lower_side.write_text(lower + "\n" + upper + "\n", encoding="ascii")
        return lower, upper
    lower_side.write_text(lower + "\n", encoding="ascii")
    upper_side.write_text(upper + "\n", encoding="ascii")
    return lower, upper


# -------------------------
# Deterministic Seeding
# -------------------------
def deterministic_seed_all(seed: int) -> Dict[str, Any]:
    measures = {
        "PYTHONHASHSEED_set": None,
        "numpy_seed_set": None,
        "random_seed_set": int(seed),
    }
    try:
        os.environ["PYTHONHASHSEED"] = str(int(seed))
        measures["PYTHONHASHSEED_set"] = str(int(seed))
    except Exception:
        pass
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(int(seed))
        measures["numpy_seed_set"] = int(seed)
    except Exception:
        pass
    return measures


# -------------------------
# Schema Firewall & Preflight
# -------------------------
def validate_payload_schema(payload: Dict[str, Any], forbidden_keys: Optional[List[str]] = None) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary object.")
    if forbidden_keys:
        for fk in forbidden_keys:
            if fk in payload or any(fk.lower() in str(k).lower() for k in payload.keys()):
                raise RuntimeError(f"Firewall Violation: Forbidden structural key detected -> '{fk}'")


# -------------------------
# Environment & Manifest Validation
# -------------------------
def _env_versions() -> Dict[str, Optional[str]]:
    py_ver = sys.version.split()[0]
    np_ver = None
    try:
        import numpy as np  # type: ignore
        np_ver = np.__version__
    except Exception:
        np_ver = None
    return {
        "python_version": py_ver,
        "numpy_version": np_ver,
        "platform": platform.platform(),
    }


def validate_manifest_basic(man: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    required_keys = [
        "manifest_version",
        "mode",
        "namespace",
        "seed",
        "timestamp",
        "environment",
        "outputs",
        "inputs",
    ]
    for k in required_keys:
        if k not in man:
            return False, f"Manifest missing key: {k}"
    outs = man.get("outputs", {})
    if "primary_output" not in outs or "primary_sha_lower" not in outs or "primary_sha_upper" not in outs:
        return False, "Manifest outputs incomplete"
    return True, None


# -------------------------
# Core Execution Harness
# -------------------------
def execute_governed_pipeline(
    input_manifest_path: Optional[str],
    outdir: str,
    *,
    seed: int = 20260910,
    mode: str = "deterministic",
    namespace: str = "DEFAULT",
    forbidden_keys: Optional[List[str]] = None,
    enforce_shas: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    def _log(*args):
        if verbose:
            eprint(*args)

    # 1. Input Validation & SHA Enforcement
    inputs: Dict[str, str] = {}
    if input_manifest_path:
        in_p = Path(input_manifest_path)
        if not in_p.exists():
            raise FileNotFoundError(f"Input resource manifest not found: {in_p}")
        inputs["source"] = str(in_p.resolve())
        if enforce_shas and "source" in enforce_shas:
            expected = enforce_shas["source"].strip()
            actual_upper = compute_sha256_hex(in_p, uppercase=True)
            if expected not in (actual_upper, actual_upper.lower()):
                raise RuntimeError(f"SHA MISMATCH for source input. Expected {expected}, got {actual_upper}")

    # 2. Seeding & Environment Capture
    measures = deterministic_seed_all(seed)
    _log("RUN_INFO: deterministic_seed initialized", seed)
    env_info = _env_versions()

    # 3. Load Payload & Fire Schema Check
    payload = {}
    if input_manifest_path and inputs.get("source"):
        try:
            with open(inputs["source"], "r", encoding="utf-8-sig") as fh:
                payload = json.load(fh)
        except Exception as e:
            _log("RUN_WARN: Failed to parse input payload as JSON; treating as empty/raw.")
            payload = {"raw_path": inputs["source"]}

    validate_payload_schema(payload, forbidden_keys=forbidden_keys)

    # 4. Core Execution Branches (Multi-Branch Evaluation)
    branch_results = []
    branches = payload.get("branches", ["alpha", "beta", "gamma"])
    
    for idx, branch_name in enumerate(branches, start=1):
        branch_out = outdir_p / f"branch_{branch_name}"
        branch_out.mkdir(parents=True, exist_ok=True)
        
        # Simulate deterministic transformation per branch
        simulated_score = round(random.random() * 1000, 4)
        branch_data = {
            "branch_id": idx,
            "branch_name": branch_name,
            "status": "GOVERNED_SUCCESS",
            "score": simulated_score,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        b_path = branch_out / "result.json"
        atomic_write_json(b_path, branch_data)
        l_sha, u_sha = write_sidecars_both(b_path)
        branch_results.append({"branch": branch_name, "path": str(b_path.resolve()), "sha": l_sha})
        _log(f"RUN_INFO: Branch '{branch_name}' executed successfully.")

    # 5. Master Summary Output
    master_output_path = outdir_p / "master_execution_summary.json"
    master_obj = {
        "namespace": namespace,
        "mode": mode,
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "branch_results": branch_results,
    }
    atomic_write_json(master_output_path, master_obj)
    primary_lower, primary_upper = write_sidecars_both(master_output_path)

    # 6. Final Run Manifest
    manifest = {
        "manifest_version": "v0.1-beta-closure",
        "framework": "DI-Core",
        "namespace": namespace,
        "mode": mode,
        "seed": int(seed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "determinism_measures": measures,
        "environment": env_info,
        "inputs": inputs,
        "outputs": {
            "primary_output": str(master_output_path.resolve()),
            "primary_sha_lower": primary_lower,
            "primary_sha_upper": primary_upper,
        },
    }
    manifest_path = outdir_p / f"run_manifest.{namespace.lower()}.json"
    atomic_write_json(manifest_path, manifest)
    write_sidecars_both(manifest_path)

    ok, msg = validate_manifest_basic(manifest)
    if not ok:
        raise RuntimeError(f"Manifest validation failed: {msg}")

    _log("RUN_INFO: Governed execution complete successfully.", str(master_output_path))
    return {
        "status": "SUCCESS",
        "primary_output": str(master_output_path),
        "primary_sha_lower": primary_lower,
        "primary_sha_upper": primary_upper,
        "manifest": str(manifest_path),
    }


# -------------------------
# CLI Wrapper
# -------------------------
def _cli_main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="DI-Core Agnostic Governance Harness (Beta v0.1)")
    parser.add_argument("--input", required=False, help="Path to input resource json manifest")
    parser.add_argument("--outdir", required=True, help="Output directory for governed artifacts")
    parser.add_argument("--namespace", default="GENERIC", help="Operational namespace identifier")
    parser.add_argument("--seed", type=int, default=20260910, help="Deterministic seed")
    parser.add_argument("--mode", choices=["deterministic", "sandbox"], default="deterministic", help="Execution mode")
    parser.add_argument("--forbidden", action="append", help="Forbidden keys to block via schema firewall (can repeat)")
    parser.add_argument("--enforce-sha", help="Enforce input SHA in format source=SHA")
    parser.add_argument("--debug", action="store_true", help="Print full traceback on error")
    args = parser.parse_args(argv)

    eprint("RUN_INFO:", f"harness=DI-Core-Beta", f"namespace={args.namespace}", f"outdir={args.outdir}")

    enforce_shas = {}
    if args.enforce_sha and "=" in args.enforce_sha:
        k, v = args.enforce_sha.split("=", 1)
        enforce_shas[k.strip()] = v.strip()

    try:
        result = execute_governed_pipeline(
            input_manifest_path=args.input,
            outdir=args.outdir,
            seed=args.seed,
            mode=args.mode,
            namespace=args.namespace,
            forbidden_keys=args.forbidden,
            enforce_shas=enforce_shas or None,
            verbose=True,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        eprint("RUN_ERROR:", str(exc))
        if args.debug:
            eprint(traceback.format_exc())
        failure = {
            "status": "FAILURE",
            "error": str(exc),
            "namespace": args.namespace,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        print(json.dumps(failure, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())