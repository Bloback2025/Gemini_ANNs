README v1.1 Deterministic Core Shell 

Methodological Specification & Operational Protocol
Author: Brian Lowe

Copyright (c) 2026 Brian Lowe. All rights reserved.

Date of Creation: September 9, 2026

Status: Production Architecture Standard

1. Executive Summary & Methodological Mandate
Project DI (Deterministic Inference / Deterministic Core Shell) is a universal architectural harness and state-management control plane designed to govern arbitrary computational payloads. To achieve true audit-proof execution, the framework rejects casual scripting assumptions in favor of strict byte-level determinism, defensive parsing, and atomic persistence.

This document codifies the core engineering principles derived from rigorous architectural reviews, translating them from theoretical axioms into hard rules that govern every line of code and every phase of execution.

2. Chapter and Verse: Methodological Pillars
The DI architecture directly addresses every point of vulnerability identified in rigorous system audits. The following breakdown maps methodological requirements to concrete implementation standards:

I. Byte-Level Determinism (Eliminating Serialization Variance)
The Vulnerability: Relying on default JSON serialization (json.dump without structural constraints) introduces cross-platform variance in key ordering and whitespace, breaking cryptographic hashes even when logic is identical.

The DI Standard: All state objects and manifests must pass through _deterministic_json_bytes(), which enforces:

sort_keys=True to eliminate dictionary key-ordering drift.

Compact separators (separators=(",", ":")) to strip trailing whitespace.

Explicit UTF-8 byte encoding (.encode("utf-8")) to standardize character representation.

II. Defensive Sidecar Parsing & Normalization
The Vulnerability: Assuming sidecar files contain clean, single-line hashes makes parsers brittle to manual edits, trailing newlines, or metadata strings (e.g., standard tool output formats like hash filename).

The DI Standard: Sidecar validation must be aggressively defensive:

Safely split lines and extract the first token (content.splitlines()[0].strip().split()[0]).

Normalize all hashes to lowercase (.lower()).

Enforce strict validation: exact length of 64 characters and pure hexadecimal character sets (0-9, a-f).

Explicitly handle edge cases such as empty files, missing sidecars, or unreadable paths via structured exception handling.

III. Atomic State Replacement & Concurrency Safety
The Vulnerability: Writing files directly or using incomplete temporary file patterns risks partial writes, race conditions during concurrent updates, and corrupted disk states if a process crashes mid-write.

The DI Standard: State writes must utilize low-level atomic primitives (_atomic_write_bytes):

Create temporary files within the same target directory using tempfile.mkstemp().

Flush buffers and explicitly call os.fsync() on file descriptors to force disk persistence.

Execute an atomic rename (os.replace()) to swap the temp file into place instantly.

Guarantee cleanup via try...finally blocks to prevent disk clutter.

IV. Immutable Provenance & The DI Alignment Sentinel
The Vulnerability: Systems drift away from their primary mandate when temporary logic or domain-specific code contaminates core architecture.

The DI Standard: The core/sentinel.py module acts as an automated state anchor:

Encodes non-negotiable DI axioms (domain-agnosticism, absolute determinism, zero drift).

Automatically generates cryptographic sidecars upon initialization.

Validates its own hash and integrity checks prior to executing any workflow.

3. Moment-by-Moment Project Roll-Out & Execution Lifecycle
The methodology described above is not passive documentation; it is actively enforced during every execution cycle. When the framework executes—whether handling a synthetic pre-flight check or a full-scale payload—it follows a strict, non-negotiable operational sequence:

Step 1: Pre-Execution Alignment Verification
Before any payload is loaded or executed, the entrypoint invokes verify_di_alignment().

The system inspects its root workspace for di_alignment_anchor.json and its SHA-256 sidecars.

It recalculates the hash of the anchor file on disk, parses the sidecar defensively, and asserts bit-for-bit equality.

Outcome: If drift, corruption, or manual tampering is detected, execution halts immediately with a structured diagnostic error.

Step 2: Environment and Seed Locking
Once alignment is verified, the execution harness locks down runtime randomness.

Environment flags (TF_DETERMINISTIC_OPS, TF_CUDNN_DETERMINISTIC) are set.

Random number generators for Python, NumPy, and deep learning frameworks are seeded with the user-defined cryptographic seed.

Step 3: Atomic Payload Execution & Manifest Generation
The computational workload executes within the protected shell.

Outputs (predictions, logs, processed data) are written to disk using deterministic serialization and atomic file primitives.

Dual-case cryptographic sidecars (.sha256.txt and .SHA256.TXT) are automatically generated for every output artifact.

Step 4: Final Closure & Audit Signing
A master run manifest is compiled, linking all input files, environment metrics, and output hashes.

The manifest is serialized deterministically, written atomically, and sealed with its own sidecars.

The system returns a structured success payload, guaranteeing that any external test bed or peer reviewer can independently audit and reproduce the exact bit-state of the run.