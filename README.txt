README v1.2 Deterministic Core Shell (DI): Technical Specification & Architecture
1. Architectural Overview
The Deterministic Core Shell (DI) is a domain-agnostic execution harness engineered to eliminate non-deterministic behavior, race conditions, environmental drift, and metadata corruption across computational workflows.

Traditional execution pipelines suffer from platform-dependent serialization variances, partial writes under concurrent load, and unverified runtime environments. DI replaces ad-hoc scripting with an audit-proof, byte-level deterministic control plane that enforces strict cryptographic provenance and mandatory pre-flight alignment validation prior to workload execution.

2. Core Technical Pillars
A. Byte-Level Determinism
To prevent cross-platform serialization drift (e.g., dictionary key reordering or whitespace variances across Python versions), all internal object serialization adheres to a strict canonical byte specification:

JSON Normalization: Enforced via json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False).

Encoding: Explicit UTF-8 byte emission (.encode("utf-8")).

B. Atomic Persistence & Concurrency Safety
File write operations are vulnerable to partial writes, race conditions, and disk synchronization gaps. DI guarantees atomicity through a transactional four-step file replacement primitive:

Isolated Allocation: Temporary files are generated within the target directory via tempfile.mkstemp(dir=dirpath).

Buffer Flushing: Data is written and explicitly pushed to the OS buffer via f.flush().

Descriptor Synchronization: Low-level synchronization is forced via os.fsync(f.fileno()) to guarantee physical disk persistence.

Atomic Pointer Swapping: The temporary file replaces the target path atomically via os.replace(), eliminating intermediate corrupt states.

C. Cryptographic Provenance & Dual-Case Sidecars
Every artifact, manifest, and workspace anchor is cryptographically bound using SHA-256 hashing.

Recursive & Single-File Hashing: Supports efficient chunked reading (8KB buffers) for large files and sorted path traversal for directory structures.

Dual-Case Sidecar Seeding: Automatically generates both lowercase (.sha256.txt) and uppercase (.SHA256.TXT) sidecar verification files to accommodate cross-platform file system convention differences.

D. Environmental Seeding & Hardware Locking
Non-determinism introduced by runtime libraries is neutralized at initialization:

Environment Variable Locking: Automatically forces TF_DETERMINISTIC_OPS=1, TF_CUDNN_DETERMINISTIC=1, and PYTHONHASHSEED.

Multi-Engine Seeding: Synchronizes random state seed values across Python's random, NumPy (np.random.seed), and TensorFlow (tf.random.set_seed).

Thread Throttling: Restricts TensorFlow intra-op and inter-op parallelism threads to 1 where applicable to prevent non-deterministic multi-threaded scheduling variance.

3. Control Plane Components
Xanthommatin/
├── core/
│   ├── harness.py    # Utility plane (atomic I/O, hashing, environment locking)
│   └── sentinel.py   # Alignment gate (workspace anchoring & cryptographic verification)
├── tests/
│   └── test_di.py    # Adversarial regression suite
├── cli.py            # Top-level execution control plane
└── pyproject.toml
core/sentinel.py: Manages workspace alignment anchoring (di_alignment_anchor.json). Acts as an absolute gatekeeper; if an anchor is missing, corrupted, or cryptographically invalidated, execution halts immediately.

core/harness.py: Provides the foundational utility plane managing deterministic serialization, atomic I/O, hash computation, and environment seeding.

cli.py: The unified entrypoint routing subcommands (init, verify, run) with structured logging and pre-flight sentinel enforcement.

4. Adversarial Regression Suite (tests/test_di.py)
The system is continuously validated against an automated adversarial test harness covering:

Key-Ordering Invariance: Asserts identical byte output across randomized input dictionary structures.

Atomic Persistence: Validates zero partial write residuals and clean temporary file disposal.

Tamper Resistance: Asserts that manual mutation of anchor files or output artifacts triggers immediate cryptographic checksum mismatches.

Defensive Parsing: Asserts graceful handling of malformed, empty, or truncated sidecar files without unhandled runtime exceptions.

5. Operational Reference
Initialize Workspace Anchor
Generates the foundational DI workspace alignment anchor and dual-case cryptographic sidecars:

Bash
python cli.py init --workspace .
Verify Alignment Integrity
Performs pre-flight cryptographic verification against workspace sidecars:

Bash
python cli.py verify --workspace .
Execute Sandboxed Payload
Enforces sentinel pre-flight verification, locks environment seeds, dispatches the payload, and seals the output within an immutable execution manifest:

Bash
python cli.py run --payload payload_script.py --seed 42


**********************
python cli.py init --workspace .
python cli.py verify --workspace .
python cli.py run --payload payload_script.py --seed 42