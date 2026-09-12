Markdown
# ==============================================================================
# File Name: README_1.3.md
# File Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\Xanthommatin\READMEs\README_1.7.md
# Date: September 11, 2026
# Time: 11:46 AEST
# Author: Brian Lowe
# Copyright: Brian Lowe. (c) 2026 All rights reserved.
# 
# Description:
# Master architectural standard defining documentation protocols, semantic 
# container naming rules, living audit trails, and append-only ledger management 
# for the Xanthommatin framework.
# ==============================================================================

**README v1.7 XANTHOMMATIN DOCUMENTATION AND NAMING STANDARD**

### 1. Core Philosophy: Anti-Drift Documentation
The Xanthommatin framework rejects decoupled, static, and generic documentation practices. To eliminate conversational, structural, and semantic drift across multi-step development cycles, all documentation and code assets must adhere to strict determinism, self-documentation, and semantic precision.

---

### 2. Semantic Naming Protocols & Versioning
Generic labels (`engine.py`, `utils.py`) are strictly prohibited across the repository. Every asset must function as a domain-specific semantic container.

* **Filename Versioning:** Core executable modules, master charters, and major release standards must embed explicit versioning directly in the filename and path (e.g., `ommatidium_v1.1.py`, `README_1.7.md`).
* **Header Metadata Block:** Every core asset must open with an immutable metadata header detailing:
  * Absolute File Name and Path
  * Creation/Update Timestamp (Date and Time AEST)
  * Author & Copyright Notice
  * Descriptive Summary & Development Audit Trail
* **The Version Redundancy Rule:** Internal file headers must **omit** standalone `Version:` fields. The filename and path serve as the absolute, immutable source of truth for version state, preventing double-bookkeeping conflicts.

---

### 3. The Living Audit Trail (Inline Documentation)
To prevent structural amnesia during refactoring, evolutionary history and error taxonomy mappings are bound directly into the execution body:

* **Header-Level Tracking:** Major refactors and version transitions are chronicled chronologically in the module's opening comment block.
* **Taxonomy Linkage:** Docstrings and comments explicitly reference how specific methods mitigate known failure modes (e.g., recency bias, semantic over-generalization).

---

### 4. Immutable Modules vs. Dynamic Living Ledgers
To balance deterministic tracking with operational practicality, the framework enforces a two-tier file management strategy:

* **Immutable Core Modules:** Domain-specific names with explicit version suffixes (e.g., `ommatidium_v1.1.py`, `README_1.7.md`). Versions increment on breaking logic or structural rewrites.
* **Dynamic Living Ledgers:** Stable, non-versioned semantic names with internal sequential tracking (e.g., `ERROR_TAXONOMY_LOG.md`). The filename remains stable to prevent broken links; schema versions only increment upon major structural format overhauls.

---

### 5. Summary of Processed Error Signatures
All components must account for the primary failure modes cataloged in the framework's error processing pipeline:

1. **Recency-Inversion Glitch:** Over-indexing on trailing conversational keywords; mitigated by immutable filename anchors and attention-dampening weights.
2. **Semantic Over-Generalization:** Interpreting targeted node edits as blanket rewrites; mitigated by granular vector-diff baselines.
3. **Boilerplate Bias:** Dropping mandatory metadata during modality switches; mitigated by enforced header templates.
4. **Generic Label Regression:** Defaulting to uninformative labels; mitigated by domain-specific semantic container mandates globally.