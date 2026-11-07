# STAGE 2h AUDIT REPORT: FINAL LOCAL DATA CLEANUP & SCHEMA CONVERGENCE
**Multilingual and Code-Mixed LLM Safety Framework (Manifest v3 Frozen)**

- **Execution Timestamp**: 2026-11-07T18:42:00+05:30
- **Git Branch**: `main`
- **Git Status**: Clean working tree; **NO TAGS CREATED**.

---

## 1. SHA-256 Checksum Manifest

| File Path | Records / Size | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| `outputs/stage1/seed_split_manifest_v3.json` | 553 seeds | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` | **FROZEN** |
| `outputs/stage1/archive/deprecated_hinglish_mixed.jsonl` | 1,659 rows | `023d7426bcdc5baf9146a009ebcc2e6e96aee971c88d2523f59a7c203c0903b8` | **ARCHIVED** |
| `outputs/stage1/dataset.jsonl` | 4,424 rows | `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf` | **CURRENT** |
| `stage1_dataset.py` | 1,289 lines | `fc28ae722af5be03e390bf51597d9df9f05103a249e3af54cb021c0a41bf519a` | **COMMITTED** |
| `stage2_route_a.py` | 924 lines | `308dfc6e1bb8832355e8e41ad128533c024f93968abdf8de3f01aa189e0afc02` | **COMMITTED** |
| `outputs/stage1/cache/nllb_translations.json` | 1,106 items | `e928ed34fdd76ef5f25c07f1704dd3a0f17f0f8cc99fdcc36027d53f3eac0abd` | **CACHED** |
| `outputs/stage1/cache/nllb_back_translations.json` | 1,106 items | `84376af09dd520e4ddb48ea254712bc9d739757cce2000be5250838485b2f498` | **CACHED** |

---

## 2. Archive of Deprecated Rows

- **Source Commit**: `5bdd175` ("Stage 1 close-out 5: deterministic one-chain execution, alignment-based code mixing, loanword tuning, drift tiering, and bit-level audit")
- **Row Count**: Exactly 1,659 records (553 seeds $\times$ 3 synthetic bands: `low`, `mid`, `high`)
- **Original Metadata Restored**: `qc_scores` with `length_ratio` and `labse_cosine`
- **Destination Path**: `outputs/stage1/archive/deprecated_hinglish_mixed.jsonl`
- **Archive SHA-256**: `023d7426bcdc5baf9146a009ebcc2e6e96aee971c88d2523f59a7c203c0903b8`

---

## 3. Active Dataset Determinism & Reproducibility Chain

- **Target Row Count**: Exactly 4,424 records (553 seeds $\times$ 8 active variants: `en`, `hi`, `bho`, `hinglish_translit`, `hinglish_translit_formal`, `hinglish_mixed_low`, `hinglish_mixed_mid`, `hinglish_mixed_high`)
- **Temp Run 1 SHA-256**: `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf`
- **Temp Run 2 SHA-256**: `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf`
- **Committed File SHA-256**: `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf`
- **Byte-Identity**: 100% byte-identical across all three runs (`Run 1 == Run 2 == Committed`).
- **Fixed `qc_scores` Key Order**: Exactly `["labse_cosine", "length_ratio", "is_qc_passed", "labse_direct", "labse_bt"]` across all 4,424 records.
- **Dynamic Empirical `length_ratio`**: Computed dynamically via `round(len(variant_text) / max(1, len(en_text)), 3)` across all records with zero hardcoded values (en = 1.000).

---

## 4. Verification & Consistency Audit

- **Active Text Changes vs `84533fba...`**: Exactly **0** changes (0 / 4,424).
- **`(seed_id, lang)` Uniqueness**: 4,424 distinct pairs across 4,424 records (0 duplicates). Asserted in `stage1_dataset.py` and `stage2_route_a.py`.
- **Status Filtering**: All 4,424 rows have `status: "ACTIVE"`. Excluded any `status != "ACTIVE"` in `stage2_route_a.py`.
- **Resumable Key**: `(seed_id, lang, model)` with explicit error assertion on duplicates in `stage2_route_a.py`.

---

## 5. Plumbing Test Verification (0.5B CPU)

- **Model ID**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Sampled Benign Dev Seeds**: `xstest_v2-157`, `xstest_v2-109`, `xstest_v2-260`
- **Runtime Provenance Check**: Dataset SHA-256 `e599106a...` (`VERIFIED`)
- **Total Records Generated**: 12
- **Overall Status**: **PASS** (`[Plumbing Test Status] PASS: Patched schema verified successfully.`)
