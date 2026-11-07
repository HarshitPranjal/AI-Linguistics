# STAGE 1 CLOSE-OUT 5 FINAL REPORT & BIT-LEVEL AUDIT
**Multilingual and Code-Mixed LLM Safety Framework (Manifest v3 Frozen)**

- **Execution Timestamp**: 2026-11-06T15:00:00+05:30
- **Git Status**: Clean working tree on `main` branch. **NO TAGS CREATED**.

## 1. SHA-256 File Manifest

| File Path | SHA-256 Checksum |
| :--- | :--- |
| `outputs/stage1/dataset.jsonl` | `e40fdba1de8c95c7ca2835a4c367d87f56c7febaf6785336309b4fa23ab5ea7e` |
| `outputs/stage1/cache/nllb_translations.json` | `e928ed34fdd76ef5f25c07f1704dd3a0f17f0f8cc99fdcc36027d53f3eac0abd` |
| `outputs/stage1/cache/nllb_back_translations.json` | `84376af09dd520e4ddb48ea254712bc9d739757cce2000be5250838485b2f498` |
| `outputs/stage1/seed_split_manifest_v3.json` | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` |
| `outputs/stage1/audit/closeout3_rest.md` | `975dfe5323ef42865acf2d616a189c4c630f1b2565efdee670f4f1209549b69d` |
| `outputs/stage1/audit/mixed_variant_report.md` | `b0e2d120d624fd448df5f363a55272c10a81ad9108e1f8de39e1b80aac2a9cbb` |
| `outputs/stage1/audit/mixed_samples.csv` | `870a8316ff2fe25446e1cdc88d5c722c40b27bc5126313fe6c6fc23855b1b440` |
| `outputs/stage1/audit/mixed_samples.md` | `a658338e7a8f5be3bd54e7c37b536df7e9b6d880a5a4d71060e6aaa4aaca308e` |
| `outputs/stage1/audit/loanword_replacements.csv` | `a8a92b84fe5cfd679a746ae6dbfccc97c66d4a7ead354be8d523fd76e0bbdb36` |
| `outputs/stage1/audit/loanword_check_100.csv` | `75d1f3eba38c3351e3c95d73d8e1fd46073e277f5da2b9afe02bec4471c51fe2` |
| `outputs/stage1/audit/drift_candidates.csv` | `1aa2a8684dfc195b355658306922615727449371b44273646ff2ab5eaf5adfd8` |
| `outputs/stage1/audit/hand_fix_proposals.csv` | `8bab71fe8f34f12112f74b4d52d8cdfabfd47322efb114fc4912c46002fe35ff` |
| `outputs/stage1/audit/regenerated_20_pair_table.md` | `fee2c08a126c158aa401c85102b7ab2d4ed308cb77f9ef2da3400a7be32373ea` |

## 2. Alignment-Based Code-Mixing Quality

- **Stopword Insertions**: Exactly 0 across all 7957 inserted tokens (Assertion PASS).
- **Head-Word Preservation**: Exactly 100.0% preserved.
- **Strict Monotonicity**: 92.9% for `unsafe_request`, 97.0% for `utility_nli`, 90.0% for `self_harm`, 88.0% for `borderline_benign`.
- **In-Band Compliance**: 90.2% for `unsafe_request`, 88.2% for `borderline_benign`, 73.3% for `self_harm`, 55.0% for `utility_nli`.

## 3. Romanization & Loanword Quality

- **Devanagari Relics**: 0 across all 553 seeds (Assertion PASS).
- **Tuned Loanword Threshold**: T = 0.65 on dev subset with gold XNLI native stoplist protection.
- **Review Sample**: 100 replacements exported to `outputs/stage1/audit/loanword_check_100.csv` (`seed=7`).

## 4. Label Drift & Safety Audit

- **Back-Translation Model**: `facebook/nllb-200-distilled-600M` (`hin_Deva` / `bho_Deva` -> `eng_Latn`).
- **P1 Flags**: 16 items in `hi`, 24 items in `bho`.
- **Hand-Fix Proposals**: Limited to P1 `hi` only (16 items); P1 `bho` marked `needs native translation`.
- **Colab Plan**: Re-translation plan documented for NLLB-3.3B (bho) and IndicTrans2 (hi).

## 5. Integrity Verification

- **Close-out 2 Table Errors**: Rows 12 (`dna_537`) and 14 (`dna_568`) reconciled.
- **HarmBench IDs**: Exactly 99 HarmBench seeds in manifest v3 and dataset.jsonl (0 mismatch).
