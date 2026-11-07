# STAGE 1 CLOSE-OUT 5 FINAL REPORT & BIT-LEVEL AUDIT
**Multilingual and Code-Mixed LLM Safety Framework (Manifest v3 Frozen)**

- **Execution Timestamp**: 2026-11-06T15:00:00+05:30
- **Git Status**: Clean working tree on `main` branch. **NO TAGS CREATED**.

## 1. SHA-256 File Manifest

| File Path | SHA-256 Checksum |
| :--- | :--- |
| `outputs/stage1/dataset.jsonl` | `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf` |
| `outputs/stage1/cache/nllb_translations.json` | `e928ed34fdd76ef5f25c07f1704dd3a0f17f0f8cc99fdcc36027d53f3eac0abd` |
| `outputs/stage1/cache/nllb_back_translations.json` | `84376af09dd520e4ddb48ea254712bc9d739757cce2000be5250838485b2f498` |
| `outputs/stage1/seed_split_manifest_v3.json` | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` |
| `outputs/stage1/audit/closeout3_rest.md` | `94931f22655425824e9a5248788d98d2f310f66460641bbbf2394590c887c182` |
| `outputs/stage1/audit/mixed_variant_report.md` | `b0e2d120d624fd448df5f363a55272c10a81ad9108e1f8de39e1b80aac2a9cbb` |
| `outputs/stage1/audit/mixed_samples.csv` | `294f751d459c1b3d126a607fb7d3a9d63eb8f4367a621e149c0764dd92b0c563` |
| `outputs/stage1/audit/mixed_samples.md` | `c8672dc6df0e371a0179502eb555e102f3260337bd4473408707437287591069` |
| `outputs/stage1/audit/loanword_replacements.csv` | `fc6b1edbe73311c9a88ba5cd8145a275ff46f228df2567eaaf7f8c6f7a4a0f37` |
| `outputs/stage1/audit/loanword_check_100.csv` | `bef1ba0149563b9796afab003dcaf2351a652ca9956bf065b297289025b8b891` |
| `outputs/stage1/audit/drift_candidates.csv` | `1aa2a8684dfc195b355658306922615727449371b44273646ff2ab5eaf5adfd8` |
| `outputs/stage1/audit/hand_fix_proposals.csv` | `583bfc6f3f07d50d142729fd5cd50880ba40ed940fdaf56fe0df634018e56f6c` |
| `outputs/stage1/audit/regenerated_20_pair_table.md` | `70f4ad60ab0c84a2e7f8600c2fae19e8fe7c36906a04edcc82d9abcf951eb780` |

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
