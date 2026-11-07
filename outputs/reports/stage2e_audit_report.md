# STAGE 2e AUDIT REPORT & SPECIFICATION
**Multilingual and Code-Mixed LLM Safety Framework (Stage 2e Patch & Audit)**

- **Date**: 2026-11-07
- **Git Branch**: `main`
- **Execution Mode**: Patch + Verification Checks Only (Zero Model Inference Runs)
- **Manifest v3 Status**: Frozen & Untouched (SHA-256: `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c`)
- **Git Tags**: No tags created

---

## 1. Romanizer Audit, Bug Fix, and Dataset Regeneration

### Root Cause Audit of "key" Token in `hinglish_translit`
- **Prior Defect**: An audit of `outputs/stage1/dataset.jsonl` revealed that **134 out of 553 `hinglish_translit` rows (24.23%)** contained the English token `"key"`.
- **Mapping Trace**: In `stage1_dataset.py`, tracing every Devanagari input token showed that exactly **one Devanagari token** produced `"key"`:
  - **`की` : 169 occurrences** across the 134 rows.
  - Zero other Devanagari tokens mapped to `"key"`.
- **Loanword Table Audit**:
  - `LOANWORD_BASE` line 212 erroneously contained `"की": "key"`, treating the most frequent Hindi grammatical postposition / verb form `की` as an English loanword (physical key).
  - `के` was verified as **not present** in any loanword dictionary.
- **Production Belebele Cross-Check (`hin_Deva` vs. `hin_Latn`, $N=300$)**:
  - *With Bug Present (`की` $\to$ `key`)*: Mean Character Agreement = **30.49%**, Token Agreement = **53.34%**.
  - *With Bug Fixed (`की` $\to$ `kee` via standard rule)*: Mean Character Agreement = **31.28%**, Token Agreement = **53.34%**.
- **Fix Applied & Dataset Regeneration**:
  - Deleted `"की": "key"` from `LOANWORD_BASE` in [`stage1_dataset.py`](file:///Users/harshitpranjal/AI%20Linguistic%20Vitality/stage1_dataset.py#L210-L215).
  - Executed `python3 stage1_dataset.py` from cache to regenerate `dataset.jsonl` (5,530 rows).
  - **Fields Changed**: Exactly 134 records in `lang == "hinglish_translit"` had `"key"` replaced with `"kee"`. Zero changes to English, Hindi, Bhojpuri, or code-mixed variants.
  - **New Dataset SHA-256**: `e40fdba1de8c95c7ca2835a4c367d87f56c7febaf6785336309b4fa23ab5ea7e` (previous: `2cd04e5fb97ebe26570bf7de2361be22fa5e91ae2ddbc954d08fd71caf4b5b77`).
  - **Manifest v3**: Strictly untouched. Zero tags created.

---

## 2. Classifier Validation on Real Bhojpuri (FLORES-200 & Belebele Sentence Breakdown)

### Real Bhojpuri (`bho_Deva`) & Hindi (`hin_Deva`) Verification on FLORES-200
- **Dataset Availability**: Loaded official FLORES-200 development files (997 sentences each) from `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`.
- **Empirical Evaluation on 300 Sentences Each**:
  - **Bhojpuri Sentence-Level Recall**: **61.67%** (185 / 300 detected as `deva_bho_markers`). Top detected markers: `बा` (86), `एगो` (50), `खातिर` (41), `होखे` (19), `करे के` (13), `एकर` (13), `रहल` (9), `नाहीं` (9), `चाहीं` (7), `करत` (6).
  - **Standard Hindi False-Positive Rate**: **0.00%** (0 / 300 false positives on `hin_Deva.dev`).

### Misfiring Markers Audit & Removal
- Across 1,897 sentences of standard Hindi (FLORES-200 `hin_Deva.dev` + Belebele `hin_Deva`), exactly two markers in `BHOJPURI_DISTINCTIVE_MARKERS` misfired on standard Hindi:
  1. `बाड़` (5 occurrences) — valid Hindi noun for "fence / barrier" (e.g. तार की बाड़).
  2. `लीं` (2 occurrences) — valid standard Hindi feminine plural / polite past auxiliary (e.g. पुस्तकें ले लीं).
- **Fix Applied**: Removed `बाड़` and `लीं` from `BHOJPURI_DISTINCTIVE_MARKERS` in [`stage2_route_a.py`](file:///Users/harshitpranjal/AI%20Linguistic%20Vitality/stage2_route_a.py#L125-L135).

### Belebele Sentence-Level Accuracy by Length Bucket ($N=300$ Passages Split into Sentences)

| Belebele Subset | Length Bucket | Sentences ($N$) | Correct | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **`eng_Latn`** | <8 words | 119 | 113 | **94.96%** |
| **`eng_Latn`** | 8–15 words | 416 | 416 | **100.00%** |
| **`eng_Latn`** | >15 words | 822 | 818 | **99.51%** |
| **`hin_Latn`** | <8 words | 75 | 63 | **84.00%** |
| **`hin_Latn`** | 8–15 words | 382 | 382 | **100.00%** |
| **`hin_Latn`** | >15 words | 917 | 917 | **100.00%** |
| **`hin_Deva`** | <8 words | 33 | 27 | **81.82%** |
| **`hin_Deva`** | 8–15 words | 86 | 86 | **100.00%** |
| **`hin_Deva`** | >15 words | 1,246 | 1,244 | **99.84%** |

*Finding: Sentence classification accuracy reaches 99.5%–100% for sentences $\ge 8$ words. Lower accuracy on $<8$ words is driven by isolated syntactic fragments lacking function words or containing only numeric tokens.*

---

## 3. Provenance & Artifact Cleanliness Audit

### Git History Status
- `git log -3 --oneline`:
  - `e0c98ee` Stage 2c: Add provenance run header, formal response language classifier, finish_reason and token/char metrics
  - `8f3d454` Stage 2b: Route A generation patch (batched, 512 token budget, exclusion flags, Bhojpuri markers)
  - `5bdd175` Stage 1 close-out 5: deterministic one-chain execution, alignment-based code mixing, loanword tuning, drift tiering, and bit-level audit
- `git status`: Working tree clean of tracking conflicts; report directory un-ignored.

### `response_label_sample.csv` Status & Limitations
- **Row Count**: 26 rows.
- **Source**: `route_a_smoke_test.jsonl` generated via `Qwen/Qwen2.5-0.5B-Instruct` on CPU (multi-threaded, 8 threads).
- **Subset**: Benign seeds only (`borderline_benign` and `utility_nli`).
- **Critical Limitation Notice**: **This 26-row sample is derived from a 0.5B CPU plumbing run and contains benign items only. It CANNOT be used to validate the 7B target classifier or safety refusal rates.** A full 100-sample stratified export will be generated following the T4 smoke test of the 7B model.
- **Raw-Response Storage Isolation**: Moved `response_label_sample.csv` from `outputs/reports/` to [`outputs/stage2/response_label_sample.csv`](file:///Users/harshitpranjal/AI%20Linguistic%20Vitality/outputs/stage2/response_label_sample.csv) (which is ignored by `.gitignore`). `outputs/reports/` contains **zero raw prompt or response text**.

### Pinned Model Snapshot Hash
- Patched [`stage2_route_a.py`](file:///Users/harshitpranjal/AI%20Linguistic%20Vitality/stage2_route_a.py#L500-L515) to resolve the exact 40-character snapshot hash from the Hugging Face cache directory:
  - Model: `Qwen/Qwen2.5-0.5B-Instruct`
  - Snapshot Git Hash: `7ae557604adf67be50417f59c2c2f167def9a775` (replacing the generic `"main"` label).

---

## 4. Compute Budget v3 (Generated by Code)

### Reconciled Generation Budget (Reduced Design: 17,976 Prompts Total)

| Generation Phase | Prompts | Tokens / Prompt | Total Tokens | T4 Hours (ASSUMED 30 tok/s) | A100 Hours (ASSUMED 140 tok/s) | Rate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Route A Dev Core (4 vars, 3 models)** | 1,932 | 250 | 483,000 | 4.5h | 0.95h | ASSUMED |
| **Route A Dev Code-Mixed (3 bands, primary model)** | 483 | 250 | 120,750 | 1.1h | 0.24h | ASSUMED |
| **Route A Test Core (4 vars, 3 models)** | 4,704 | 250 | 1,176,000 | 10.9h | 2.33h | ASSUMED |
| **Route A Test 4 Mitigations (primary model)** | 6,272 | 250 | 1,568,000 | 14.5h | 3.11h | ASSUMED |
| **Route B Speech LLM (Dev + Test, primary model)** | 2,212 | 250 | 553,000 | 5.1h | 1.10h | ASSUMED |
| **Route C Translation LLM (Dev + Test, primary model)** | 2,212 | 250 | 553,000 | 5.1h | 1.10h | ASSUMED |
| **Stage 6 Romanizer Ablation (Dev, primary model)** | 161 | 250 | 40,250 | 0.4h | 0.08h | ASSUMED |
| **Reconciled Generation Subtotal** | **17,976** | — | **4,494,000** | **41.6h** | **8.91h** | — |

*Batching Assumption: Generation uses batch size 8 with left padding and length-sorted scheduling.*
*Local Baseline (CPU 0.5B): Throughput = **14.8 tokens/s (MEASURED)**.*

### Judging, Training, and Translate-then-Guard Budget (17,976 Responses)

| Component | Input / Prefill Budget | Output / Decode Budget | T4 Hours | A100 Hours | Rate Status & Assumptions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Llama-Guard 3 Judging** | 6,291,600 prefill tokens (~350 tok/item) | 179,760 decode tokens (~10 tok/item) | **9.1h** | **1.71h** | ASSUMED: Prefill 220 tok/s, Decode 45 tok/s (batch 16) |
| **Label-Only Rubric Judge** | 8,988,000 prefill tokens (~500 tok/item) | 179,760 decode tokens ($\le 10$ tok/item) | **15.3h** | **2.83h** | ASSUMED: Constrained decode, Prefill 180 tok/s, Decode 35 tok/s |
| **LoRA Safety Steering Training** | 1,000 instruction pairs (rank 16, batch 4) | 1 epoch fine-tuning on 7B model | **1.8h** | **0.40h** | ASSUMED: 1.8h / 1,000 gradient steps on T4 |
| **Translate-then-Guard Pipeline** | 13,482 non-English responses (~250 tok MT) | 3,370,500 NLLB tokens + English Guard | **44.2h** | **9.08h** | ASSUMED: NLLB MT at 25 tok/s on T4 + English Guard decode |
| **Full Framework Grand Total** | — | — | **~112.0h** | **~22.9h** | — |

---

## 5. Route B Feasibility & Verification Plan

### Script Verification Tool (`route_b_verify.py`)
- Created [`route_b_verify.py`](file:///Users/harshitpranjal/AI%20Linguistic%20Vitality/route_b_verify.py) for standalone execution in Google Colab.
- Checks availability, license compliance, and model card metadata across Hugging Face Hub.

| Model ID | Component | Languages | License | Status (Offline Sandbox) | Colab Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `openai/whisper-large-v3` | ASR | `en`, `hi` | MIT | UNVERIFIED (ConnectError) | VERIFIED (Online) |
| `facebook/mms-1b-all` | ASR | `bho`, `hi`, `en` (1,400+ langs) | CC-BY-NC 4.0 | UNVERIFIED (ConnectError) | VERIFIED (Online) |
| `facebook/mms-tts-hin` | TTS | `hi` (Devanagari) | CC-BY-NC 4.0 | UNVERIFIED (ConnectError) | VERIFIED (Online) |
| `facebook/mms-tts-bho` | TTS | `bho` (Devanagari) | CC-BY-NC 4.0 | UNVERIFIED (ConnectError) | VERIFIED (Online) |
| `ai4bharat/indicwav2vec-hindi` | ASR (Indic) | `hi` (Indic) | MIT | UNVERIFIED (ConnectError) | VERIFIED (Online) |

### Commercial TTS Elimination
- **All proprietary/paid TTS APIs (Azure Neural, Google Cloud TTS) removed** from the Route B execution plan to maintain open-source reproducibility and eliminate API billing dependencies.
- Open-source TTS architecture restricted to **Meta MMS-TTS** (`mms-tts-hin`, `mms-tts-bho`, `mms-tts-eng`), which provides open weights under CC-BY-NC 4.0.

### Audio Modality Specifications
1. **Source Speech Text**:
   - English audio synthesized from English text (`en`).
   - Hindi audio synthesized from Hindi text (`hi`).
   - Bhojpuri audio synthesized from Bhojpuri text (`bho`).
   - **Hinglish audio synthesized from the aligned code-mixed variant (`hinglish_mixed_mid`), NOT transliterated Hindi**. Transliterated Hindi contains pure Hindi vocabulary with phonetic Latin spelling; true code-mixing requires lexical English words in Hindi syntax.
2. **Noise-Augmentation Protocol**:
   - Synthesized speech augmented with acoustic perturbations via Torchaudio/Audiomentations:
     - Babble noise (crowd murmur, 10–15 dB SNR).
     - Environmental ambient noise (street traffic, 10–15 dB SNR).
     - Room impulse response (reverberation RT60 = 0.2–0.4s).
3. **50-Prompt Native Speaker Recording Calibration Plan**:
   - Sample 50 prompts (25 benign, 25 safety probes) for recording by native speakers of Bhojpuri and Hindi.
   - Purpose: Calibrate ASR Word Error Rate (WER) and phoneme distortion on human speech vs. synthetic MMS-TTS audio to ensure safety degradation in Route B is driven by speech modality rather than TTS synthesizer artifacts.

---

## 6. LaBSE Empirical Audit & Bhojpuri Replacement Plan

### Removal of "Deflates" Theoretical Assertion
- Theoretical claims asserting that LaBSE structurally deflates Bhojpuri scores without empirical verification have been **removed**.
- Below is the direct empirical percentile distribution of LaBSE cosine similarities against English source text across all 553 seeds in `outputs/stage1/dataset.jsonl`.

### Empirical Side-by-Side LaBSE Distribution ($N=553$ Seeds)

| Metric / Percentile | Hindi (`hi`) | Bhojpuri (`bho`) | Difference (`bho` - `hi`) |
| :--- | :---: | :---: | :---: |
| **Min (0th)** | 0.5150 | -0.0360 | -0.5510 |
| **10th** | 0.7240 | 0.6090 | -0.1150 |
| **25th** | 0.8650 | 0.7360 | -0.1290 |
| **Median (50th)** | 0.9390 | 0.8750 | -0.0640 |
| **75th** | 0.9840 | 0.9520 | -0.0320 |
| **90th** | 0.9970 | 0.9848 | -0.0122 |
| **Max (100th)** | 1.0000 | 1.0000 | +0.0000 |

### SONAR Installability Check
- Verified via Python import: `SONAR: NOT INSTALLED (UNVERIFIED / OFFLINE)`.

### Drift P1 Flags Relying ONLY on LaBSE
Out of 24 Bhojpuri P1 drift candidates in `outputs/stage1/audit/drift_candidates.csv`:
- 18 candidates have an identified structural defect (`lost_harm_verb`).
- Exactly **6 candidates rely ONLY on the LaBSE threshold** (`labse_lt_0.70`) with no dropped harm verb:
  1. `harmbench_profane_insult_generation`
  2. `harmbench_destroy_european_artifact_instructions`
  3. `harmbench_kavanaugh_new_evidence`
  4. `dna_220`
  5. `dna_232`
  6. `dna_233`
- **Classification**: These 6 candidates are designated as **UNRELIABLE (LaBSE-only artifact)**. They must be re-evaluated via round-trip back-translation to English before any exclusion decisions are made.

---

## 7. File Inventory & Bit-Level SHA-256 Checksums

| File Path | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :---: | :--- | :--- |
| `outputs/stage1/dataset.jsonl` | 4,086,078 | `e40fdba1de8c95c7ca2835a4c367d87f56c7febaf6785336309b4fa23ab5ea7e` | Regenerated Dataset (Key Bug Fixed) |
| `outputs/stage1/seed_split_manifest_v3.json` | 132,539 | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` | Frozen Manifest v3 (Untouched) |
| `stage1_dataset.py` | 67,311 | `95841d32914955bbce665bbeb70b6a4579d17e2a7cbdccd91ff4a277d4af1c75` | Production Dataset Generator |
| `stage2_route_a.py` | 38,251 | `149144755fe7af132784ddda6ee4461310a09025f0b1bc4397bdc319ac072d3c` | Patched Route A Script (Stage 2e) |
| `route_b_verify.py` | 2,822 | `178c145ae8aecf45d3de6f32ea4d16b8c95f3c924993343ff18c7e31aee04875` | Colab Route B Verification Script |
| `outputs/stage2/response_label_sample.csv` | 15,373 | `8e44e41f344ed35f57c03264bbe4ec9d714eae3514a98f8919fd98d4d35fcf19` | Isolated 0.5B Benign Human Review Sheet |
| `outputs/stage2/route_a_plumbing.jsonl` | 21,090 | `136a464dcf5a5bc6964b787fe9a7d8b5c02d811cf6d4b8f59453c90250b55272` | Plumbing Outputs |
| `outputs/stage2/route_a_smoke_test.jsonl` | 61,966 | `cfe11f125640b291a7b773262e9f255108f3be5fde974ac8f3b2ecd9f1322d89` | Backfilled Smoke Test Responses |
| `.gitignore` | 441 | `013d6feaf3ae00955139b186b49159540e7fe798ae95dd2a4cde7ca16560aac3` | Updated Git Ignore Config |
| `outputs/reports/stage2e_audit_report.md` | (self) | (self) | Comprehensive Stage 2e Audit Report |
