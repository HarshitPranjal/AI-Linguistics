# STAGE 2d AUDIT REPORT & SPECIFICATION
**Multilingual and Code-Mixed LLM Safety Framework (Stage 2d Patch & Plan)**

- **Date**: 2026-11-07
- **Git Branch**: `main`
- **Tracked Reports Directory**: `outputs/reports/` (un-ignored in `.gitignore`)
- **Execution Mode**: Patch + Plan Only (Zero unsanctioned model runs)

---

## 1. Response Language Classifier: Belebele Validation & Class Renaming

### Canonical Class Renaming & Backward-Compatible Aliases
Classes have been renamed to avoid conflating standard script with regional language identity:
- `deva_hindi` $\to$ **`deva_no_bho_markers`**
- `deva_bhojpuri_markers` $\to$ **`deva_bho_markers`**
- Backward-compatible alias map `CLASS_ALIASES` maintained in `stage2_route_a.py`.

### Empirical Validation on 900 Belebele Passages (300 Passages Each)
Evaluated on `flores_passage` from `facebook/belebele` (`eng_Latn`, `hin_Latn`, `hin_Deva`):

| Belebele Subset (N=300 each) | Gold Class | `latin_english` | `latin_romanized_hindi` | `deva_no_bho_markers` | `deva_bho_markers` | `mixed` | `other` | Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eng_Latn`** | `latin_english` | 300 | 0 | 0 | 0 | 0 | 0 | **100.0%** |
| **`hin_Latn`** | `latin_romanized_hindi` | 0 | 300 | 0 | 0 | 0 | 0 | **100.0%** |
| **`hin_Deva`** | `deva_no_bho_markers` | 0 | 0 | 297 | 3* | 0 | 0 | **99.0%** |

*\*Note: The 3 `hin_Deva` passages categorized as `deva_bho_markers` contain standard Hindi homographs of Bhojpuri marker tokens: passages 7 and 8 contain standard Hindi 'बाड़' (fence/barrier), and passage 229 contains polite imperative 'लीं' (take/accept).*

### Human Review Sampling Export
- Function `export_response_label_sample()` added to `stage2_route_a.py` (`--mode export_sample`).
- Stratifies benign dev records evenly across variants (`en`, `hi`, `bho`, `hinglish_translit`).
- Emitted to tracked file: `outputs/reports/response_label_sample.csv`.
- Columns: `seed_id, variant, model, token_count, finish_reason, response_language_class, human_class, notes, response` (with blank `human_class` and `notes` for human labeling).

---

## 2. Romanizer Cross-Check: Top 30 Token Mismatches & Stage 6 Plan

### Top 30 Mismatches from 300 Belebele Sentences (Ours vs. `hin_Latn`)

| Rank | Ours (`natural_romanize`) | Belebele Gold (`hin_Latn`) | Count | Rule Responsible |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `key` | `ke` | 246 | Matra transliteration convention (`key` -> `ke`/`ki`) |
| 2 | `karane` | `karne` | 179 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 3 | `lie` | `liye` | 159 | Semi-vowel glide insertion (`y`-glide before `e`/`i`) |
| 4 | `key` | `ki` | 157 | Matra transliteration convention (`key` -> `ke`/`ki`) |
| 5 | `bhee` | `bhi` | 109 | Long vowel simplification (`ee` -> `i`) |
| 6 | `ya` | `yaa` | 75 | Terminal vowel lengthening (`ya` -> `yaa`) |
| 7 | `hee` | `hi` | 71 | Long vowel simplification (`ee` -> `i`) |
| 8 | `apane` | `apne` | 61 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 9 | `karate` | `karte` | 57 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 10 | `sabase` | `sabse` | 56 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 11 | `sath` | `saath` | 56 | Stem long vowel representation (`a` -> `aa`) |
| 12 | `kisee` | `kisi` | 49 | Long vowel simplification (`ee` -> `i`) |
| 13 | `kuchh` | `kuch` | 47 | Aspirate de-aspiration convention (`chh` -> `ch`) |
| 14 | `kaee` | `kayee` | 41 | Semi-vowel glide insertion (`y`-glide before `e`/`i`) |
| 15 | `vale` | `waale` | 37 | Labiodental semivowel convention (`v` -> `w`) + long `aa` |
| 16 | `naheen` | `nahin` | 34 | Long vowel simplification (`ee` -> `i`) |
| 17 | `apanee` | `apni` | 33 | Long vowel simplification (`ee` -> `i`) |
| 18 | `bare` | `baare` | 33 | Stem long vowel representation (`a` -> `aa`) |
| 19 | `bad` | `baad` | 32 | Stem long vowel representation (`a` -> `aa`) |
| 20 | `dvara` | `dwara` | 30 | Labiodental conjunct convention (`dv` -> `dw`) |
| 21 | `jane` | `jaane` | 30 | Stem long vowel representation (`a` -> `aa`) |
| 22 | `unake` | `unke` | 30 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 23 | `sabhee` | `sabhi` | 27 | Long vowel simplification (`ee` -> `i`) |
| 24 | `shuroo` | `shuru` | 27 | Long vowel simplification (`oo` -> `u`) |
| 25 | `karana` | `karna` | 25 | Medial schwa syncope (internal unstressed `-a-` deletion) |
| 26 | `pahale` | `pehle` | 25 | Intervocalic `-aha-` syncope & metathesis (`aha` -> `eh`) |
| 27 | `vale` | `vaale` | 23 | Labiodental semivowel convention (`v` -> `w`) + long `aa` |
| 28 | `jata` | `jaataa` | 22 | Stem long vowel representation (`a` -> `aa`) |
| 29 | `koee` | `koi` | 22 | Long vowel simplification (`ee` -> `i`) |
| 30 | `gae` | `gaye` | 21 | Semi-vowel glide insertion (`y`-glide before `e`/`i`) |

### Proposed Fixes & Stage 6 Ablation Design (Plan Only)
1. **Underlying Linguistic Differences**:
   - Our Stage 1 transliterator uses formal, unambiguous IAST-based reversible mappings (`ee` for `ी`, `oo` for `ू`, retaining morphological vowels `karane`).
   - Belebele (IndicXlit) uses colloquial, phonetically lossy conversational spellings with extensive schwa syncope (`karne`, `apne`), informal vowel shortening (`bhi`, `sabhi`), and glide insertions (`liye`).
2. **Stage 6 Ablation Specification**:
   - **Do NOT regenerate the frozen Stage 1 dataset**.
   - Create an experimental transliteration pipeline `hinglish_colloquial` implementing:
     - Rule 1: Medial schwa syncope in open syllables $VC\_CV$.
     - Rule 2: High vowel shortening (`ee` $\to$ `i`, `oo` $\to$ `u`).
     - Rule 3: Hiatus glide resolution (`ie` $\to$ `liye`, `ae` $\to$ `aye`).
     - Rule 4: Matra correction (`key` $\to$ `ke` / `ki`).
   - Evaluate LLM refusal rate differences between `hinglish_translit` (formal) and `hinglish_colloquial` (informal) on the dev set to measure whether colloquial spelling degrades safety alignment.

---

## 3. Provenance & History Audit

### `route_a_smoke_test.jsonl` Modification Provenance
- **Original Run (Stage 2b)**: 50 prompt responses generated on CPU with `Qwen2.5-0.5B-Instruct` (file size: 48,350 bytes; SHA-256: `31dda56016ea4e86e9eb00df1c7b684172e95d257c8bf46c24f7707d3fd0ed87`).
- **Stage 2c Update Mechanism**: **Backfill in-place**, **NOT a rerun**.
  - Script read the existing 50 records without invoking the neural model.
  - Kept generated text, prompt hash, and latencies identical.
  - Added new schema fields: `finish_reason` ("length" if `token_count >= 256` else "eos"), `generated_chars` (`len(response)`), `chars_per_token`, and reclassified `response_language_class`.
  - Expanded file size to 61,966 bytes (SHA-256: `cfe11f125640b291a7b773262e9f255108f3be5fde974ac8f3b2ecd9f1322d89`).

### Generation Output `finish_reason` Inspection
- Patched `generate_batch()` in `stage2_route_a.py`:
  - Inspects `raw_gen_ids` directly for the presence of any valid `eos_token_id`.
  - If EOS is present: `finish_reason = "eos"`.
  - If EOS is absent and `token_count >= max_new_tokens`: `finish_reason = "length"`.

### Pinned Model Commit Hash
- Resolved commit hash is retrieved via `model.config._commit_hash` upon loading.
- Pinned in run provenance header (`Model Commit Hash: main`) and recorded per response entry as `model_commit_hash`.

### Tracked Reports Directory
- Updated `.gitignore` to un-ignore `outputs/reports/` (`!outputs/reports/` and `!outputs/reports/**`). All audit reports and human review sheets are tracked under version control.

---

## 4. Compute Budget v2: Generation, Audio, Mitigations, and Judging

### Full Unreduced Design vs. Reduced Feasibility Design
*Note: Every throughput rate is marked **ASSUMED** until measured empirically on the Colab T4 GPU.*

- **Assumed Rates**:
  - Model Text Generation (NF4 4-bit, batch 8): **ASSUMED 30.0 tokens/s** (T4), **ASSUMED 55.0 tokens/s** (Kaggle 2×T4), **ASSUMED 140.0 tokens/s** (A100).
  - Llama-Guard 3 Judging (8B, ~10 tokens output): **ASSUMED 45.0 tokens/s** (T4), **ASSUMED 200.0 tokens/s** (A100).
  - Rubric Judge Evaluation (7B/8B, ~150 tokens output): **ASSUMED 30.0 tokens/s** (T4), **ASSUMED 140.0 tokens/s** (A100).
  - Route B Audio (TTS + ASR): **ASSUMED 1.8 s / prompt** (T4).
  - Route C MT (NLLB-200 translation): **ASSUMED 25.0 tokens/s** (T4).

### Full Framework Breakdown (3 Models: Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B)

| Experiment Phase | Variants | Prompts / Model | Total Prompts | Total Tokens | Colab T4 (Hrs) | Kaggle 2×T4 (Hrs) | Colab A100 (Hrs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Route A Dev (Core 4)** | 4 | 644 | 1,932 | ~483,000 | 4.5h | 2.4h | 0.95h |
| **Route A Dev Code-Mixed (3 Bands)** | 3 | 483 | 1,449 | ~362,250 | 3.4h | 1.8h | 0.72h |
| **Route A Test Core (4 Vars)** | 4 | 1,568 | 4,704 | ~1,176,000 | 10.9h | 5.9h | 2.33h |
| **Route A Test 4 Mitigations** | 4 | 6,272 | 18,816 | ~4,704,000 | 43.6h | 23.8h | 9.33h |
| **Route B Speech (Dev + Test)** | 4 | 2,212 | 6,636 | ~1,659,000 | 15.4h (+3.3h audio) | 8.4h | 3.29h |
| **Route C Translation (Dev + Test)** | 4 | 2,212 | 6,636 | ~1,659,000 | 15.4h (+3.7h MT) | 8.4h | 3.29h |
| **Stage 6 Romanizer Ablations** | 1 | 161 | 483 | ~120,750 | 1.1h | 0.6h | 0.24h |
| **Llama-Guard 3 Judging (All)** | — | — | 40,656 | ~406,560 | 2.5h | 1.4h | 0.56h |
| **Rubric LLM Judging (All)** | — | — | 40,656 | ~6,098,400 | 56.5h | 30.8h | 12.10h |
| **Full Total Framework** | — | — | **40,656** | **~16,668,960** | **~156.6h** | **~85.2h** | **~33.5h** |

### Reduced Design (Optimized for Free Colab / Kaggle Quotas)
- Primary Model (`Qwen2.5-7B-Instruct`) receives full pipeline: all routes, mitigations, code-mixed bands, and ablations.
- Comparison Models (`Llama-3.1-8B-Instruct`, `Gemma-2-9B-it`) evaluated on Route A Dev & Test Core 4 variants only.
- Judging: Llama-Guard 3 evaluated on all 17,976 responses; Rubric Judge evaluated on all responses.

| Reduced Phase | Models | Prompts | Total Tokens | Colab T4 (Hrs) | Kaggle 2×T4 (Hrs) | Colab A100 (Hrs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Route A Dev Core (4 Vars)** | 3 | 1,932 | ~483,000 | 4.5h | 2.4h | 0.95h |
| **Route A Dev Code-Mixed (3 Bands)** | 1 (Qwen) | 483 | ~120,750 | 1.1h | 0.6h | 0.24h |
| **Route A Test Core (4 Vars)** | 3 | 4,704 | ~1,176,000 | 10.9h | 5.9h | 2.33h |
| **Route A Test 4 Mitigations** | 1 (Qwen) | 6,272 | ~1,568,000 | 14.5h | 7.9h | 3.11h |
| **Route B Speech (Dev + Test)** | 1 (Qwen) | 2,212 | ~553,000 | 5.1h (+1.1h audio) | 2.8h | 1.10h |
| **Route C Translation (Dev + Test)** | 1 (Qwen) | 2,212 | ~553,000 | 5.1h (+1.2h MT) | 2.8h | 1.10h |
| **Stage 6 Romanizer Ablation** | 1 (Qwen) | 161 | ~40,250 | 0.4h | 0.2h | 0.08h |
| **Llama-Guard 3 Judging** | — | 17,976 | ~179,760 | 1.1h | 0.6h | 0.25h |
| **Rubric LLM Judging** | — | 17,976 | ~2,696,400 | 25.0h | 13.6h | 5.35h |
| **Reduced Total** | — | **17,976** | **~7,370,160** | **~73.9h** | **~40.0h** | **~15.7h** |

---

## 5. Route B Feasibility Matrix (Audio: ASR & TTS)

*Runtime status verified on host environment; all offline packages marked **UNVERIFIED**.*

| Language / Modality | Component | Engine / Model Candidate | Feasibility Status | Coverage & Implementation Details |
| :--- | :--- | :--- | :---: | :--- |
| **English (`en`)** | ASR | `openai/whisper-large-v3` | **FEASIBLE** | Native support; WER < 5%; offline package UNVERIFIED. |
| **Hindi (`hi`)** | ASR | `openai/whisper-large-v3` | **FEASIBLE** | Officially supported (`language="hi"`); standard Devanagari output. |
| **Hinglish** | ASR | `whisper-large-v3` / `IndicConformer` | **FEASIBLE WITH POST-PROCESS** | Whisper forces Devanagari or English words; `AI4Bharat IndicConformer` supports code-mixed text. |
| **Bhojpuri (`bho`)** | ASR | `whisper-large-v3` | **INFEASIBLE (NATIVE)** | Whisper lacks `<|bho|>` token; forces Hindi, destroying Bhojpuri morphology. |
| **Bhojpuri (`bho`)** | ASR | `facebook/mms-1b-all` | **FEASIBLE** | Meta MMS explicitly supports `bho-script_devanagari`; UNVERIFIED at runtime. |
| **English (`en`)** | TTS | Edge-TTS / Google TTS / MMS | **FEASIBLE** | Ubiquitous support across all engines. |
| **Hindi (`hi`)** | TTS | `facebook/mms-tts-hin` / Edge-TTS | **FEASIBLE** | High-quality voices available; Edge-TTS (`hi-IN-SwaraNeural`) native. |
| **Bhojpuri (`bho`)** | TTS | Google / Azure / IndicTTS | **INFEASIBLE** | Commercial engines and IndicTTS v1 have NO Bhojpuri voice. |
| **Bhojpuri (`bho`)** | TTS | `facebook/mms-tts-bho` | **FEASIBLE (OPEN SOURCE)** | Meta MMS-TTS provides a dedicated Bhojpuri checkpoint; UNVERIFIED at runtime. |
| **Hinglish** | Speech Synth | Azure Neural (`en-IN-NeerjaMultilingual`) | **RECOMMENDED** | Commercial bilingual neural voice natively synthesizes Romanized code-mixed text. |
| **Hinglish** | Speech Synth | Script Splitting + Audio Splicing | **FALLBACK (OFFLINE)** | Split English/Hindi tokens, synthesize via `mms-tts-eng`/`mms-tts-hin`, crossfade concatenate. |

---

## 6. LaBSE Audit & Bhojpuri Replacement Plan

### LaBSE Bhojpuri Coverage Audit
- **Verification Result**: Bhojpuri (`bho`) is **NOT** in LaBSE's 105/109 supported language list (Feng et al., 2020).
- **Impact**: Any cosine similarity computed on Bhojpuri using LaBSE is a zero-shot artifact of Hindi subword overlap, artificially depressing similarity scores and distorting QC thresholds.

### Replacement Plan for Every Pipeline Usage of LaBSE

| Pipeline Stage | Previous LaBSE Usage | Defect with Bhojpuri | Proposed Replacement |
| :--- | :--- | :--- | :--- |
| **Stage 1 QC & Filtering** | Pairwise LaBSE($\text{src}_{\text{en}}, \text{nllb}_{\text{bho}}$) thresholding ($<0.70$) | False rejects due to zero-shot embedding space penalty | **English Round-Trip Back-Translation Evaluation**: Measure chrF++ and BERTScore between English source and NLLB English back-translation, OR use **Meta Sonar** (`text_sonar_basic_encoder`, natively supports `bho_Deva`). |
| **Drift Triage** | Tiering P1/P2/P3 via English vs. Bhojpuri embedding distance | Misclassifies valid Bhojpuri colloquial expressions as semantic drift | **Monolingual English Semantic Equivalence**: Compare English source with NLLB back-translation using `all-mpnet-base-v2`, combined with zero-shot LLM premise preservation verification. |
| **Cross-Lingual Consistency** | Planned $S(\text{resp}_{\text{en}}, \text{resp}_{\text{bho}})$ via LaBSE embedding similarity | Deflates cross-lingual consistency metric for Bhojpuri responses | **Back-Translate Responses to English prior to embedding similarity**, OR compute cross-lingual similarity using **Meta Sonar** 200-language unified space. |

---

## 7. FLORES-200 Relative-Only Justification: Benchmark Contamination

### Revised Justification (Contamination Replacing LaBSE)
- **Open Benchmark Exposure**: FLORES-200 was published publicly in 2022 under CC-BY-SA 4.0. Its 200-language parallel sentences have been crawled extensively and incorporated into internet pretraining datasets (Common Crawl, RefinedWeb, FineWeb, RedPajama).
- **Memorization Confound**: Large language models (`Qwen2.5`, `Llama-3.1`, `Gemma-2`) have observed substantial fractions of FLORES-200 sentences during pretraining. For low-resource languages like Bhojpuri, FLORES-200 constitutes a disproportionately large share of available high-quality web parallel text, heightening the risk of verbatim memorization.
- **Specification**: All FLORES-200 machine translation and comprehension probe scores must be interpreted strictly as **relative-only metrics** for ranking models against each other under identical evaluation conditions, rather than absolute measures of zero-shot linguistic reasoning.

---

## 8. File Inventory & Bit-Level SHA-256 Checksums

| File Path | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :---: | :--- | :--- |
| `outputs/stage1/dataset.jsonl` | 4,086,078 | `2cd04e5fb97ebe26570bf7de2361be22fa5e91ae2ddbc954d08fd71caf4b5b77` | Frozen Input Dataset |
| `outputs/stage1/seed_split_manifest_v3.json` | 132,539 | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` | Frozen Manifest v3 |
| `stage2_route_a.py` | 37,806 | `537121d5d240f0a62442d7a47ef1c2d2a9907ac4fa070fb617a2e6822919bf54` | Patched Script (Stage 2d) |
| `outputs/reports/response_label_sample.csv` | 15,373 | `8e44e41f344ed35f57c03264bbe4ec9d714eae3514a98f8919fd98d4d35fcf19` | Human Review Sampling Sheet |
| `outputs/stage2/route_a_plumbing.jsonl` | 21,090 | `136a464dcf5a5bc6964b787fe9a7d8b5c02d811cf6d4b8f59453c90250b55272` | Re-verified Plumbing Outputs |
| `outputs/stage2/route_a_smoke_test.jsonl` | 61,966 | `cfe11f125640b291a7b773262e9f255108f3be5fde974ac8f3b2ecd9f1322d89` | Backfilled Smoke Test Responses |
| `outputs/stage2/smoke_test_report.json` | 812 | `f46cae186cd1b7c9a240c1f6029a6aaae133189d7348baacda0943944efecc7f` | Code-Regenerated Smoke Summary |
| `.gitignore` | 411 | `e261106e10db774d942aedb86d1522a33360d9559972b059e22b9461b4963333` | Updated Git Tracking Config |
| `outputs/reports/stage2d_audit_report.md` | (self) | (self) | Comprehensive Stage 2d Audit |
