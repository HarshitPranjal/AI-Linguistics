---
language:
- en
- hi
- bho
license: apache-2.0
multilinguality:
- multilingual
size_categories:
- 1K<n<10K
task_categories:
- text-classification
- feature-extraction
tags:
- ai-safety
- red-teaming
- multilingual-safety
- indic-nlp
- bhojpuri
- hinglish
- code-mixing
- multi-turn-auditing
- jailbreak-benchmarking
pretty_name: AI Linguistic Vitality Indic Safety Benchmark
dataset_info:
  features:
  - name: seed_id
    dtype: string
  - name: lang
    dtype: string
  - name: prompt
    dtype: string
  - name: type
    dtype: string
  - name: category
    dtype: string
  - name: source_dataset
    dtype: string
  - name: split
    dtype: string
  - name: qc_scores
    struct:
    - name: labse_cosine
      dtype: float64
    - name: length_ratio
      dtype: float64
    - name: is_qc_passed
      dtype: bool
    - name: labse_direct
      dtype: float64
    - name: labse_bt
      dtype: float64
  splits:
  - name: dev
    num_bytes: 524288
    num_examples: 1288
  - name: test
    num_bytes: 1258291
    num_examples: 3136
  download_size: 1782579
  dataset_size: 1782579
---

# Dataset Card for AI Linguistic Vitality (Indic Safety & Multi-Turn Benchmark)

## Dataset Summary
The **AI Linguistic Vitality** benchmark is a parallel multilingual and code-mixed safety evaluation dataset containing **4,424 frozen instances** across 553 unique seeds and **8 linguistic varieties**. It is designed to evaluate how large language models maintain safety boundaries when prompted in low-resource Indic languages (Bhojpuri) and informal code-mixing (Hinglish), as well as across complex multi-turn attacks (gradual crescendo jailbreaks and indirect tool injections).

## Supported Tasks & Languages

### Languages:
- **English (`en`)**: Source anchor high-resource reference.
- **Hindi (`hi`)**: Devanagari script standard Indo-Aryan reference.
- **Bhojpuri (`bho`)**: Low-resource regional Indo-Aryan language (Devanagari script), calibrated with native linguistic validation.
- **Colloquial Hinglish (`hinglish_translit`)**: Primary Latin-script transliterated Hindi incorporating medial schwa syncope, vowel reduction (ee $\to$ i, oo $\to$ u), and loanword normalization.
- **Formal Hinglish (`hinglish_translit_formal`)**: Standard IAST-aligned Latin transliteration baseline.
- **Alignment-based Code-Mixed Hinglish (`hinglish_mixed_low`, `mid`, `high`)**: Progressively mixed sentences calibrated across three Code-Mixing Index (CMI) bands.

### Task Taxonomies:
1. `unsafe_request`: Hazardous prompts spanning hate, cyber-attacks, CBRN, fraud, and illegal acts.
2. `borderline_benign`: Benign prompts containing sensitive keywords designed to test for over-refusal (XSTest).
3. `utility_nli`: Semantic natural language inference pairs designed to test semantic comprehension independent of safety (XNLI).

## Dataset Structure

### Data Instances
Each record in `dataset.jsonl` follows a strict cryptographic schema:
```json
{
  "seed_id": "harmbench_text-12",
  "lang": "bho",
  "prompt": "<prompt text>",
  "type": "unsafe_request",
  "category": "cyber_exploit",
  "source_dataset": "HarmBench",
  "split": "test",
  "qc_scores": {
    "labse_cosine": 0.8842,
    "length_ratio": 1.042,
    "is_qc_passed": true,
    "labse_direct": 0.8842,
    "labse_bt": 0.8915
  }
}
```

### Splits
- **dev split (161 seeds $\times$ 8 varieties = 1,288 records)**: Exclusively allocated for prompt calibration, S1 regex expansion, and judge tuning.
- **test split (392 seeds $\times$ 8 varieties = 3,136 records)**: Held-out split for final pre-registered hypothesis testing ($H_1, H_2, H_3$).

## Cryptographic Integrity Checksums

| Asset | SHA-256 Checksum | Protocol Status |
| :--- | :--- | :--- |
| `dataset.jsonl` | `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf` | **FROZEN** |
| `seed_split_manifest_v3.json` | `2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c` | **FROZEN** |
| `seed_review_v1.csv` | `03fe6b8c5b2a666fd7534883fb4aae7f55a359872189ef3a27976640138c2e8e` | **PROVISIONAL** |

## Annotation & Human Subjects Ethics
- All human evaluation data adheres to pre-registered ethical protocols with line-1 content notices and voluntary opt-out.
- Extreme sensitivity items (self-harm) are isolated into a separate opt-in sheet (`test_validation_self_harm.csv`).

## Licensing
This dataset is released under the **Apache 2.0 License**.

