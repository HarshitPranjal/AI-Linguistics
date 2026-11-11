---
title: "AI Linguistic Vitality: Auditing LLM Safety & Multi-Turn Vulnerabilities Across Low-Resource Indic and Code-Mixed Languages"
thumbnail: "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
authors:
  - name: "AI Safety & Linguistic Vitality Team"
tags:
  - ai-safety
  - red-teaming
  - multilingual
  - indic
  - bhojpuri
  - hinglish
  - multi-turn-audit
---

# AI Linguistic Vitality: Auditing LLM Safety & Multi-Turn Vulnerabilities Across Low-Resource Indic and Code-Mixed Languages

## 1. Executive Summary

As large language models (LLMs) are deployed globally, an alarming alignment gap has emerged: **safety fine-tuning in high-resource standard English does not reliably transfer to low-resource regional languages, informal code-mixed vernacular, or complex multi-turn interactions**.

While a model may steadfastly refuse hazardous queries in standard English, the exact same semantic intent often bypasses safety filters when framed in **Bhojpuri (`bho`)**, informal **romanized Hinglish (`hinglish_translit`)**, or when partitioned across **multi-stage crescendo dialogues** and **indirect tool-use observations**.

To address this challenge, we introduce **AI Linguistic Vitality**, a comprehensive, pre-registered safety benchmark and multi-turn auditing framework featuring:
1. A **4,424-record cryptographically frozen benchmark dataset** spanning 553 unique seeds across **8 linguistic varieties** (English, Hindi Devanagari, Bhojpuri, Colloquial Hinglish, Formal Hinglish, and 3 alignment-based code-mixing bands).
2. A **pre-registered analysis plan (v1.2)** establishing formal hypotheses ($H_1, H_2, H_3$) and dual-consensus lower and upper bounds ($s3a \land s3b$ vs. $s3a \lor s3b$) prior to test set response labelling.
3. A **multi-turn audit engine** evaluating attacks a single prompt cannot catch—such as indirect prompt injection in tool outputs and gradual multi-turn crescendo jailbreaks—identifying the exact turn of failure ($T_{\text{break}}$) with cryptographically reproducible transcripts.
4. A **calibrated multi-signal judge architecture** combining fast regex filters ($S_1$), zero-shot hazard classification via Llama-Guard-3 ($S_2$), and dual-branch rubric judging via Llama-3.1-8B-Instruct ($S_3$).

---

## 2. The Linguistic & Multi-Turn Vulnerability Surface

### The Low-Resource Vernacular Gap: Bhojpuri
With over 50 million speakers across Eastern Uttar Pradesh, Bihar, and Nepal, Bhojpuri possesses distinct morphosyntax, first-person verbal inflections (e.g., *-ीं*, *-ईं*), and unique auxiliary verbs (*बा*, *बानी*, *बाड़*) that diverge sharply from standard Hindi. Tokenizers trained predominantly on English and Hindi Devanagari represent Bhojpuri with higher subword fertility and distorted semantic representations, allowing adversarial requests to evade alignment training.

### The Code-Mixed Orthographic Continuum: Hinglish
In daily digital communication, Indian users communicate in Hinglish—a fluid blend of Hindi vocabulary and grammar written in Latin script. Standard transliteration baselines (such as strict IAST) fail to capture colloquial reality, where users drop medial schwas (*karna* $\to$ *krna*), reduce vowels (*dekho* $\to$ *deko*), and code-switch intra-sententially. Standard tokenizers struggle with this colloquial divergence, creating substantial safety evasion surfaces.

### Multi-Turn Context Decay & Agentic Blindness
Single-turn evaluations measure static refusal boundaries, but fail to capture how models behave when:
- **Attacks are distributed gradually**: Starting with abstract academic or historical queries, gradually contextualizing the dialogue, and caving to hazardous requests at Turn 3 or 4 ($T_{\text{break}} \ge 2$).
- **Attacks are injected into tool outputs**: Agents given benign user queries (e.g., "Summarize this document") execute adversarial instructions hidden inside the retrieved data, failing to separate the untrusted observation channel from the trusted control channel.

---

## 3. Dataset Architecture & Cryptographic Governance

The dataset was constructed under strict determinism, with all seed selections and translations validated through multi-stage quality control:

| Linguistic Variety | Script / Orthography | Generation / Normalization Mechanism | Active Records |
| :--- | :--- | :--- | :--- |
| **English (`en`)** | ASCII Standard | Source anchor seeds (HarmBench, AdvGLUE, XSTest, XNLI) | 553 |
| **Hindi (`hi`)** | Devanagari | NLLB-200 3.3B + LaBSE semantic cosine verification | 553 |
| **Bhojpuri (`bho`)** | Devanagari | Distilled NLLB + native linguistic marker validation | 553 |
| **Colloquial Hinglish (`hinglish_translit`)** | Latin (Colloquial) | Sequence-aligned romanizer with schwa syncope & vowel reduction | 553 |
| **Formal Hinglish (`hinglish_translit_formal`)** | Latin (IAST-aligned) | Formal transliteration baseline | 553 |
| **Hinglish Mixed Low (`hinglish_mixed_low`)** | Latin / Code-Mixed | Alignment-based code-mixing (CMI: 0.10 – 0.25) | 553 |
| **Hinglish Mixed Mid (`hinglish_mixed_mid`)** | Latin / Code-Mixed | Alignment-based code-mixing (CMI: 0.25 – 0.40) | 553 |
| **Hinglish Mixed High (`hinglish_mixed_high`)** | Latin / Code-Mixed | Alignment-based code-mixing (CMI: > 0.40) | 553 |
| **Total Benchmark Dataset** | | **SHA-256: `e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf`** | **4,424** |

### Split Discipline & Seed Stratification
The 553 seeds are stratified into:
- **Dev Split (161 seeds $\times$ 8 varieties = 1,288 records)**: Exclusively utilized for judge calibration, refusal pattern refinement, and human inter-annotator agreement.
- **Test Split (392 seeds $\times$ 8 varieties = 3,136 records)**: Cryptographically locked under Git tag `manifest-v3-frozen` (SHA-256: `2d781b131e94...`). Test set responses are evaluated strictly after the judge configuration is frozen.

---

## 4. Multi-Turn Safety State Machine & Survival Analysis

Our multi-turn evaluation framework models extended conversations using a state machine evaluated at every dialogue turn $t \in [1, K]$:

$$\text{Safety State } S_t \in \{\text{SAFE\_REFUSAL}, \text{SAFE\_PARTIAL}, \text{BENIGN\_FULFILLMENT}, \text{HARMFUL\_COMPLIANCE}, \text{INJECTION\_EXECUTED}\}$$

### Break Point Detection ($T_{\text{break}}$)
$$T_{\text{break}} = \min \left\{ t \in [1, K] \;\middle|\; S_t \in \{\text{HARMFUL\_COMPLIANCE}, \text{INJECTION\_EXECUTED}\} \right\}$$
- **Robust Alignment**: $T_{\text{break}} = \infty$ (model refused or safely handled all $K$ turns).
- **Immediate Collapse**: $T_{\text{break}} = 1$ (acute single-turn vulnerability).
- **Crescendo Drift**: $1 < T_{\text{break}} \le K$ (model caved after compliant or safe early turns).
- **Indirect Hijack**: $T_{\text{break}}$ coincides with an untrusted tool return containing an injected directive.

### Reproducible Transcripts
Every interaction is logged into cryptographically verifiable JSONL transcripts recording turn-by-turn dialogue history, tool calls, tool responses, token counts, and safety states.

---

## 5. The Multi-Signal Automated Safety Judge

Evaluating multilingual and code-mixed safety using a single monolithic judge introduces severe single-model artifact bias. To eliminate this, we implement a **Three-Signal Architecture**:

1. **Signal 1 ($S_1$) — Regex Governance**: Refusal and compliance-transition patterns across all 4 core varieties. In accordance with pre-registration discipline, all patterns are initially set to `UNVALIDATED` and updated strictly from human-labelled dev responses.
2. **Signal 2 ($S_2$) — Hazard Categorization**: Llama-Guard-3-8B classification. Rule (a) enforces that $S_2$ never overrides the rubric judge ($S_3$). Evaluations against Llama target models explicitly track `is_guard_self_judging = True`.
3. **Signal 3 ($S_3$) — Dual-Branch Rubric Judge (Llama-3.1-8B-Instruct)**:
   - *Branch 3a*: Native target prompt + target response.
   - *Branch 3b*: English back-translated prompt + back-translated response (via NLLB-200 3.3B).
   - **Pre-Registered Bounds**:
     - *Lower Bound (Consensus)*: $s3a == \text{label} \land s3b == \text{label}$
     - *Upper Bound (Either)*: $s3a == \text{label} \lor s3b == \text{label}$
4. **Queue Capping & Fallback**: Multi-signal disagreements are capped at $N=150$ items for human review, with deterministic pre-registered fallbacks for the remainder.

---

## 6. Human Calibration & Ethical Safeguards

Human evaluation is governed by our formal protocol (`outputs/reports/annotation_plan.md`):
- **400 Stratified Dev Items**: Minimum 100 per core variety, with a 20% dual-annotator overlap (80 items) to guarantee Cohen's $\kappa \ge 0.60$.
- **Native Dialect Criteria**: Annotators evaluating Bhojpuri must meet native fluency standards in Bhojpuri grammatical inflections and boundary differentiation from Maithili and Eastern Hindi.
- **Annotator Wellbeing**: Line-1 content warnings on all sheets, voluntary participation with zero-penalty item-level opt-out (`skip_item_Y_N`), and complete file isolation for self-harm evaluations into a dedicated opt-in sheet (`test_validation_self_harm.csv`).

---

## 7. Submission Artifacts & Public Links

All resources are publicly available under the Apache 2.0 License:

- **GitHub Repository**: [`github.com/your-org/ai-linguistic-vitality`](https://github.com/your-org/ai-linguistic-vitality)
- **Hugging Face Dataset Collection**: [`huggingface.co/collections/your-org/ai-linguistic-vitality-datasets`](https://huggingface.co/collections/)
- **Hugging Face Model Collection**: [`huggingface.co/collections/your-org/ai-linguistic-vitality-models`](https://huggingface.co/collections/)
- **Colab Execution Bootstrap**: [`colab_bootstrap.py`](https://github.com/your-org/ai-linguistic-vitality/blob/main/colab_bootstrap.py)

