# Hugging Face Model Collection Specification

**Collection Title**: `ai-linguistic-vitality-models`  
**Visibility**: Public (Deadline: Nov 15, 11:59 PM AoE)  
**Description**: Complete public model collection containing target models, safety judges, translation engines, and model artifacts evaluated in the AI Linguistic Vitality benchmark.

---

## 1. Collection Layout & Component Registry

The public Hugging Face collection must include the following models and artifacts:

### A. Primary & Secondary Evaluated Target Models
These open-weights models are evaluated across the 8 linguistic varieties (4,424 records) in the staged benchmark design:

| Model Repository | Parameter Size | Architecture | Evaluation Scope |
| :--- | :--- | :--- | :--- |
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.61B | Qwen2 | **Full Dataset** (all 8 varieties, 4,424 records) |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 8.03B | Llama 3.1 | **Core 4 Varieties** (`en`, `hi`, `bho`, `hinglish_translit`, 2,212 records) |
| [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | 9.24B | Gemma 2 | **Core 4 Varieties** (`en`, `hi`, `bho`, `hinglish_translit`, 2,212 records) |

### B. Automated Safety & Moderation Judges
The multi-signal arbitration pipeline utilizes these moderation models:

| Model Repository | Role in Framework | Function |
| :--- | :--- | :--- |
| [`meta-llama/Llama-Guard-3-8B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | **Signal 2 ($S_2$) Judge** | Zero-shot hazard categorization across MLCommons safety taxonomies |
| [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | **Signal 3 ($S_3$) Rubric Judge** | Dual-branch evaluation (native target prompt vs. English back-translated prompt) |

### C. Semantic Embedding & Translation Encoders
Auxiliary models for translation mediation and quality control:

| Model Repository | Function in Framework |
| :--- | :--- |
| [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) | Roundtrip back-translation engine for Branch 3b and Bhojpuri synthesis |
| [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) | Language-agnostic semantic embedding cosine QC verifier (`qc_scores.labse_cosine`) |

---

## 2. Steps to Create the Public Collection on Hugging Face

### Option 1: Via the Hugging Face Web UI
1. Log in to [Hugging Face](https://huggingface.co/).
2. Navigate to your profile $\to$ **Collections** $\to$ **New collection**.
3. Set Collection Name: `AI Linguistic Vitality: Models & Evaluators`.
4. Set Visibility to **Public**.
5. Click **Add items** and add:
   - `Qwen/Qwen2.5-7B-Instruct`
   - `meta-llama/Llama-3.1-8B-Instruct`
   - `google/gemma-2-9b-it`
   - `meta-llama/Llama-Guard-3-8B`
   - `facebook/nllb-200-3.3B`
   - `sentence-transformers/LaBSE`
6. Add description: *"Open-weights target models, safety judges, and translation engines evaluated in the AI Linguistic Vitality benchmark for cross-lingual and multi-turn safety."*

### Option 2: Programmatically via `huggingface_hub` Python API
```python
from huggingface_hub import HfApi

api = HfApi(token="YOUR_HF_TOKEN")

collection = api.create_collection(
    title="AI Linguistic Vitality: Models & Evaluators",
    description="Public model collection for the AI Linguistic Vitality safety benchmark.",
    namespace="your-hf-username"
)

models_to_add = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-Guard-3-8B",
    "facebook/nllb-200-3.3B",
    "sentence-transformers/LaBSE"
]

for model_id in models_to_add:
    api.add_collection_item(
        collection_slug=collection.slug,
        item_id=model_id,
        item_type="model"
    )
print(f"Collection created: {collection.url}")
```

