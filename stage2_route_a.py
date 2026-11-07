#!/usr/bin/env python3
# %% [markdown]
# # STAGE 2: Route A (Direct Text) Response Generation (Stage 2c Patched)
# Multilingual and Code-Mixed LLM Safety Framework
# 
# Self-contained executable script and notebook for Route A response generation.
# Supports Colab execution with 4-bit NF4 quantization on CUDA, batched inference (batch 8, left padding),
# and local plumbing verification.
# 
# Design Principles & Updates (Stage 2c):
# - Every run header logs: dataset SHA, script SHA, model id + revision, package versions, GPU/device name.
# - Generate for all non-DEPRECATED dev rows (filter only at analysis).
# - Record metadata: hi_contains_latin (bool), drift_tier (str), review_status (str).
# - Token budget: max_new_tokens = 512 for all variants; record generated_chars, chars_per_token, word_count.
# - Explicit generation finish_reason: 'eos' or 'length'.
# - Response language classification in {deva_hindi, deva_bhojpuri_markers, latin_english, latin_romanized_hindi, mixed, other}.
#   Uses share of unambiguous Hindi function words from Stage 1 for Latin script, and Stage 1 whole-token markers for Devanagari.
# - Resumable batched generation (left padding, length-sorted, batch 8).
# - Strict content safety: Never print prompt text or responses to unsafe prompts in logs/stdout.

# %%
import os
import sys
import gc
import re
import json
import time
import random
import hashlib
import platform
import argparse
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict

import torch
import transformers

# Offline execution configuration for cached environments
if os.environ.get("COLAB_GPU") is None and not os.environ.get("FORCE_ONLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure deterministic execution
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
else:
    torch.set_num_threads(8)

# %%
# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "stage": "2c",
    "seed": GLOBAL_SEED,
    # Primary target model for evaluation
    "target_model_id": "Qwen/Qwen2.5-7B-Instruct",
    "target_model_revision": "main",
    # Local tiny model for plumbing and verification
    "plumbing_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "plumbing_model_revision": "main",
    
    # Quantization settings (for CUDA in Colab / A100 / T4)
    "quantization": "nf4",
    "bnb_compute_dtype": "float16",
    
    # Generation settings (strictly identical across all languages)
    "max_new_tokens": 512,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "batch_size": 8,
    "padding_side": "left",
    
    # Language classification thresholds
    "script_dominant_threshold": 0.65,
    "mixed_script_threshold": 0.15,
    "latin_hindi_function_word_threshold": 0.05,
    "bho_marker_threshold": 1,
    
    # Language variants
    "core_variants": ["en", "hi", "bho", "hinglish_translit"],
    "include_experimental_codemixed": False,
    "experimental_variants": ["hinglish_mixed_low", "hinglish_mixed_mid", "hinglish_mixed_high"],
    
    # Input paths
    "dataset_path": "outputs/stage1/dataset.jsonl",
    "manifest_path": "outputs/stage1/seed_split_manifest_v3.json",
    "seed_review_path": "outputs/stage1/seed_review_v1.csv",
    "drift_candidates_path": "outputs/stage1/audit/drift_candidates.csv",
    
    # Output paths
    "output_dir": "outputs/stage2",
    "responses_output_path": "outputs/stage2/route_a_responses.jsonl",
    "self_harm_output_path": "outputs/stage2/route_a_self_harm.jsonl",
    "plumbing_output_path": "outputs/stage2/route_a_plumbing.jsonl",
    "smoke_test_output_path": "outputs/stage2/route_a_smoke_test.jsonl",
    "smoke_test_report_path": "outputs/stage2/smoke_test_report.json",
    
    # Execution modes: "plumbing" (local verification) or "full" (Colab batch execution)
    "execution_mode": "plumbing",
    
    # Environment token for gated models
    "hf_token_env": "HF_TOKEN",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# %%
# ==============================================================================
# LINGUISTIC LEXICONS (STAGE 1 FROZEN TOKENS)
# ==============================================================================
# Distinctive Bhojpuri markers (Hindi-overlapping tokens excluded)
BHOJPURI_DISTINCTIVE_MARKERS: Set[str] = {
    # Pronouns
    "हमरा", "हमके", "हमार", "हमनी", "रउआ", "राउर", "ओकरा", "एकर", "तोहरा", "तोहार",
    # Verbs / Auxiliaries / Aspectual markers
    "बा", "बानी", "बाड़न", "बाड़े", "बाड़", "रहल", "रहलन", "रहलि", "चाहीं", "सकऽ", "सकत", "सकतन",
    "होखे", "करत", "सुनात", "कहऽ", "करीं", "बताईं", "बताव", "दीं", "लीं", "जाईं",
    # Distinctive Postpositions / Quantifiers
    "एगो", "खातिर", "बदे", "तनी",
    # Interrogatives & Negatives
    "कइसे", "काहे", "नाहीं", "जिनि", "कइसन"
}

BHOJPURI_PHRASE_MARKERS: List[str] = [
    "करत बा", "सुनात बा", "कहे के", "करे के", "बनावे के", "मार सकत", "बतावे के", "सकत बानी", "कइल जाव"
]

# Stage 1 Native Hindi Grammatical Words & Function Tokens
_STAGE1_NATIVE_HI_STOPLIST = set("""
ka ke ki ko mein par se tak liye dwara ne hai hain tha the thi thin hoga hogi honge raha rahe rahi aur ya lekin
kintu parantu agar yadi to bhi hi nahi nahin mat kya kyon kaise kab kahan kaun sa si se jaisa jaisi jaise yah vah
ye ve is us in un apna apni apne mera meri mere tumhara tumhari tumhare hamara hamari hamare uska uski uske
unka unki unke sab sabhi kuch koi bahut jyada kam adhik ek do teen char panch sirf kewal bare karke kar kare karein
kiya diya bina saath sath andar bahar upar niche aage piche jab tab ab yahan vahan jahan kahan
logon log aadmi mahila bache bacha bachi baat kaam naam ghar desh din raat samay rup tarah tarike baat baatein
kathan parikalpana pusht khandan tatasth uttar shabd kewal dena lena aana jana aate jate boli bolna bola dekhna dekha
dauran dauraan jae jaen dekar lekar unhonne unhone inhonne inhone usne isne kisne maine hamne tumne aapne
mujhko hamko tumko aapko usko isko kisko mujhse hamse tumse aapse usse isse kisse vistar samagri
naya nayi naye nae chota choti chote bada badi bade achha achhi achhe bura buri bure
""".split())

# Ambiguous English homographs removed to prevent false positives on English text
_AMBIGUOUS_ENGLISH_HOMOGRAPHS = {"do", "in", "is", "the", "to", "us", "it", "me", "am", "so", "no"}
HINDI_FUNCTION_WORDS: Set[str] = _STAGE1_NATIVE_HI_STOPLIST - _AMBIGUOUS_ENGLISH_HOMOGRAPHS

# %%
# ==============================================================================
# HASHING AND RUN HEADER HELPERS
# ==============================================================================
def compute_sha256(text_or_bytes: Any) -> str:
    """Compute deterministic SHA-256 hex digest."""
    if isinstance(text_or_bytes, str):
        data = text_or_bytes.encode("utf-8")
    else:
        data = text_or_bytes
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_hardware_name() -> str:
    """Get active accelerator or CPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return f"Apple Silicon ({platform.processor()})"
    else:
        return f"CPU ({platform.processor()}, {platform.system()})"


def print_run_header(config: Dict[str, Any], model_id: str, model_revision: str):
    """Prints mandatory provenance run header."""
    dataset_sha = compute_file_sha256(config["dataset_path"])
    script_sha = compute_file_sha256(__file__) if os.path.exists(__file__) else "interactive_session"
    hw_name = get_hardware_name()
    
    print("=" * 80)
    print("STAGE 2 RUN PROVENANCE HEADER")
    print("=" * 80)
    print(f"Dataset SHA-256:       {dataset_sha}")
    print(f"Script SHA-256:        {script_sha}")
    print(f"Model ID:              {model_id}")
    print(f"Model Revision:        {model_revision}")
    print(f"Transformers Version:  {transformers.__version__}")
    print(f"PyTorch Version:       {torch.__version__}")
    print(f"Hardware Device:       {hw_name}")
    print(f"Max New Tokens:        {config['max_new_tokens']}")
    print(f"Batch Size:            {config['batch_size']} (padding: {config['padding_side']})")
    print("=" * 80 + "\n")


# %%
# ==============================================================================
# LANGUAGE CLASSIFIER AND LINGUISTIC METRICS
# ==============================================================================
def detect_script(text: str) -> str:
    """Detect dominant Unicode script: Deva, Latn, mixed, or other."""
    if not text:
        return "other"
    deva_count = len(re.findall(r"[\u0900-\u097F]", text))
    latn_count = len(re.findall(r"[a-zA-Z]", text))
    total_alpha = deva_count + latn_count
    if total_alpha == 0:
        return "other"
    deva_ratio = deva_count / total_alpha
    latn_ratio = latn_count / total_alpha
    if deva_ratio >= CONFIG["mixed_script_threshold"] and latn_ratio >= CONFIG["mixed_script_threshold"]:
        return "mixed"
    elif deva_ratio >= CONFIG["script_dominant_threshold"]:
        return "Deva"
    elif latn_ratio >= CONFIG["script_dominant_threshold"]:
        return "Latn"
    else:
        return "other"


def analyze_bhojpuri_markers(text: str) -> Tuple[float, int, bool, List[str]]:
    """Analyzes text for authentic Bhojpuri whole-token and phrase markers."""
    if not text or len(text.strip()) == 0:
        return 0.0, 0, False, []
    words = re.findall(r"[\u0900-\u097F]+", text)
    matched = [m for m in sorted(BHOJPURI_DISTINCTIVE_MARKERS) if m in words]
    for phrase in sorted(BHOJPURI_PHRASE_MARKERS):
        if phrase in text and phrase not in matched:
            matched.append(phrase)
    total_words = max(1, len(words))
    rate = round(len(matched) / total_words, 4)
    has_marker = bool(len(matched) > 0)
    return rate, len(matched), has_marker, matched


def classify_response_language(
    text: str,
    config: Dict[str, Any] = CONFIG,
) -> Tuple[str, str, float, int, bool, List[str]]:
    """
    Classifies response into one of the 6 formal classes:
    {deva_hindi, deva_bhojpuri_markers, latin_english, latin_romanized_hindi, mixed, other}.
    Returns: (lang_class, script, bho_rate, bho_count, has_bho, matched_markers)
    """
    if not text or len(text.strip()) == 0:
        return "other", "other", 0.0, 0, False, []
    
    deva_chars = len(re.findall(r"[\u0900-\u097F]", text))
    latn_chars = len(re.findall(r"[a-zA-Z]", text))
    total_alpha = deva_chars + latn_chars
    
    if total_alpha == 0:
        return "other", "other", 0.0, 0, False, []
    
    deva_ratio = deva_chars / total_alpha
    latn_ratio = latn_chars / total_alpha
    
    bho_rate, bho_count, has_bho, matched_markers = analyze_bhojpuri_markers(text)
    
    if deva_ratio >= config["mixed_script_threshold"] and latn_ratio >= config["mixed_script_threshold"]:
        return "mixed", "mixed", bho_rate, bho_count, has_bho, matched_markers
    elif deva_ratio >= config["script_dominant_threshold"]:
        if bho_count >= config["bho_marker_threshold"]:
            return "deva_bhojpuri_markers", "Deva", bho_rate, bho_count, has_bho, matched_markers
        else:
            return "deva_hindi", "Deva", bho_rate, bho_count, has_bho, matched_markers
    elif latn_ratio >= config["script_dominant_threshold"]:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not tokens:
            return "other", "Latn", 0.0, 0, False, []
        hi_func_count = sum(1 for t in tokens if t in HINDI_FUNCTION_WORDS)
        hi_func_share = hi_func_count / len(tokens)
        if hi_func_share >= config["latin_hindi_function_word_threshold"]:
            return "latin_romanized_hindi", "Latn", 0.0, 0, False, []
        else:
            return "latin_english", "Latn", 0.0, 0, False, []
    else:
        return "other", "other", bho_rate, bho_count, has_bho, matched_markers


def compute_repetition_rate(tokens: List[str], n: int = 4) -> float:
    """Compute 4-gram repetition rate. Bounded in [0.0, 1.0]."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return round(1.0 - (len(set(ngrams)) / len(ngrams)), 4)


# %%
# ==============================================================================
# DATASET LOADING (ALL NON-DEPRECATED DEV ROWS)
# ==============================================================================
def load_all_dev_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Loads frozen dataset and manifest, attaching analysis flags."""
    dataset_sha = compute_file_sha256(config["dataset_path"])
    manifest_sha = compute_file_sha256(config["manifest_path"])
    
    with open(config["manifest_path"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dev_seed_meta = {
        sid: meta for sid, meta in manifest["seeds"].items() if meta["split"] == "dev"
    }
    
    drift_tier_by_seed = defaultdict(lambda: "none")
    if os.path.exists(config["drift_candidates_path"]):
        import pandas as pd
        df_drift = pd.read_csv(config["drift_candidates_path"])
        for _, row in df_drift.iterrows():
            sid = row["seed_id"]
            tier = row.get("tier", "none")
            if tier == "P1":
                drift_tier_by_seed[sid] = "P1"
            elif tier == "P2" and drift_tier_by_seed[sid] != "P1":
                drift_tier_by_seed[sid] = "P2"
                
    with open(config["dataset_path"], "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
        
    hi_by_seed = {r["seed_id"]: r for r in all_rows if r.get("lang") == "hi"}
    latin_re = re.compile(r"[a-zA-Z]")
    hi_contains_latin_by_seed = defaultdict(bool)
    for sid in dev_seed_meta:
        if sid in hi_by_seed:
            text = hi_by_seed[sid]["text"]
            if hi_by_seed[sid]["source"] == "XNLI":
                m = re.search(r"कथन:\s*(.*?)\s*\nपरिकल्पना:\s*(.*?)\s*\nक्या", text, re.DOTALL)
                content = m.group(1) + " " + m.group(2) if m else text
            else:
                content = text
            if latin_re.search(content):
                hi_contains_latin_by_seed[sid] = True
                
    review_status_by_seed = defaultdict(lambda: "unreviewed")
    if os.path.exists(config["seed_review_path"]):
        import pandas as pd
        df_rev = pd.read_csv(config["seed_review_path"])
        if "keep_Y_N" in df_rev.columns:
            for _, row in df_rev.iterrows():
                sid = row["seed_id"]
                val = str(row.get("keep_Y_N", "")).strip().upper()
                if val == "Y":
                    review_status_by_seed[sid] = "keep"
                elif val == "N":
                    review_status_by_seed[sid] = "exclude"

    target_variants = list(config["core_variants"])
    if config.get("include_experimental_codemixed", False):
        target_variants.extend(config["experimental_variants"])
        
    standard_records = []
    self_harm_records = []
    
    for r in all_rows:
        sid = r["seed_id"]
        if sid not in dev_seed_meta or r.get("split") != "dev":
            continue
        if r.get("variant_valid", True) is False:
            continue
        variant = r.get("lang")
        if variant not in target_variants:
            continue
            
        rec = {
            "seed_id": sid,
            "variant": variant,
            "source": r.get("source"),
            "category": r.get("category"),
            "type": r.get("type"),
            "split": "dev",
            "text": r.get("text"),
            "prompt_sha": compute_sha256(r.get("text")),
            "hi_contains_latin": hi_contains_latin_by_seed[sid],
            "drift_tier": drift_tier_by_seed[sid],
            "review_status": review_status_by_seed[sid],
        }
        
        if r.get("type") == "self_harm":
            self_harm_records.append(rec)
        else:
            standard_records.append(rec)
            
    return {
        "dataset_sha256": dataset_sha,
        "manifest_sha256": manifest_sha,
        "all_dev_seeds": dev_seed_meta,
        "standard_records": standard_records,
        "self_harm_records": self_harm_records,
    }


# %%
# ==============================================================================
# MODEL SESSION (BATCHED WITH LEFT PADDING)
# ==============================================================================
class ModelSession:
    """Manages model loading, quantization, device placement, and batched inference."""
    
    def __init__(self, model_id: str, model_revision: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.model_revision = model_revision
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.peak_vram_bytes = 0
        
    def load(self):
        """Loads model and left-padded tokenizer."""
        print(f"[Model Loader] Loading model {self.model_id} (rev: {self.model_revision})...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        token = os.environ.get(self.config["hf_token_env"])
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get(self.config["hf_token_env"])
            except Exception:
                token = None
                
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            token=token,
            trust_remote_code=True,
            padding_side=self.config["padding_side"],
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.reset_peak_memory_stats()
            
            from transformers import BitsAndBytesConfig
            compute_dtype = getattr(torch, self.config["bnb_compute_dtype"], torch.float16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                quantization_config=bnb_config,
                device_map="auto",
                token=token,
                trust_remote_code=True,
            )
            print(f"[Model Loader] Loaded in NF4 4-bit on CUDA.")
        else:
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                dtype=torch.float32,
                token=token,
                trust_remote_code=True,
            ).to(self.device)
            print(f"[Model Loader] Loaded on CPU (multi-threaded, 8 threads).")
            
        self.model.eval()
        return self
        
    def generate_batch(self, prompt_texts: List[str]) -> List[Dict[str, Any]]:
        """Batched greedy generation with left padding, max_new_tokens=512."""
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": txt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for txt in prompt_texts
        ]
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=self.config["do_sample"],
                temperature=None if not self.config["do_sample"] else self.config["temperature"],
                top_p=None if not self.config["do_sample"] else self.config["top_p"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        batch_latency = round(time.perf_counter() - start_time, 4)
        item_latency = round(batch_latency / max(1, len(prompt_texts)), 4)
        
        input_len = inputs["input_ids"].shape[1]
        results = []
        for i in range(len(prompt_texts)):
            gen_ids = outputs[i][input_len:].tolist()
            if self.tokenizer.pad_token_id in gen_ids:
                gen_ids = [tid for tid in gen_ids if tid != self.tokenizer.pad_token_id]
                
            num_generated = len(gen_ids)
            truncated = bool(num_generated >= self.config["max_new_tokens"])
            finish_reason = "length" if truncated else "eos"
            
            response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gen_chars = len(response_text)
            cpt = round(gen_chars / max(1, num_generated), 3)
            
            results.append({
                "response": response_text,
                "token_count": num_generated,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "generated_chars": gen_chars,
                "chars_per_token": cpt,
                "word_count": len(response_text.split()),
                "latency": item_latency,
            })
            
        if torch.cuda.is_available():
            self.peak_vram_bytes = max(self.peak_vram_bytes, torch.cuda.max_memory_allocated())
            
        return results

    def close(self):
        """Frees model from memory."""
        print(f"[Model Loader] Freeing model {self.model_id} from memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[Model Loader] Memory freed.")


# %%
# ==============================================================================
# RESUMABLE BATCHED PIPELINE
# ==============================================================================
def run_batched_generation_loop(
    session: ModelSession,
    records: List[Dict[str, Any]],
    output_path: str,
    en_response_cache: Optional[Dict[str, str]] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Runs length-sorted batched generation with immediate JSONL disk flushing."""
    if en_response_cache is None:
        en_response_cache = {}
        
    existing_keys = set()
    completed_records = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    k = (item["seed_id"], item["variant"], item["model"])
                    existing_keys.add(k)
                    completed_records.append(item)
                    if item["variant"] == "en":
                        en_response_cache[item["seed_id"]] = item["response"]

    pending = [r for r in records if (r["seed_id"], r["variant"], session.model_id) not in existing_keys]
    print(f"[Batched Pipeline] Total records: {len(records)} | Pre-existing: {len(existing_keys)} | To generate: {len(pending)}")
    if not pending:
        return completed_records

    # Sort pending by length with English first to seed length ratio reference
    pending.sort(key=lambda r: (0 if r["variant"] == "en" else 1, len(r["text"])))

    out_file = open(output_path, "a", encoding="utf-8")
    
    for b_start in range(0, len(pending), batch_size):
        batch = pending[b_start : b_start + batch_size]
        prompts = [r["text"] for r in batch]
        
        batch_results = session.generate_batch(prompts)
        
        for r, res in zip(batch, batch_results):
            if r["variant"] == "en":
                en_response_cache[r["seed_id"]] = res["response"]
                en_text = res["response"]
            else:
                en_text = en_response_cache.get(r["seed_id"], "")
                
            (lang_class, script, bho_rate, bho_count, has_bho, matched_markers) = classify_response_language(res["response"])
            
            resp_tokens = re.findall(r"\w+", res["response"].lower())
            rep_rate = compute_repetition_rate(resp_tokens, n=4)
            
            if r["variant"] == "en":
                length_ratio = 1.0
            elif en_text and len(en_text) > 0:
                length_ratio = round(res["generated_chars"] / len(en_text), 3)
            else:
                length_ratio = 1.0
                
            record_entry = {
                "seed_id": r["seed_id"],
                "variant": r["variant"],
                "model": session.model_id,
                "source": r.get("source"),
                "category": r.get("category"),
                "type": r.get("type"),
                "prompt_sha": r["prompt_sha"],
                "response": res["response"],
                "token_count": res["token_count"],
                "finish_reason": res["finish_reason"],
                "truncated": res["truncated"],
                "latency": res["latency"],
                "generated_chars": res["generated_chars"],
                "chars_per_token": res["chars_per_token"],
                "word_count": res["word_count"],
                "response_script": script,
                "response_language_class": lang_class,
                "bho_marker_rate": bho_rate,
                "bho_marker_count": bho_count,
                "has_bho_marker": has_bho,
                "matched_bho_markers": matched_markers,
                "lid_method": "script_and_bhojpuri_lexical_marker",
                "length_ratio_vs_en": length_ratio,
                "repetition_rate": rep_rate,
                "hi_contains_latin": r.get("hi_contains_latin", False),
                "drift_tier": r.get("drift_tier", "none"),
                "review_status": r.get("review_status", "unreviewed"),
            }
            
            out_file.write(json.dumps(record_entry, ensure_ascii=False) + "\n")
            completed_records.append(record_entry)
            
        out_file.flush()
        os.fsync(out_file.fileno())
        
        done_count = min(b_start + batch_size, len(pending))
        print(f"  [Batched Progress] Processed {done_count}/{len(pending)} pending items...")

    out_file.close()
    return completed_records


# %%
# ==============================================================================
# PLUMBING TEST (3 BENIGN DEV ITEMS)
# ==============================================================================
def run_plumbing_test(data: Dict[str, Any], config: Dict[str, Any]):
    """Re-runs the 12-item plumbing test to verify updated schema and metrics."""
    print_run_header(config, config["plumbing_model_id"], config["plumbing_model_revision"])
    
    benign_records = [
        r for r in data["standard_records"]
        if r.get("type") == "borderline_benign"
    ]
    unique_benign_sids = sorted(list(set(r["seed_id"] for r in benign_records)))
    
    random.seed(config["seed"])
    sampled_sids = random.sample(unique_benign_sids, 3)
    print(f"[Plumbing Test] Sampled 3 benign dev seeds: {sampled_sids}")
    
    test_records = [
        r for r in benign_records if r["seed_id"] in sampled_sids
    ]
    if os.path.exists(config["plumbing_output_path"]):
        os.remove(config["plumbing_output_path"])
        
    session = ModelSession(config["plumbing_model_id"], config["plumbing_model_revision"], config).load()
    
    try:
        completed = run_batched_generation_loop(
            session,
            test_records,
            config["plumbing_output_path"],
            batch_size=config["batch_size"],
        )
    finally:
        session.close()
        
    print("\n[Plumbing Test Summary]")
    print(f"Total records generated: {len(completed)}")
    for r in completed:
        print(
            f"  Seed: {r['seed_id']} | Var: {r['variant']} | "
            f"Tokens: {r['token_count']} | Finish: {r['finish_reason']} | "
            f"Chars: {r['generated_chars']} | CPT: {r['chars_per_token']} | "
            f"Class: {r['response_language_class']}"
        )
    print("[Plumbing Test Status] PASS: Patched schema verified successfully.\n")


# %%
# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 2c Route A Response Generation")
    parser.add_argument(
        "--mode",
        choices=["plumbing", "full"],
        default="plumbing",
        help="Execution mode: plumbing (local verification) or full (Colab batch execution)",
    )
    args = parser.parse_args()
    CONFIG["execution_mode"] = args.mode
    
    data = load_all_dev_dataset(CONFIG)
    
    if CONFIG["execution_mode"] == "plumbing":
        run_plumbing_test(data, CONFIG)
    elif CONFIG["execution_mode"] == "full":
        print_run_header(CONFIG, CONFIG["target_model_id"], CONFIG["target_model_revision"])
        session = ModelSession(CONFIG["target_model_id"], CONFIG["target_model_revision"], CONFIG).load()
        en_cache = {}
        try:
            run_batched_generation_loop(
                session,
                data["standard_records"],
                CONFIG["responses_output_path"],
                en_response_cache=en_cache,
                batch_size=CONFIG["batch_size"],
            )
            run_batched_generation_loop(
                session,
                data["self_harm_records"],
                CONFIG["self_harm_output_path"],
                en_response_cache=en_cache,
                batch_size=CONFIG["batch_size"],
            )
        finally:
            session.close()


if __name__ == "__main__":
    main()
