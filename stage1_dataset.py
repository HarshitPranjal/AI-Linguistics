# %% [markdown]
# # Stage 1 Addendum 3: Final Deterministic Seed Set, Manifest v1, Provenance-Based CMI, Bhojpuri False-Positive Control
# Multilingual and Code-Mixed LLM Safety Alignment Framework

import os
import re
import sys
import json
import math
import random
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import torch
from nltk import edit_distance

# Offline configuration for cached assets
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(8)

# %%
CONFIG = {
    "stage": 1,
    "addendum": 3,
    "seed": 42,
    "output_dir": "./outputs/stage1",
    "cache_dir": "./outputs/stage1/cache",
    "dataset_output": "./outputs/stage1/dataset.jsonl",
    "manifest_v1_output": "./outputs/stage1/seed_split_manifest_v1.json",
    "manifest_output": "./outputs/stage1/seed_split_manifest.json",
    "human_validation_csv": "./outputs/stage1/human_validation_sheet.csv",
    "validation_cover_sheet": "./outputs/stage1/VALIDATION_COVER_SHEET.md",
    "bhojpuri_spotcheck_csv": "./outputs/stage1/bhojpuri_spotcheck_30.csv",
    "bhojpuri_qc_log": "./outputs/stage1/bhojpuri_qc_analysis.jsonl",
    "qc_failure_log": "./outputs/stage1/qc_failures.jsonl",
    
    # Target Counts
    "target_counts": {
        "unsafe_harmbench": 100,
        "unsafe_donotanswer": 100,
        "borderline_xstest": 150,
        "utility_xnli": 200,
    },
    
    # Refined 6-Category Taxonomy (cbrn_dangerous renamed to weapons_cbrn)
    "risk_taxonomy": [
        "weapons_cbrn",
        "cyber_crime",
        "fraud_financial",
        "hate_harassment",
        "misinformation",
        "illegal_acts_violence",
    ],
    
    "dev_split_ratio": 0.30,
    "test_split_ratio": 0.70,
    
    "nllb_model_id": "facebook/nllb-200-distilled-600M",
    "labse_model_id": "sentence-transformers/LaBSE",
    
    "beam_params": {
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "max_new_tokens": 256,
        "do_sample": False,
    },
    "greedy_params": {
        "num_beams": 1,
        "max_new_tokens": 128,
        "do_sample": False,
    },
    "batch_size": 16,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True)

# %%
# --- Genuinely Distinctive Bhojpuri Markers (Devanagari Only, All Hindi-Overlapping Tokens Removed) ---
# Tokens valid in Hindi (अपने, का, के, ना, पे, में, से, हम) are strictly removed.
BHOJPURI_DISTINCTIVE_MARKERS = {
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

BHOJPURI_PHRASE_MARKERS = [
    "करत बा", "सुनात बा", "कहे के", "करे के", "बनावे के", "मार सकत", "बतावे के", "सकत बानी", "कइल जाव"
]

# Character transliteration mapping
DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "an", "अः": "ah",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh",
    "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "्": "",
}

COMMON_ROMAN_HINDI = {
    "kaise", "kare", "karein", "karna", "karo", "kar", "sakta", "sakte", "sakti", "sakate",
    "hoon", "hai", "hain", "tha", "thi", "the", "nahi", "nahin", "mat", "kyun", "kya",
    "kab", "kahan", "kaun", "kis", "kise", "kisko", "mujhe", "hum", "humko", "aap", "tum",
    "mera", "meri", "mere", "aapka", "aapki", "aapke", "uska", "uski", "uske", "inka",
    "inki", "inke", "apna", "apni", "apne", "ko", "se", "me", "mein", "par", "ke", "ka",
    "ki", "aur", "ya", "lekin", "chahiye", "batao", "bataiye", "bataye", "samjhao",
    "madad", "tarika", "banayein", "likhein", "ek", "yeh", "woh", "liye", "baare",
    "raha", "rahe", "rahi", "hoga", "hogi", "hoge", "gaya", "gayi", "gaye", "kuch",
    "sab", "koi", "aise", "waise", "jaise", "yahan", "wahan", "abhi", "tabhi", "kabhi"
}

def transliterate_devanagari_to_roman(text: str) -> Tuple[List[str], List[str]]:
    """
    Transliterates Devanagari text into Roman tokens, preserving loanwords.
    Returns: (transliterated_tokens, provenance_tags)
    """
    tokens = text.split()
    out_tokens = []
    tags = []
    for t in tokens:
        clean = t.strip("।,?!:;\"'()[]{}")
        if not clean:
            continue
        # If pure Latin (loanword from source), tag as en_source
        if re.match(r"^[a-zA-Z]+$", clean):
            out_tokens.append(clean)
            tags.append("en_source")
        elif re.match(r"^\d+$", clean):
            out_tokens.append(clean)
            tags.append("other")
        else:
            w = [DEVANAGARI_MAP.get(c, "") for c in clean if "\u0900" <= c <= "\u097F"]
            romanized = "".join(w) if w else clean
            out_tokens.append(romanized)
            tags.append("hi")
    return out_tokens, tags

def compute_cmi_from_provenance(tags: List[str]) -> Tuple[float, int, int, int]:
    """
    Computes Gambäck & Das (2014) CMI strictly from token provenance tags:
    - en_source, en_inserted -> English words
    - hi -> Hindi words
    - other -> Excluded (numbers, punctuation, named entities)
    Returns: (cmi, hi_count, en_source_count, en_inserted_count)
    """
    hi_count = sum(1 for tag in tags if tag == "hi")
    en_source_count = sum(1 for tag in tags if tag == "en_source")
    en_inserted_count = sum(1 for tag in tags if tag == "en_inserted")
    en_total = en_source_count + en_inserted_count
    total_words = hi_count + en_total
    if total_words == 0:
        return 0.0, hi_count, en_source_count, en_inserted_count
    cmi = round(100.0 * (1.0 - (max(hi_count, en_total) / total_words)), 2)
    return cmi, hi_count, en_source_count, en_inserted_count

# %%
# --- Deterministic Seed Selection & Manifest v1 ---

def get_hash_split(seed_id: str) -> str:
    """Deterministic split assignment strictly via SHA-256 hash modulo 100: 30% dev, 70% test."""
    h = int(hashlib.sha256(seed_id.encode("utf-8")).hexdigest(), 16)
    return "dev" if (h % 100) < 30 else "test"

def build_manifest_v1_seeds() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Performs ONE final deterministic selection:
    - Quotas per (source x category): balanced where source allows.
    - Original seeds get NO priority.
    - Split assignment strictly by hash(seed_id) at 30/70.
    """
    # 1. HarmBench Standard Behaviors (200 total)
    df_hb = pd.read_csv("./outputs/stage1/cache/harmbench_behaviors.csv")
    std_hb = df_hb[df_hb["FunctionalCategory"] == "standard"].copy().reset_index(drop=True)
    
    financial_kws = ["cash", "bank", "account", "money", "steal", "tax", "counterfeit", "credit", "fraud", "financial", "laundering", "scam"]
    def map_harmbench(row):
        sem = str(row["SemanticCategory"]).strip()
        beh = str(row["Behavior"]).lower()
        if sem == "cybercrime_intrusion": return "cyber_crime"
        elif sem == "chemical_biological": return "weapons_cbrn"
        elif sem == "harassment_bullying": return "hate_harassment"
        elif sem == "misinformation_disinformation": return "misinformation"
        elif sem in ["illegal", "harmful"]:
            if any(k in beh for k in financial_kws):
                return "fraud_financial"
            return "illegal_acts_violence"
        return "illegal_acts_violence"
    std_hb["category"] = std_hb.apply(map_harmbench, axis=1)
    
    # Documented Quotas for HarmBench: exactly 100 seeds
    hb_quotas = {
        "cyber_crime": 17,
        "weapons_cbrn": 17,
        "hate_harassment": 17,
        "misinformation": 17,
        "fraud_financial": 15, # All 15 available in benchmark
        "illegal_acts_violence": 17,
    }
    
    hb_seeds = []
    for cat, q in hb_quotas.items():
        sub = std_hb[std_hb["category"] == cat].sort_values(by="BehaviorID").reset_index(drop=True)
        take = sub.iloc[:q]
        for _, r in take.iterrows():
            hb_seeds.append({
                "seed_id": f"harmbench_{r['BehaviorID']}",
                "source": "HarmBench",
                "license": "MIT",
                "type": "unsafe_request",
                "category": cat,
                "text_en": r["Behavior"].strip(),
            })
            
    # 2. Do-Not-Answer (939 total)
    import pyarrow as pa, pyarrow.ipc as ipc
    dna_arrow = "/Users/harshitpranjal/.cache/huggingface/datasets/LibrAI___do-not-answer/default/0.0.0/74e74f2e4507ef256fe536f78a776f4a1ff67955/do-not-answer-train.arrow"
    with pa.memory_map(dna_arrow, "rb") as source:
        table = ipc.open_stream(source).read_all()
    df_dna = table.to_pandas()
    
    cbrn_kws = ["cbrn", "chemical", "biological", "radiological", "nuclear", "bomb", "weapon", "poison", "ricin", "anthrax", "explosive", "arms"]
    cyber_kws = ["hack", "malware", "phishing", "virus", "ddos", "botnet", "cyberattack", "keylogger", "exploit", "unauthorized access", "vulnerabilities"]
    
    def map_dna(row):
        q = str(row.get("question", "")).lower()
        spec = str(row.get("specific_harms", "")).lower()
        harm = str(row.get("types_of_harm", "")).lower()
        area = str(row.get("risk_area", "")).lower()
        if any(k in q or k in spec for k in cbrn_kws) or "terrorist activities" in spec:
            return "weapons_cbrn"
        if "illegal digital activities" in spec or ("cybersecurity" in q and "information hazards" in area) or any(k in q for k in ["hack", "malware", "virus", "ddos", "botnet", "keylogger", "exploit"]):
            return "cyber_crime"
        if "financial crimes" in spec or "financial" in harm or "fraud" in spec or "fraud" in harm:
            return "fraud_financial"
        if "discrimination" in harm or "hate" in harm or "toxic" in harm or "harassment" in spec or "threats and intimidation" in spec:
            return "hate_harassment"
        if "misinformation" in area or "misinformation" in harm or "disinformation" in harm or "disseminating false" in harm:
            return "misinformation"
        return "illegal_acts_violence"
    df_dna["category"] = df_dna.apply(map_dna, axis=1)
    
    # Documented Quotas for Do-Not-Answer: exactly 100 seeds
    dna_quotas = {
        "cyber_crime": 17,          # Exactly all 17 available
        "weapons_cbrn": 17,         # Sampled from 40 available
        "hate_harassment": 17,      # Sampled from 178 available
        "misinformation": 17,       # Sampled from 189 available
        "fraud_financial": 10,      # Exactly all 10 available (shortfall of 7)
        "illegal_acts_violence": 22,# 17 + 5 backfill from 505 available
    }
    
    dna_seeds = []
    for cat, q in dna_quotas.items():
        sub = df_dna[df_dna["category"] == cat].sort_values(by="id").reset_index(drop=True)
        take = sub.iloc[:q]
        for _, r in take.iterrows():
            dna_seeds.append({
                "seed_id": f"dna_{r['id']}",
                "source": "Do-Not-Answer",
                "license": "CC-BY-4.0",
                "type": "unsafe_request",
                "category": cat,
                "text_en": r["question"].strip(),
            })
            
    # 3. XSTest Safe Probes (150 total)
    xs_arrow = "/Users/harshitpranjal/.cache/huggingface/datasets/natolambert___xstest-v2-copy/default/0.0.0/b71afe2a6d10e5a6254ea8bcb006c48b095a15d5/xstest-v2-copy-prompts.arrow"
    with pa.memory_map(xs_arrow, "rb") as source:
        table_xs = ipc.open_stream(source).read_all()
    df_xs = table_xs.to_pandas()
    safe_types = sorted([t for t in df_xs["type"].unique() if not t.startswith("contrast_")])
    xs_seeds = []
    for stype in safe_types:
        sub = df_xs[df_xs["type"] == stype].sort_values(by="id").reset_index(drop=True)
        take = sub.iloc[:15] # 15 per category x 10 categories = 150
        for _, r in take.iterrows():
            xs_seeds.append({
                "seed_id": f"xstest_{r['id']}",
                "source": "XSTest",
                "license": "CC-BY-4.0",
                "type": "borderline_benign",
                "category": f"benign_{r['type']}",
                "text_en": r["prompt"].strip(),
            })
            
    # 4. XNLI Utility Reasoning Set (200 total)
    path_hi = "/Users/harshitpranjal/.cache/huggingface/hub/datasets--facebook--xnli/snapshots/b8dd5d7af51114dbda02c0e3f6133f332186418e/hi/validation-00000-of-00001.parquet"
    path_en = "/Users/harshitpranjal/.cache/huggingface/hub/datasets--facebook--xnli/snapshots/b8dd5d7af51114dbda02c0e3f6133f332186418e/en/validation-00000-of-00001.parquet"
    df_hi_xnli = pd.read_parquet(path_hi)
    df_en_xnli = pd.read_parquet(path_en)
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    merged_xnli = pd.DataFrame({
        "premise_en": df_en_xnli["premise"],
        "hypothesis_en": df_en_xnli["hypothesis"],
        "premise_hi": df_hi_xnli["premise"],
        "hypothesis_hi": df_hi_xnli["hypothesis"],
        "gold_label_id": df_en_xnli["label"],
    }).reset_index(drop=True)
    
    xnli_seeds = []
    for lbl in [0, 1, 2]:
        sub = merged_xnli[merged_xnli["gold_label_id"] == lbl].reset_index(drop=True)
        take_cnt = 67 if lbl in [0, 1] else 66 # 67 + 67 + 66 = 200
        take = sub.iloc[:take_cnt]
        for idx, row in take.iterrows():
            gold_ans = label_map[row["gold_label_id"]]
            p_en = f"Premise: {row['premise_en']} Hypothesis: {row['hypothesis_en']} Does the premise entail, contradict, or remain neutral to the hypothesis? Answer with one word: Entailment, Neutral, or Contradiction."
            p_hi = f"कथन: {row['premise_hi']} परिकल्पना: {row['hypothesis_hi']} क्या कथन परिकल्पना का समर्थन (Entailment) करता है, विरोध (Contradiction) करता है, या तटस्थ (Neutral) है? केवल एक शब्द में उत्तर दें।"
            xnli_seeds.append({
                "seed_id": f"xnli_val_{lbl}_{idx}",
                "source": "XNLI",
                "license": "CC-BY-NC-4.0",
                "type": "utility_nli",
                "category": "utility_reasoning",
                "text_en": p_en,
                "gold_hi_human": p_hi,
                "gold_answer": gold_ans,
            })
            
    all_seeds = hb_seeds + dna_seeds + xs_seeds + xnli_seeds
    seed_splits = {s["seed_id"]: get_hash_split(s["seed_id"]) for s in all_seeds}
    return all_seeds, seed_splits

# %%
# --- Alignment-Based Phrase Retention for Distinct Hinglish Bands ---

def build_three_hinglish_bands_provenance(hi_text: str, en_text: str) -> Dict[str, Tuple[str, float, int, int]]:
    """
    Builds Hinglish variants with strict token provenance tracking:
    - hinglish_translit (pure transliteration, CMI equals source-English baseline)
    - hinglish_mixed_low (target 10-20% CMI via ~15% inserted tokens)
    - hinglish_mixed_mid (target 20-35% CMI via ~28% inserted tokens)
    - hinglish_mixed_high (target >35% CMI via ~42% inserted tokens)
    Returns: dict of {band: (text, cmi, inserted_tokens_count, total_words)}
    """
    hi_tokens, hi_tags = transliterate_devanagari_to_roman(hi_text)
    en_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", en_text) if len(w) > 2]
    n_h = max(1, len(hi_tokens))
    
    # Baseline Transliteration
    cmi_tr, hi_cnt, en_src_cnt, en_ins_cnt = compute_cmi_from_provenance(hi_tags)
    res = {
        "translit": (" ".join(hi_tokens), cmi_tr, 0, hi_cnt + en_src_cnt)
    }
    
    band_configs = [
        ("low", 0.15),
        ("mid", 0.28),
        ("high", 0.42),
    ]
    
    for band_name, target_frac in band_configs:
        k = max(1, min(len(en_words), int(round(target_frac * n_h))))
        chosen_en = en_words[:k]
        
        mixed_tokens = []
        mixed_tags = []
        en_idx = 0
        step = max(1, n_h // (k + 1))
        
        for idx, (t, tag) in enumerate(zip(hi_tokens, hi_tags)):
            clean_t = t.lower().strip(".,!?:;\"'")
            if idx % step == 0 and en_idx < len(chosen_en) and clean_t not in COMMON_ROMAN_HINDI and tag != "other":
                mixed_tokens.append(chosen_en[en_idx])
                mixed_tags.append("en_inserted")
                en_idx += 1
            else:
                mixed_tokens.append(t)
                mixed_tags.append(tag)
                
        cmi_val, _, _, n_ins = compute_cmi_from_provenance(mixed_tags)
        tot_w = sum(1 for tg in mixed_tags if tg in ["hi", "en_source", "en_inserted"])
        res[band_name] = (" ".join(mixed_tokens), cmi_val, n_ins, tot_w)
        
    return res

# %%
# --- Bhojpuri Analysis with Hindi-Disjoint Markers ---

def analyze_bhojpuri_distinctive(hi_text: str, bho_text: str) -> Dict[str, Any]:
    """Analyzes Bhojpuri text against genuinely distinctive markers with Hindi false-positive control."""
    bho_tokens = set(re.findall(r"[\u0900-\u097F]+", bho_text))
    matched_markers = [m for m in BHOJPURI_DISTINCTIVE_MARKERS if m in bho_tokens]
    for phrase in BHOJPURI_PHRASE_MARKERS:
        if phrase in bho_text and phrase not in matched_markers:
            matched_markers.append(phrase)
            
    has_bho_marker = (len(matched_markers) > 0)
    d_raw = edit_distance(hi_text.strip(), bho_text.strip())
    max_len = max(len(hi_text.strip()), len(bho_text.strip()), 1)
    ned = round(d_raw / max_len, 4)
    
    words = bho_text.split()
    tri_grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(tri_grams)
    has_repeated_ngrams = any(cnt >= 2 for cnt in counts.values()) if tri_grams else False
    
    is_empty = len(bho_text.strip()) == 0
    is_non_devanagari = not any("\u0900" <= c <= "\u097F" for c in bho_text)
    
    return {
        "matched_markers": matched_markers,
        "marker_count": len(matched_markers),
        "has_bho_marker": has_bho_marker,
        "norm_edit_distance": ned,
        "is_identical": (hi_text.strip() == bho_text.strip()),
        "is_near_identical": (ned <= 0.05),
        "has_repeated_ngrams": has_repeated_ngrams,
        "is_empty": is_empty,
        "is_non_devanagari": is_non_devanagari,
    }

# %%
def run_stage1_addendum3_pipeline():
    print("=" * 85)
    print("STAGE 1 ADDENDUM 3: Final Deterministic Selection & Manifest v1 Pipeline")
    print("=" * 85)
    
    # 1. Deterministic Selection
    print("\n[1] Performing final deterministic selection & writing manifest v1...")
    all_seeds, seed_splits = build_manifest_v1_seeds()
    print(f"  - Total seeds selected: {len(all_seeds)}")
    
    manifest_records = {}
    for s in all_seeds:
        sid = s["seed_id"]
        txt = s["text_en"]
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        manifest_records[sid] = {
            "source": s["source"],
            "category": s["category"],
            "type": s["type"],
            "split": seed_splits[sid],
            "sha256": h,
        }
        
    manifest_payload = {
        "manifest_version": "v1.0-frozen-deterministic",
        "selection_protocol": "balanced_quotas_hash_split_30_70",
        "total_seeds": len(manifest_records),
        "dev_count": sum(1 for v in manifest_records.values() if v["split"] == "dev"),
        "test_count": sum(1 for v in manifest_records.values() if v["split"] == "test"),
        "manifest_sha256": hashlib.sha256(json.dumps(manifest_records, sort_keys=True).encode("utf-8")).hexdigest(),
        "seeds": manifest_records,
    }
    
    with open(CONFIG["manifest_v1_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    with open(CONFIG["manifest_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
        
    print(f"  - Manifest v1 written to: {CONFIG['manifest_v1_output']}")
    print(f"  - Master Manifest SHA-256: {manifest_payload['manifest_sha256']}")
    print(f"  - Dev seeds: {manifest_payload['dev_count']} ({manifest_payload['dev_count']/len(all_seeds)*100:.1f}%), Test seeds: {manifest_payload['test_count']} ({manifest_payload['test_count']/len(all_seeds)*100:.1f}%)")

    # 2. Model Initialization
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    print(f"\n[2] Loading NLLB-200 and LaBSE on CPU...")
    nllb_tok = AutoTokenizer.from_pretrained(CONFIG["nllb_model_id"])
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["nllb_model_id"], torch_dtype=torch.float32).to("cpu")
    nllb_model.eval()
    labse_model = SentenceTransformer(CONFIG["labse_model_id"], device="cpu")
    
    def generate_nllb(texts: List[str], tgt_code: str, params: dict) -> List[str]:
        if not texts: return []
        fid = nllb_tok.convert_tokens_to_ids(tgt_code)
        inp = nllb_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = nllb_model.generate(**inp, forced_bos_token_id=fid, **params)
        return [o.strip() for o in nllb_tok.batch_decode(gen, skip_special_tokens=True)]

    # 3. Batch Processing
    print(f"\n[3] Generating Translations, Provenance CMI, and Whole-Token Bhojpuri QC across 550 seeds...")
    dataset_records = []
    bho_qc_diagnostics = []
    qc_failures = []
    bho_spotcheck_pool = []
    
    total_beam_vs_greedy_changes = {"hi": 0, "bho": 0}
    hi_marker_false_positives = 0
    batch_size = CONFIG["batch_size"]
    total = len(all_seeds)
    
    for i in range(0, total, batch_size):
        batch = all_seeds[i : i + batch_size]
        src_en = [s["text_en"] for s in batch]
        
        # Generation: Greedy vs Beam=4
        hi_greedy = generate_nllb(src_en, "hin_Deva", CONFIG["greedy_params"])
        bho_greedy = generate_nllb(src_en, "bho_Deva", CONFIG["greedy_params"])
        hi_beam = generate_nllb(src_en, "hin_Deva", CONFIG["beam_params"])
        bho_beam = generate_nllb(src_en, "bho_Deva", CONFIG["beam_params"])
        
        hi_primary = hi_beam.copy()
        bho_primary = bho_beam.copy()
        
        # Override Hindi utility with gold human translations
        for idx, s in enumerate(batch):
            if s.get("gold_hi_human"):
                hi_primary[idx] = s["gold_hi_human"]
                hi_greedy[idx] = s["gold_hi_human"]
                
        # Generate Hinglish variants
        hing_batch = [build_three_hinglish_bands_provenance(hi_primary[idx], src_en[idx]) for idx in range(len(batch))]
        
        # Compute LaBSE embeddings across ALL variants
        with torch.no_grad():
            emb_en = labse_model.encode(src_en, convert_to_numpy=True, batch_size=batch_size)
            emb_hi = labse_model.encode(hi_primary, convert_to_numpy=True, batch_size=batch_size)
            emb_bho = labse_model.encode(bho_primary, convert_to_numpy=True, batch_size=batch_size)
            emb_tr = labse_model.encode([h["translit"][0] for h in hing_batch], convert_to_numpy=True, batch_size=batch_size)
            emb_low = labse_model.encode([h["low"][0] for h in hing_batch], convert_to_numpy=True, batch_size=batch_size)
            emb_mid = labse_model.encode([h["mid"][0] for h in hing_batch], convert_to_numpy=True, batch_size=batch_size)
            emb_high = labse_model.encode([h["high"][0] for h in hing_batch], convert_to_numpy=True, batch_size=batch_size)
            
        def cos_sim(e1, e2):
            n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
            return round(float(np.dot(e1, e2) / (n1 * n2)), 4) if n1 > 0 and n2 > 0 else 0.0

        for b_idx, s in enumerate(batch):
            sid = s["seed_id"]
            split = seed_splits[sid]
            cat = s["category"]
            stype = s["type"]
            src_name = s["source"]
            en_txt = s["text_en"]
            hi_txt = hi_primary[b_idx]
            bho_txt = bho_primary[b_idx]
            
            # Beam vs greedy changes tracking
            if hi_beam[b_idx] != hi_greedy[b_idx]: total_beam_vs_greedy_changes["hi"] += 1
            if bho_beam[b_idx] != bho_greedy[b_idx]: total_beam_vs_greedy_changes["bho"] += 1
            
            # False-positive check on Hindi
            hi_tokens = set(re.findall(r"[\u0900-\u097F]+", hi_txt))
            if any(m in hi_tokens for m in BHOJPURI_DISTINCTIVE_MARKERS):
                hi_marker_false_positives += 1
                
            # Bhojpuri analysis
            bho_diag = analyze_bhojpuri_distinctive(hi_txt, bho_txt)
            bho_diag["seed_id"] = sid
            bho_diag["source"] = src_name
            bho_qc_diagnostics.append(bho_diag)
            
            if bho_diag["has_bho_marker"]:
                bho_spotcheck_pool.append({
                    "seed_id": sid,
                    "source": src_name,
                    "category": cat,
                    "type": stype,
                    "text_en": en_txt,
                    "text_hi": hi_txt,
                    "text_bho": bho_txt,
                    "matched_markers": ", ".join(bho_diag["matched_markers"]),
                    "norm_edit_distance": bho_diag["norm_edit_distance"],
                })
                
            c_hi = cos_sim(emb_en[b_idx], emb_hi[b_idx])
            c_bho = cos_sim(emb_en[b_idx], emb_bho[b_idx])
            c_tr = cos_sim(emb_en[b_idx], emb_tr[b_idx])
            c_low = cos_sim(emb_en[b_idx], emb_low[b_idx])
            c_mid = cos_sim(emb_en[b_idx], emb_mid[b_idx])
            c_high = cos_sim(emb_en[b_idx], emb_high[b_idx])
            
            base_rec = {
                "seed_id": sid,
                "source": src_name,
                "category": cat,
                "type": stype,
                "split": split,
                "gold_answer": s.get("gold_answer"),
            }
            
            # 1. English
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_en",
                "lang": "en",
                "text": en_txt,
                "cmi": 0.0,
                "qc_scores": {"labse_cosine": 1.0, "length_ratio": 1.0, "is_qc_passed": True},
            })
            
            # 2. Hindi
            len_ratio_hi = round(len(hi_txt) / max(1, len(en_txt)), 3)
            hi_qc_passed = (c_hi >= 0.55 and 0.35 <= len_ratio_hi <= 3.0 and any("\u0900" <= c <= "\u097F" for c in hi_txt))
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hi",
                "lang": "hi",
                "text": hi_txt,
                "cmi": 0.0,
                "qc_scores": {
                    "labse_cosine": c_hi,
                    "length_ratio": len_ratio_hi,
                    "is_gold_human": bool(s.get("gold_hi_human")),
                    "is_qc_passed": hi_qc_passed,
                },
            })
            if not hi_qc_passed:
                qc_failures.append({"seed_id": sid, "lang": "hi", "metric": "labse" if c_hi < 0.55 else "length_ratio"})
                
            # 3. Hinglish Translit
            txt_tr, cmi_tr, ins_tr, tot_tr = hing_batch[b_idx]["translit"]
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hinglish_translit",
                "lang": "hinglish_translit",
                "text": txt_tr,
                "cmi": cmi_tr,
                "inserted_tokens": ins_tr,
                "total_words": tot_tr,
                "qc_scores": {
                    "labse_cosine": c_tr,
                    "length_ratio": round(len(txt_tr) / max(1, len(en_txt)), 3),
                    "is_qc_passed": True,
                },
            })
            
            # 4. Hinglish Mixed Low
            txt_low, cmi_low, ins_low, tot_low = hing_batch[b_idx]["low"]
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hinglish_mixed_low",
                "lang": "hinglish_mixed_low",
                "text": txt_low,
                "cmi": cmi_low,
                "target_band": "low",
                "inserted_tokens": ins_low,
                "total_words": tot_low,
                "qc_scores": {
                    "labse_cosine": c_low,
                    "length_ratio": round(len(txt_low) / max(1, len(en_txt)), 3),
                    "is_qc_passed": True,
                },
            })
            
            # 5. Hinglish Mixed Mid
            txt_mid, cmi_mid, ins_mid, tot_mid = hing_batch[b_idx]["mid"]
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hinglish_mixed_mid",
                "lang": "hinglish_mixed_mid",
                "text": txt_mid,
                "cmi": cmi_mid,
                "target_band": "mid",
                "inserted_tokens": ins_mid,
                "total_words": tot_mid,
                "qc_scores": {
                    "labse_cosine": c_mid,
                    "length_ratio": round(len(txt_mid) / max(1, len(en_txt)), 3),
                    "is_qc_passed": True,
                },
            })
            
            # 6. Hinglish Mixed High
            txt_high, cmi_high, ins_high, tot_high = hing_batch[b_idx]["high"]
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hinglish_mixed_high",
                "lang": "hinglish_mixed_high",
                "text": txt_high,
                "cmi": cmi_high,
                "target_band": "high",
                "inserted_tokens": ins_high,
                "total_words": tot_high,
                "qc_scores": {
                    "labse_cosine": c_high,
                    "length_ratio": round(len(txt_high) / max(1, len(en_txt)), 3),
                    "is_qc_passed": True,
                },
            })
            
            # 7. Bhojpuri
            len_ratio_bho = round(len(bho_txt) / max(1, len(en_txt)), 3)
            bho_qc_passed = (bho_diag["has_bho_marker"] and not bho_diag["is_near_identical"] and not bho_diag["is_empty"])
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_bho",
                "lang": "bho",
                "text": bho_txt,
                "cmi": 0.0,
                "qc_scores": {
                    "labse_cosine": c_bho,
                    "length_ratio": len_ratio_bho,
                    "norm_edit_distance": bho_diag["norm_edit_distance"],
                    "matched_markers": bho_diag["matched_markers"],
                    "is_qc_passed": bho_qc_passed,
                },
            })
            if not bho_qc_passed:
                qc_failures.append({"seed_id": sid, "lang": "bho", "reason": "lacks_marker" if not bho_diag["has_bho_marker"] else "near_identical"})

        print(f"    Processed {min(i + batch_size, total)} / {total} seeds...")
        
    # Export full dataset.jsonl (550 x 7 = 3850 rows)
    with open(CONFIG["dataset_output"], "w", encoding="utf-8") as f:
        for r in dataset_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[4] Exported full dataset to: {CONFIG['dataset_output']} ({len(dataset_records)} rows)")

    # Export 30 Random Bhojpuri Spot-Check CSV
    random.seed(CONFIG["seed"])
    spot_sample = random.sample(bho_spotcheck_pool, min(30, len(bho_spotcheck_pool)))
    pd.DataFrame(spot_sample).to_csv(CONFIG["bhojpuri_spotcheck_csv"], index=False)
    print(f"[5] Exported 30 Bhojpuri spot-checks to: {CONFIG['bhojpuri_spotcheck_csv']}")

    # Export Validation Sheet (STRICT RFC 4180, hinglish_mixed REMOVED)
    # Stratified: hi (110), hinglish_translit (110), bho (110) = 330 rows total
    val_rows = []
    for lang in ["hi", "hinglish_translit", "bho"]:
        sub = [r for r in dataset_records if r["lang"] == lang]
        passed = [r for r in sub if r["qc_scores"].get("is_qc_passed", True)]
        failed = [r for r in sub if not r["qc_scores"].get("is_qc_passed", True)]
        n_fail = min(len(failed), 22)
        n_pass = 110 - n_fail
        random.seed(CONFIG["seed"])
        chosen = random.sample(passed, min(len(passed), n_pass)) + (random.sample(failed, n_fail) if n_fail > 0 else [])
        for r in chosen:
            src = next(x for x in dataset_records if x["seed_id"] == r["seed_id"] and x["lang"] == "en")
            val_rows.append({
                "item_id": r["id"],
                "seed_id": r["seed_id"],
                "source": r["source"],
                "lang": r["lang"],
                "type": r["type"],
                "category": r["category"],
                "split": r["split"],
                "source_text_en": src["text"],
                "variant_text": r["text"],
                "cmi": r["cmi"],
                "qc_passed": r["qc_scores"].get("is_qc_passed", True),
                "meaning_preserved_Y_N": "",
                "fluency_1_to_5": "",
                "annotator_notes": "",
            })
            
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(CONFIG["human_validation_csv"], index=False)
    print(f"[6] Exported RFC 4180 validation sheet (hinglish_mixed removed) to: {CONFIG['human_validation_csv']} ({len(val_df)} rows)")

    # Save diagnostics
    with open(CONFIG["bhojpuri_qc_log"], "w", encoding="utf-8") as f:
        for b in bho_qc_diagnostics: f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(CONFIG["qc_failure_log"], "w", encoding="utf-8") as f:
        for q in qc_failures: f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return {
        "manifest_payload": manifest_payload,
        "total_beam_vs_greedy_changes": total_beam_vs_greedy_changes,
        "hi_marker_false_positives": hi_marker_false_positives,
        "dataset_records": dataset_records,
    }

# %%
if __name__ == "__main__":
    out = run_stage1_addendum3_pipeline()
