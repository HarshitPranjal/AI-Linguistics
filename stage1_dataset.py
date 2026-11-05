# %% [markdown]
# # Stage 1 Addendum 4: Manifest v2 Seeded Selection, DNA Eligibility Filter, Native Taxonomy, Stratified Hash Split
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
    "addendum": 4,
    "seed": 42,
    "output_dir": "./outputs/stage1",
    "cache_dir": "./outputs/stage1/cache",
    "dataset_output": "./outputs/stage1/dataset.jsonl",
    "manifest_v2_output": "./outputs/stage1/seed_split_manifest_v2.json",
    "manifest_output": "./outputs/stage1/seed_split_manifest.json",
    "human_validation_csv": "./outputs/stage1/human_validation_sheet.csv",
    "validation_cover_sheet": "./outputs/stage1/VALIDATION_COVER_SHEET.md",
    "bhojpuri_spotcheck_csv": "./outputs/stage1/bhojpuri_spotcheck_30.csv",
    "bhojpuri_qc_log": "./outputs/stage1/bhojpuri_qc_analysis.jsonl",
    "qc_failure_log": "./outputs/stage1/qc_failures.jsonl",
    
    # Target Quotas per Core Category Cell
    "cell_quota": 20,
    
    # HarmBench Native Semantic Categories (used AS-IS, no keyword heuristics)
    "harmbench_native_categories": [
        "chemical_biological",
        "cybercrime_intrusion",
        "harassment_bullying",
        "illegal",
        "misinformation_disinformation",
    ],
    
    "dev_split_ratio": 0.30,
    
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
# --- Genuinely Distinctive Bhojpuri Markers (Hindi-Overlapping Tokens Removed) ---
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

DEVANAGARI_REV_MAP = {
    "aa": "आ", "ai": "ऐ", "au": "औ", "ee": "ई", "oo": "ऊ", "an": "अं", "ah": "अः",
    "kh": "ख", "gh": "घ", "ng": "ङ", "chh": "छ", "jh": "झ", "ny": "ञ",
    "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ", "sh": "श",
    "a": "अ", "i": "इ", "u": "उ", "e": "ए", "o": "ओ",
    "k": "क", "g": "ग", "c": "च", "j": "ज", "t": "त", "d": "द", "n": "न",
    "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व",
    "s": "स", "h": "ह", "z": "ज़", "q": "क़", "f": "फ़"
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
    tokens = text.split()
    out_tokens = []
    tags = []
    for t in tokens:
        clean = t.strip("।,?!:;\"'()[]{}")
        if not clean:
            continue
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

def roman_to_devanagari_roundtrip(text: str) -> str:
    tokens = text.lower().split()
    out = []
    for tok in tokens:
        clean = tok.strip(".,!?:;\"'()[]{}")
        i = 0
        w = []
        while i < len(clean):
            matched = False
            for length in [3, 2, 1]:
                sub = clean[i:i+length]
                if sub in DEVANAGARI_REV_MAP:
                    w.append(DEVANAGARI_REV_MAP[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                w.append(clean[i])
                i += 1
        out.append("".join(w))
    return " ".join(out)

def compute_cmi_from_provenance(tags: List[str]) -> Tuple[float, int, int, int]:
    hi_count = sum(1 for tag in tags if tag == "hi")
    en_source_count = sum(1 for tag in tags if tag == "en_source")
    en_inserted_count = sum(1 for tag in tags if tag == "en_inserted")
    en_total = en_source_count + en_inserted_count
    total_words = hi_count + en_total
    if total_words == 0:
        return 0.0, hi_count, en_source_count, en_inserted_count
    cmi = round(100.0 * (1.0 - (max(hi_count, en_total) / total_words)), 2)
    return cmi, hi_count, en_source_count, en_inserted_count

def build_three_hinglish_bands_provenance(hi_text: str, en_text: str) -> Dict[str, Tuple[str, float, int, int]]:
    hi_tokens, hi_tags = transliterate_devanagari_to_roman(hi_text)
    en_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", en_text) if len(w) > 2]
    n_h = max(1, len(hi_tokens))
    
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
# --- Seeded Sampling Rule, Exact Deduplication, and LaBSE Cluster Capping ---

def hash_seed_rank_v2(seed_id: str) -> str:
    """Deterministic pseudo-random seeded sampling rank within each cell."""
    return hashlib.sha256((seed_id + "v2").encode("utf-8")).hexdigest()

def hash_split_rank(seed_id: str) -> int:
    """Deterministic hash rank for stratified 30/70 dev/test split."""
    return int(hashlib.sha256(seed_id.encode("utf-8")).hexdigest(), 16)

def load_manifest_v2_seeds(labse_model) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
    """
    Constructs Manifest v2:
    - HarmBench SemanticCategory as-is (chemical_biological includes drug synthesis; car theft is illegal).
    - Explicit DNA native types table; moves individual-privacy, org-info, adult, chatbot-human to unscored pool.
    - Moves self-harm to dedicated 'self_harm' category.
    - Seeded sampling rank sha256(seed_id + 'v2').
    - Exact text deduplication and LaBSE cluster capping <= 2 within cells.
    - Reports shortfalls (no cross-category backfilling).
    - Stratified hash dev/test split: first round(0.3 * n) per cell to dev.
    """
    removals = []
    shortfalls = {}
    
    # 1. HarmBench standard behaviors
    df_hb = pd.read_csv("./outputs/stage1/cache/harmbench_behaviors.csv")
    std_hb = df_hb[df_hb["FunctionalCategory"] == "standard"].copy().reset_index(drop=True)
    
    hb_selected = []
    seen_hb_texts = set()
    
    for cat in CONFIG["harmbench_native_categories"]:
        sub = std_hb[std_hb["SemanticCategory"] == cat].copy()
        sub["seed_id"] = sub["BehaviorID"].apply(lambda bid: f"harmbench_{bid}")
        sub["hash_rank"] = sub["seed_id"].apply(hash_seed_rank_v2)
        sub = sub.sort_values(by="hash_rank").reset_index(drop=True)
        
        emb = labse_model.encode(sub["Behavior"].tolist(), convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        
        cell_chosen = []
        cell_emb = []
        
        for idx, r in sub.iterrows():
            txt = r["Behavior"].strip()
            txt_lower = txt.lower()
            
            # Dedupe exact text
            if txt_lower in seen_hb_texts:
                removals.append({
                    "seed_id": r["seed_id"],
                    "source": "HarmBench",
                    "category": cat,
                    "reason": "exact_text_duplicate",
                    "text": txt,
                })
                continue
                
            e = emb[idx]
            near_dup_count = 0
            matching_ids = []
            for c_idx, c_emb in enumerate(cell_emb):
                sim = float(np.dot(e, c_emb))
                if sim > 0.90:
                    near_dup_count += 1
                    matching_ids.append(cell_chosen[c_idx]["seed_id"])
                    
            if near_dup_count >= 2:
                removals.append({
                    "seed_id": r["seed_id"],
                    "source": "HarmBench",
                    "category": cat,
                    "reason": f"cluster_cap_exceeded (matched {matching_ids})",
                    "text": txt,
                })
                continue
                
            seen_hb_texts.add(txt_lower)
            cell_chosen.append({
                "seed_id": r["seed_id"],
                "source": "HarmBench",
                "license": "MIT",
                "type": "unsafe_request",
                "category": cat,
                "text_en": txt,
            })
            cell_emb.append(e)
            if len(cell_chosen) == CONFIG["cell_quota"]:
                break
                
        hb_selected.extend(cell_chosen)
        sf = CONFIG["cell_quota"] - len(cell_chosen)
        if sf > 0:
            shortfalls[f"HarmBench_{cat}"] = sf
            
    # 2. Do-Not-Answer Explicit Classification
    import pyarrow as pa, pyarrow.ipc as ipc
    dna_arrow = "/Users/harshitpranjal/.cache/huggingface/datasets/LibrAI___do-not-answer/default/0.0.0/74e74f2e4507ef256fe536f78a776f4a1ff67955/do-not-answer-train.arrow"
    with pa.memory_map(dna_arrow, "rb") as source:
        table = ipc.open_stream(source).read_all()
    df_dna = table.to_pandas()
    
    unscored_types = [
        "Compromise privacy by leaking or inferring private information (person/individual)",
        "Risks from leaking or inferring sensitive information (organization/gov)",
        "Treat Chatbot as a Human",
        "Adult Content",
    ]
    
    def classify_dna(row):
        t_harm = row["types_of_harm"]
        s_harm = str(row.get("specific_harms", "")).split(":")[0].strip()
        if t_harm in unscored_types:
            return "unscored_pool"
        if t_harm == "Mental Health or Overreliance Crisis":
            return "self_harm" if s_harm == "Self-Harm" else "unscored_pool"
        if t_harm == "Assisting illegal activities":
            if s_harm == "Illegal Digital Activities": return "cybercrime_intrusion"
            elif s_harm in ["Illegal Drugs and Substances", "Terrorist Activities"]: return "chemical_biological"
            elif s_harm == "Threats and Intimidation": return "harassment_bullying"
            else: return "illegal"
        if t_harm in ["Social stereotypes and unfair discrimination", "Toxic language (hate speech)"]:
            return "harassment_bullying"
        if t_harm in [
            "Disseminating false or misleading information",
            "Causing material harm by disseminating misinformation e.g. in medicine or law",
            "Reducing the cost of disinformation campaigns"
        ]:
            return "misinformation_disinformation"
        if t_harm == "Nudging or advising users to perform unethical or unsafe actions":
            if s_harm in ["Defamatory Content", "Trolling or Flame baiting", "Endorsement of Cyberbullying or Online Harassment"]:
                return "harassment_bullying"
            elif s_harm == "Endorsement of Unsafe Health Practices":
                return "misinformation_disinformation"
            else:
                return "illegal"
        return "unscored_pool"

    df_dna["mapped_cat"] = df_dna.apply(classify_dna, axis=1)
    
    dna_categories = [
        "chemical_biological",
        "cybercrime_intrusion",
        "harassment_bullying",
        "illegal",
        "misinformation_disinformation",
        "self_harm",
    ]
    
    dna_selected = []
    seen_dna_texts = set()
    
    for cat in dna_categories:
        sub = df_dna[df_dna["mapped_cat"] == cat].copy()
        sub["seed_id"] = sub["id"].apply(lambda did: f"dna_{did}")
        sub["hash_rank"] = sub["seed_id"].apply(hash_seed_rank_v2)
        sub = sub.sort_values(by="hash_rank").reset_index(drop=True)
        
        emb = labse_model.encode(sub["question"].tolist(), convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        
        cell_chosen = []
        cell_emb = []
        
        for idx, r in sub.iterrows():
            txt = r["question"].strip()
            txt_lower = txt.lower()
            
            # Dedupe exact text
            if txt_lower in seen_dna_texts:
                removals.append({
                    "seed_id": r["seed_id"],
                    "source": "Do-Not-Answer",
                    "category": cat,
                    "reason": "exact_text_duplicate",
                    "text": txt,
                })
                continue
                
            e = emb[idx]
            near_dup_count = 0
            matching_ids = []
            for c_idx, c_emb in enumerate(cell_emb):
                sim = float(np.dot(e, c_emb))
                if sim > 0.90:
                    near_dup_count += 1
                    matching_ids.append(cell_chosen[c_idx]["seed_id"])
                    
            if near_dup_count >= 2:
                removals.append({
                    "seed_id": r["seed_id"],
                    "source": "Do-Not-Answer",
                    "category": cat,
                    "reason": f"cluster_cap_exceeded (matched {matching_ids})",
                    "text": txt,
                })
                continue
                
            seen_dna_texts.add(txt_lower)
            cell_chosen.append({
                "seed_id": r["seed_id"],
                "source": "Do-Not-Answer",
                "license": "CC-BY-4.0",
                "type": "self_harm" if cat == "self_harm" else "unsafe_request",
                "category": cat,
                "text_en": txt,
            })
            cell_emb.append(e)
            if len(cell_chosen) == CONFIG["cell_quota"]:
                break
                
        dna_selected.extend(cell_chosen)
        sf = CONFIG["cell_quota"] - len(cell_chosen)
        if sf > 0:
            shortfalls[f"Do-Not-Answer_{cat}"] = sf

    # 3. XSTest Safe Probes (150 total)
    xs_arrow = "/Users/harshitpranjal/.cache/huggingface/datasets/natolambert___xstest-v2-copy/default/0.0.0/b71afe2a6d10e5a6254ea8bcb006c48b095a15d5/xstest-v2-copy-prompts.arrow"
    with pa.memory_map(xs_arrow, "rb") as source:
        table_xs = ipc.open_stream(source).read_all()
    df_xs = table_xs.to_pandas()
    safe_types = sorted([t for t in df_xs["type"].unique() if not t.startswith("contrast_")])
    xs_selected = []
    for stype in safe_types:
        sub = df_xs[df_xs["type"] == stype].copy()
        sub["seed_id"] = sub["id"].apply(lambda xid: f"xstest_{xid}")
        sub["hash_rank"] = sub["seed_id"].apply(hash_seed_rank_v2)
        sub = sub.sort_values(by="hash_rank").reset_index(drop=True)
        take = sub.iloc[:15]
        for _, r in take.iterrows():
            xs_selected.append({
                "seed_id": r["seed_id"],
                "source": "XSTest",
                "license": "CC-BY-4.0",
                "type": "borderline_benign",
                "category": f"benign_{r['type']}",
                "text_en": r["prompt"].strip(),
            })

    # 4. XNLI Utility Reasoning (200 total)
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
    
    xnli_selected = []
    for lbl in [0, 1, 2]:
        sub = merged_xnli[merged_xnli["gold_label_id"] == lbl].reset_index(drop=True)
        take_cnt = 67 if lbl in [0, 1] else 66
        take = sub.iloc[:take_cnt]
        for idx, row in take.iterrows():
            gold_ans = label_map[row["gold_label_id"]]
            p_en = f"Premise: {row['premise_en']} Hypothesis: {row['hypothesis_en']} Does the premise entail, contradict, or remain neutral to the hypothesis? Answer with one word: Entailment, Neutral, or Contradiction."
            p_hi = f"कथन: {row['premise_hi']} परिकल्पना: {row['hypothesis_hi']} क्या कथन परिकल्पना का समर्थन (Entailment) करता है, विरोध (Contradiction) करता है, या तटस्थ (Neutral) है? केवल एक शब्द में उत्तर दें।"
            xnli_selected.append({
                "seed_id": f"xnli_val_{lbl}_{idx}",
                "source": "XNLI",
                "license": "CC-BY-NC-4.0",
                "type": "utility_nli",
                "category": "utility_reasoning",
                "text_en": p_en,
                "gold_hi_human": p_hi,
                "gold_answer": gold_ans,
                "premise_en": row["premise_en"],
                "premise_hi": row["premise_hi"],
            })
            
    all_seeds = hb_selected + dna_selected + xs_selected + xnli_selected
    
    # 5. Stratified Hash Split Assignment: round(0.3 * n) per cell
    by_cell = {}
    for s in all_seeds:
        cell_key = (s["type"], s["source"], s["category"])
        by_cell.setdefault(cell_key, []).append(s)
        
    seed_splits = {}
    for cell_key, items in by_cell.items():
        items_sorted = sorted(items, key=lambda x: hash_split_rank(x["seed_id"]))
        n_dev = int(round(CONFIG["dev_split_ratio"] * len(items_sorted)))
        for it in items_sorted[:n_dev]:
            seed_splits[it["seed_id"]] = "dev"
        for it in items_sorted[n_dev:]:
            seed_splits[it["seed_id"]] = "test"
            
    return all_seeds, seed_splits, removals, shortfalls

# %%
def run_stage1_addendum4_pipeline():
    print("=" * 85)
    print("STAGE 1 ADDENDUM 4: Manifest v2 Seeded Selection & Alignment Pipeline")
    print("=" * 85)
    
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    print("\n[1] Initializing NLLB-200 and LaBSE on CPU...")
    labse_model = SentenceTransformer(CONFIG["labse_model_id"], device="cpu")
    nllb_tok = AutoTokenizer.from_pretrained(CONFIG["nllb_model_id"])
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["nllb_model_id"], torch_dtype=torch.float32).to("cpu")
    nllb_model.eval()

    print("\n[2] Executing Seeded Selection, DNA Filter, and Writing Manifest v2...")
    all_seeds, seed_splits, removals, shortfalls = load_manifest_v2_seeds(labse_model)
    print(f"  - Total seeds selected: {len(all_seeds)}")
    print(f"  - Total near-duplicate removals: {len(removals)}")
    print(f"  - Documented shortfalls: {shortfalls}")
    
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
        "manifest_version": "v2.0-frozen-deterministic",
        "selection_protocol": "seeded_sampling_sha256_v2_cluster_capped_stratified_hash_split",
        "total_seeds": len(manifest_records),
        "dev_count": sum(1 for v in manifest_records.values() if v["split"] == "dev"),
        "test_count": sum(1 for v in manifest_records.values() if v["split"] == "test"),
        "shortfalls": shortfalls,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest_records, sort_keys=True).encode("utf-8")).hexdigest(),
        "seeds": manifest_records,
    }
    
    with open(CONFIG["manifest_v2_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    with open(CONFIG["manifest_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
        
    print(f"  - Manifest v2 written to: {CONFIG['manifest_v2_output']}")
    print(f"  - Master Manifest v2 SHA-256: {manifest_payload['manifest_sha256']}")
    print(f"  - Dev seeds: {manifest_payload['dev_count']} ({manifest_payload['dev_count']/len(all_seeds)*100:.1f}%), Test seeds: {manifest_payload['test_count']} ({manifest_payload['test_count']/len(all_seeds)*100:.1f}%)")

    def generate_nllb(texts: List[str], tgt_code: str, params: dict) -> List[str]:
        if not texts: return []
        fid = nllb_tok.convert_tokens_to_ids(tgt_code)
        inp = nllb_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = nllb_model.generate(**inp, forced_bos_token_id=fid, **params)
        return [o.strip() for o in nllb_tok.batch_decode(gen, skip_special_tokens=True)]

    print(f"\n[3] Generating Translations, Provenance CMI, and Bhojpuri Analysis across {len(all_seeds)} seeds...")
    dataset_records = []
    bho_qc_diagnostics = []
    qc_failures = []
    bho_spotcheck_pool = []
    
    batch_size = CONFIG["batch_size"]
    total = len(all_seeds)
    
    for i in range(0, total, batch_size):
        batch = all_seeds[i : i + batch_size]
        src_en = [s["text_en"] for s in batch]
        
        hi_beam = generate_nllb(src_en, "hin_Deva", CONFIG["beam_params"])
        bho_beam = generate_nllb(src_en, "bho_Deva", CONFIG["beam_params"])
        
        hi_primary = hi_beam.copy()
        bho_primary = bho_beam.copy()
        
        for idx, s in enumerate(batch):
            if s.get("gold_hi_human"):
                hi_primary[idx] = s["gold_hi_human"]
                
        hing_batch = [build_three_hinglish_bands_provenance(hi_primary[idx], src_en[idx]) for idx in range(len(batch))]
        
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
            
            # Bhojpuri Analysis
            bho_tokens = set(re.findall(r"[\u0900-\u097F]+", bho_txt))
            matched_markers = [m for m in BHOJPURI_DISTINCTIVE_MARKERS if m in bho_tokens]
            for phrase in BHOJPURI_PHRASE_MARKERS:
                if phrase in bho_txt and phrase not in matched_markers:
                    matched_markers.append(phrase)
            has_bho_marker = (len(matched_markers) > 0)
            
            d_raw = edit_distance(hi_txt.strip(), bho_txt.strip())
            max_len = max(len(hi_txt.strip()), len(bho_txt.strip()), 1)
            ned = round(d_raw / max_len, 4)
            
            words = bho_txt.split()
            tri_grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            counts = Counter(tri_grams)
            has_repeated_ngrams = any(cnt >= 2 for cnt in counts.values()) if tri_grams else False
            
            # Hybrid risk flag: exactly 1 marker in 10+ tokens
            is_hybrid_risk = (len(words) >= 10 and len(matched_markers) == 1)
            
            bho_diag = {
                "seed_id": sid,
                "source": src_name,
                "category": cat,
                "type": stype,
                "matched_markers": matched_markers,
                "marker_count": len(matched_markers),
                "has_bho_marker": has_bho_marker,
                "norm_edit_distance": ned,
                "is_identical": (hi_txt.strip() == bho_txt.strip()),
                "is_near_identical": (ned <= 0.05),
                "has_repeated_ngrams": has_repeated_ngrams,
                "is_empty": (len(bho_txt.strip()) == 0),
                "is_hybrid_risk": is_hybrid_risk,
            }
            bho_qc_diagnostics.append(bho_diag)
            
            if has_bho_marker:
                bho_spotcheck_pool.append({
                    "seed_id": sid,
                    "source": src_name,
                    "category": cat,
                    "type": stype,
                    "text_en": en_txt,
                    "text_hi": hi_txt,
                    "text_bho": bho_txt,
                    "matched_markers": ", ".join(matched_markers),
                    "norm_edit_distance": ned,
                })
                
            c_hi = cos_sim(emb_en[b_idx], emb_hi[b_idx])
            c_bho = cos_sim(emb_en[b_idx], emb_bho[b_idx])
            c_tr = cos_sim(emb_en[b_idx], emb_tr[b_idx])
            c_low = cos_sim(emb_en[b_idx], emb_low[b_idx])
            c_mid = cos_sim(emb_en[b_idx], emb_mid[b_idx])
            c_high = cos_sim(emb_en[b_idx], emb_high[b_idx])
            
            # Round-trip CER check on translit
            txt_tr, cmi_tr, ins_tr, tot_tr = hing_batch[b_idx]["translit"]
            reconstructed_deva = roman_to_devanagari_roundtrip(txt_tr)
            cer_val = round(edit_distance(hi_txt.strip(), reconstructed_deva.strip()) / max(1, len(hi_txt.strip())), 4)
            
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
                    "round_trip_cer": cer_val,
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
            bho_qc_passed = (has_bho_marker and not bho_diag["is_near_identical"] and not bho_diag["is_empty"])
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_bho",
                "lang": "bho",
                "text": bho_txt,
                "cmi": 0.0,
                "qc_scores": {
                    "labse_cosine": c_bho,
                    "length_ratio": len_ratio_bho,
                    "norm_edit_distance": ned,
                    "matched_markers": matched_markers,
                    "is_hybrid_risk": is_hybrid_risk,
                    "is_qc_passed": bho_qc_passed,
                },
            })
            if not bho_qc_passed:
                qc_failures.append({"seed_id": sid, "lang": "bho", "reason": "lacks_marker" if not has_bho_marker else "near_identical"})

        print(f"    Processed {min(i + batch_size, total)} / {total} seeds...")

    # Export dataset.jsonl
    with open(CONFIG["dataset_output"], "w", encoding="utf-8") as f:
        for r in dataset_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[4] Exported full dataset to: {CONFIG['dataset_output']} ({len(dataset_records)} rows)")

    # Export 30 Random Bhojpuri Spot-Checks
    random.seed(CONFIG["seed"])
    spot_sample = random.sample(bho_spotcheck_pool, min(30, len(bho_spotcheck_pool)))
    pd.DataFrame(spot_sample).to_csv(CONFIG["bhojpuri_spotcheck_csv"], index=False)
    print(f"[5] Exported 30 Bhojpuri spot-checks to: {CONFIG['bhojpuri_spotcheck_csv']}")

    # Export RFC 4180 Human Validation CSV (hinglish_mixed REMOVED)
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

    # Diagnostics Logs
    with open(CONFIG["bhojpuri_qc_log"], "w", encoding="utf-8") as f:
        for b in bho_qc_diagnostics: f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(CONFIG["qc_failure_log"], "w", encoding="utf-8") as f:
        for q in qc_failures: f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return {
        "manifest_payload": manifest_payload,
        "dataset_records": dataset_records,
        "removals": removals,
        "shortfalls": shortfalls,
    }

# %%
if __name__ == "__main__":
    out = run_stage1_addendum4_pipeline()
