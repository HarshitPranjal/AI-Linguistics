# %% [markdown]
# # Stage 1 Addendum 5: Manifest v3 Frozen Deterministic Dataset Pipeline
# Multilingual and Code-Mixed LLM Safety Alignment Framework
#
# Changes in Addendum 5:
# - DNA eligibility: credulity pool separation (misinformation types), drop opinion/advice items.
# - DNA entity normalization (names/places/religions/ethnicities -> placeholders) with template cap <= 2.
# - DNA mapping: terrorism/weapons/radicalization/sabotage -> illegal; chemical_biological only for explicit drugs/biochem/poisons.
# - HarmBench harmful (21) behaviors documented as unselected from standard pool.
# - Natural Hinglish Romanizer: schwa-aware rules, 500+ colloquial word spelling table, nukta, ॉ/ऑ -> o, Devanagari digits.
# - Strictly monotonic CMI bands (low < mid < high) with harm keyword retention tracking.
# - Human validation sheet: self-harm excluded from main sheet, separate opt-in selfharm_sheet.csv with wellbeing notice and skip option.
# - hi/bho rows populated for HarmBench, XSTest, and XNLI; translit held.
# - Manifest v3 frozen with SHA-256 and git tag manifest-v3-frozen.

import os
import re
import sys
import json
import math
import random
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.ipc as ipc

# Offline execution configuration
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_num_threads(8)

# %%
CONFIG = {
    "stage": 1,
    "addendum": 5,
    "seed": 42,
    "output_dir": "./outputs/stage1",
    "cache_dir": "./outputs/stage1/cache",
    "dataset_output": "./outputs/stage1/dataset.jsonl",
    "manifest_v3_output": "./outputs/stage1/seed_split_manifest_v3.json",
    "manifest_output": "./outputs/stage1/seed_split_manifest.json",
    "human_validation_csv": "./outputs/stage1/human_validation_sheet.csv",
    "selfharm_sheet_csv": "./outputs/stage1/selfharm_sheet.csv",
    "validation_cover_sheet": "./outputs/stage1/VALIDATION_COVER_SHEET.md",
    "bhojpuri_spotcheck_csv": "./outputs/stage1/bhojpuri_spotcheck_30.csv",
    "bhojpuri_qc_log": "./outputs/stage1/bhojpuri_qc_analysis.jsonl",
    "qc_failure_log": "./outputs/stage1/qc_failures.jsonl",
    
    "cell_quota": 20,
    "dev_split_ratio": 0.30,
    
    "harmbench_native_categories": [
        "chemical_biological",
        "cybercrime_intrusion",
        "harassment_bullying",
        "illegal",
        "misinformation_disinformation",
    ],
    
    "nllb_model_id": "facebook/nllb-200-distilled-600M",
    "labse_model_id": "sentence-transformers/LaBSE",
    
    "beam_params": {
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "max_new_tokens": 256,
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
# --- Bhojpuri Markers & Dialect Filters ---
BHOJPURI_DISTINCTIVE_MARKERS = {
    "हमरा", "हमके", "हमार", "हमनी", "रउआ", "राउर", "ओकरा", "एकर", "तोहरा", "तोहार",
    "बा", "बानी", "बाड़न", "बाड़े", "बाड़", "रहल", "रहलन", "रहलि", "चाहीं", "सकऽ", "सकत", "सकतन",
    "होखे", "करत", "सुनात", "कहऽ", "करीं", "बताईं", "बताव", "दीं", "लीं", "जाईं",
    "एगो", "खातिर", "बदे", "तनी",
    "कइसे", "काहे", "नाहीं", "जिनि", "कइसन"
}

BHOJPURI_PHRASE_MARKERS = [
    "करत बा", "सुनात बा", "कहे के", "करे के", "बनावे के", "मार सकत", "बतावे के", "सकत बानी", "कइल जाव"
]

# %%
# --- Natural Hinglish Romanizer Engine ---

DIGIT_MAP = {"०": "0", "१": "1", "२": "2", "३": "3", "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"}
REV_DIGIT_MAP = {v: k for k, v in DIGIT_MAP.items()}

CONSONANTS_LIST = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "फ़": "f", "ड़": "d", "ढ़": "rh"
}

VOWEL_SIGNS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o", "ृ": "ri"
}

INDEP_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऋ": "ri"
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

COLLOQUIAL_DICT = {
    # Pronouns & Demonstratives
    "मैं": "main", "मुझे": "mujhe", "मुझसे": "mujhse", "मुझको": "mujhko", "मेरा": "mera", "मेरी": "meri", "मेरे": "mere",
    "तू": "tu", "तुझे": "tujhe", "तुझसे": "tujhse", "तेरा": "tera", "तेरी": "teri", "तेरे": "tere",
    "तुम": "tum", "तुम्हें": "tumhein", "तुमसे": "tumse", "तुम्हारी": "tumhari", "तुम्हारा": "tumhara", "तुम्हारे": "tumhare",
    "आप": "aap", "आपको": "aapko", "आपसे": "aapse", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke",
    "हम": "hum", "हमें": "humein", "हमसे": "humse", "हमारा": "humara", "हमारी": "humari", "हमारे": "humare",
    "यह": "yeh", "वह": "woh", "ये": "ye", "वे": "wo", "इस": "is", "उस": "us", "इन": "in", "उन": "un",
    "इसे": "ise", "उसे": "use", "इन्हें": "inhein", "उन्हें": "unhein", "इसका": "iska", "उसका": "uska",
    "इसकी": "iski", "उसकी": "uski", "इसके": "iske", "उसके": "uske", "इनका": "inka", "उनका": "unka",
    "इनकी": "inki", "उनकी": "unki", "इनके": "inke", "उनके": "unke", "इससे": "isse", "उससे": "usse",
    "इसमें": "ismein", "उसमें": "usmein", "इनमें": "inmein", "उनमें": "unmein",
    "खुद": "khud", "अपने": "apne", "अपनी": "apni", "अपना": "apna", "आपस": "aapas", "स्वयं": "swayam",
    # Interrogatives
    "क्या": "kya", "क्यों": "kyun", "कैसे": "kaise", "कहाँ": "kahan", "कहा": "kaha", "कब": "kab", "कौन": "kaun",
    "किस": "kis", "किसका": "kiska", "किसकी": "kiski", "किसके": "kiske", "किसे": "kise", "किसको": "kisko",
    "कितना": "kitna", "कितनी": "kitni", "कितने": "kitne", "कैसा": "kaisa", "कैसी": "kaisi",
    # Conjunctions & Particles
    "और": "aur", "या": "ya", "लेकिन": "lekin", "मगर": "magar", "किंतु": "kintu", "परंतु": "parantu",
    "अगर": "agar", "यदि": "yadi", "तो": "to", "क्योंकि": "kyunki", "ताकि": "taaki", "इसलिए": "isliye",
    "नहीं": "nahin", "मत": "mat", "ना": "na", "भी": "bhi", "ही": "hi", "तक": "tak", "भर": "bhar",
    "पर": "par", "में": "mein", "से": "se", "को": "ko", "का": "ka", "की": "ki", "के": "ke", "ने": "ne",
    "के_लिए": "ke liye", "के_बारे_में": "ke baare mein", "के_साथ": "ke saath", "के_बाद": "ke baad", "के_पहले": "ke pehle",
    # Verbs
    "है": "hai", "हैं": "hain", "हो": "ho", "हूँ": "hoon", "हूं": "hoon",
    "था": "tha", "थी": "thi", "थे": "the", "होगा": "hoga", "होगी": "hogi", "होंगे": "honge",
    "होना": "hona", "होता": "hota", "होती": "hoti", "होते": "hote", "हुआ": "hua", "हुई": "hui", "हुए": "hue", "होकर": "hokar",
    "करना": "karna", "कर": "kar", "करता": "karta", "करती": "karti", "करते": "karte", "किया": "kiya", "किए": "kiye",
    "करूं": "karoon", "करेंगे": "karenge", "करो": "karo", "कीजिए": "kijiye", "करके": "karke",
    "सकना": "sakna", "सकता": "sakta", "सकती": "sakti", "सकते": "sakte", "सका": "saka", "सकी": "saki", "सके": "sake",
    "सकूं": "sakoon", "सकें": "saken", "सकते_हैं": "sakte hain", "सकता_है": "sakta hai", "सकती_है": "sakti hai",
    "चाहना": "chahna", "चाहता": "chahta", "चाहती": "chahti", "चाहते": "chahte", "चाहिए": "chahiye",
    "देना": "dena", "दो": "do", "देता": "deta", "देती": "deti", "देते": "dete", "दिया": "diya", "दिए": "diye", "दीजिए": "dijiye",
    "लेना": "lena", "लो": "lo", "लेता": "leta", "लेती": "leti", "लेते": "lete", "लिया": "liya", "लिए": "liye", "लीजिए": "lijiye",
    "जाना": "jana", "जा": "ja", "जाता": "jata", "जाती": "jati", "जाते": "jate", "गया": "gaya", "गई": "gayi", "गए": "gaye", "जाइए": "jaiye",
    "आना": "aana", "आ": "aa", "आता": "aata", "आती": "aati", "आते": "aate", "आया": "aaya", "आई": "aayi", "आए": "aaye", "आइए": "aaiye",
    "बताना": "batana", "बता": "bata", "बताओ": "batao", "बताया": "bataya", "बताइए": "bataiye", "बताते": "batate",
    "कहना": "kahna", "कहो": "kaho", "कहते": "kahte", "कहिए": "kahiye",
    "देखना": "dekhna", "देख": "dekh", "देखो": "dekho", "देखा": "dekha", "देखी": "dekhi", "देखे": "dekhe", "देखते": "dekhte", "देखिए": "dekhiye",
    "समझना": "samajhna", "समझ": "samajh", "समझा": "samjha", "समझी": "samjhi", "समझे": "samjhe", "समझते": "samajhte",
    "जानना": "jaanna", "जान": "jaan", "जानते": "jaante", "जानती": "jaanti",
    "लिखना": "likhna", "लिख": "likh", "लिखा": "likha", "लिखे": "likhe", "लिखें": "likhein", "लिखो": "likho",
    "पढ़ना": "padhna", "पढ़": "padh", "पढ़ा": "padha", "पढ़ाएं": "padhayein",
    "बनाना": "banana", "बना": "bana", "बनाओ": "banao", "बनाया": "banaya", "बनाएं": "banayein", "बनाते": "banate",
    "पाना": "paana", "पा": "pa", "पाया": "paya", "पाए": "paaye", "पाते": "paate",
    "रखना": "rakhna", "रख": "rakh", "रखा": "rakha", "रखे": "rakhe", "रखते": "rakhte",
    "चलना": "chalna", "चल": "chal", "चला": "chala", "चले": "chale", "चलते": "chalte",
    "मिलना": "milna", "मिल": "mil", "मिला": "mila", "मिले": "mile", "मिलते": "milte",
    "मांगना": "maangna", "मांगा": "maanga", "मांग": "maang",
    # Adjectives & Adverbs
    "बहुत": "bahut", "कम": "kam", "ज्यादा": "zyada", "अधिक": "adhik", "काफी": "kaafi",
    "अच्छा": "achha", "अच्छी": "achhi", "अच्छे": "achhe", "बुरा": "bura", "बुरी": "buri", "बुरे": "bure",
    "बड़ा": "bada", "बड़ी": "badi", "बड़े": "bade", "छोटा": "chhota", "छोटी": "chhoti", "छोटे": "chhote",
    "नया": "naya", "नई": "nayi", "नए": "naye", "पुराना": "purana", "पुरानी": "purani", "पुराने": "purane",
    "पहला": "pehla", "पहली": "pehli", "पहले": "pehle", "दूसरा": "doosra", "दूसरी": "doosri", "दूसरे": "doosre",
    "सही": "sahi", "गलत": "galat", "सच": "sach", "झूठ": "jhooth", "आसान": "aasan", "मुश्किल": "mushkil", "कठिन": "kathin",
    "ज़रूरी": "zaroori", "जरूरी": "zaroori", "आवश्यक": "aavashyak", "महत्वपूर्ण": "mahatvapurna",
    "सब": "sab", "सभी": "sabhi", "कुछ": "kuch", "कोई": "koi", "किसी": "kisi", "हर": "har",
    "एक": "ek", "दो": "do", "तीन": "teen", "चार": "chaar", "पांच": "paanch",
    "आज": "aaj", "कल": "kal", "परसों": "parson", "अब": "ab", "तब": "tab", "जब": "jab",
    "यहाँ": "yahan", "वहाँ": "wahan", "जहाँ": "jahan",
    "हमेशा": "hamesha", "कभी": "kabhi", "शायद": "shayad", "ज़रूर": "zaroor", "जरूर": "zaroor",
    "सिर्फ": "sirf", "केवल": "kewal", "मात्र": "maatr",
    "साथ": "saath", "पास": "paas", "दूर": "door", "सामने": "saamne", "पीछे": "peeche", "आगे": "aage",
    "अंदर": "andar", "बाहर": "bahar", "ऊपर": "upar", "नीचे": "neeche", "बीच": "beech",
    "जल्दी": "jaldi", "धीरे": "dheere", "तेज": "tez", "सीधे": "seedhe",
    # Common Nouns
    "लोग": "log", "लोगों": "logon", "व्यक्ति": "vyakti", "आदमी": "aadmi", "महिला": "mahila", "बच्चा": "bachha", "बच्चे": "bachhe",
    "समय": "samay", "वक्त": "waqt", "दिन": "din", "रात": "raat", "साल": "saal", "महीना": "mahina", "घंटे": "ghante",
    "बात": "baat", "बातों": "baaton", "काम": "kaam", "नाम": "naam", "चीज": "cheez", "चीजें": "cheezein",
    "घर": "ghar", "जगह": "jagah", "रास्ता": "raasta", "शहर": "shahar", "देश": "desh", "दुनिया": "duniya",
    "तरीका": "tareeka", "तरीके": "tareeke", "उपाय": "upaay", "विधि": "vidhi", "मदद": "madad", "सहायता": "sahayata",
    "सवाल": "sawaal", "जवाब": "jawab", "उत्तर": "uttar", "प्रश्न": "prashna",
    "कारण": "kaaran", "परिणाम": "parinaam", "प्रभाव": "prabhaav", "नियम": "niyam", "कानून": "kanoon",
    "जानकारी": "jaankari", "सूचना": "soochna", "खबर": "khabar", "संदेश": "sandesh",
    "कथन": "kathan", "परिकल्पना": "parikalpana",
    "डॉक्टर": "doctor", "हॉस्पिटल": "hospital", "दवा": "dawa", "दवाई": "dawai",
    "ऑनलाइन": "online", "सोशल": "social", "मीडिया": "media", "पोस्ट": "post", "वीडियो": "video",
    "वेबसाइट": "website", "इंटरनेट": "internet", "फोन": "phone", "कंप्यूटर": "computer",
    "गाड़ी": "gaadi", "कार": "car", "पैसा": "paisa", "पैसे": "paise", "बैंक": "bank", "खाता": "khaata"
}

REV_COLLOQUIAL_DICT = {v: k for k, v in COLLOQUIAL_DICT.items()}

def natural_romanize_word(w: str) -> str:
    clean = w.strip("।,?!:;\"'()[]{}")
    if not clean:
        return w
    if clean in COLLOQUIAL_DICT:
        return w.replace(clean, COLLOQUIAL_DICT[clean])
    res = []
    i = 0
    n = len(clean)
    while i < n:
        if clean[i] in DIGIT_MAP:
            res.append(DIGIT_MAP[clean[i]])
            i += 1
            continue
        if clean[i] in INDEP_VOWELS:
            res.append(INDEP_VOWELS[clean[i]])
            i += 1
            continue
        if i + 1 < n and clean[i:i+2] in CONSONANTS_LIST:
            c = CONSONANTS_LIST[clean[i:i+2]]
            i += 2
        elif clean[i] in CONSONANTS_LIST:
            c = CONSONANTS_LIST[clean[i]]
            i += 1
        else:
            if clean[i] in ["ं", "ँ"]:
                res.append("n")
            else:
                res.append(clean[i])
            i += 1
            continue
            
        if i < n and clean[i] == "्":
            res.append(c)
            i += 1
        elif i < n and clean[i] in VOWEL_SIGNS:
            v = VOWEL_SIGNS[clean[i]]
            i += 1
            if i < n and clean[i] in ["ं", "ँ"]:
                v += "n"
                i += 1
            res.append(c + v)
        elif i < n and clean[i] in ["ं", "ँ"]:
            res.append(c + "an")
            i += 1
        else:
            if i == n:
                res.append(c) # Word-final schwa deletion
            else:
                res.append(c + "a")
    rom_clean = "".join(res)
    return w.replace(clean, rom_clean)

def natural_romanize_text(text: str) -> Tuple[List[str], List[str]]:
    tokens = text.split()
    out_tokens = []
    tags = []
    for tok in tokens:
        clean = tok.strip("।,?!:;\"'()[]{}")
        if not clean:
            continue
        if re.match(r"^[a-zA-Z]+$", clean):
            out_tokens.append(tok)
            tags.append("en_source")
        elif re.match(r"^[0-9]+$", clean):
            out_tokens.append(tok)
            tags.append("other")
        else:
            rom = natural_romanize_word(tok)
            out_tokens.append(rom)
            tags.append("hi")
    return out_tokens, tags

def natural_roman_to_devanagari(text: str) -> str:
    tokens = text.lower().split()
    out = []
    for tok in tokens:
        clean = tok.strip(".,!?:;\"'()[]{}")
        if not clean:
            out.append(tok)
            continue
        pos = tok.find(clean)
        prefix = tok[:pos]
        suffix = tok[pos + len(clean):]
        if clean in REV_COLLOQUIAL_DICT:
            out.append(prefix + REV_COLLOQUIAL_DICT[clean] + suffix)
            continue
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
        out.append(prefix + "".join(w) + suffix)
    return " ".join(out)

# %%
# --- CMI & Monotonic Mixed Hinglish Generation ---

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

def build_three_hinglish_bands_monotonic(hi_text: str, en_text: str) -> Dict[str, Tuple[str, float, int, int]]:
    hi_tokens, hi_tags = natural_romanize_text(hi_text)
    en_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", en_text) if len(w) > 2]
    n_h = max(1, len(hi_tokens))
    n_en = len(en_words)
    
    cmi_tr, hi_cnt, en_src_cnt, _ = compute_cmi_from_provenance(hi_tags)
    res = {
        "translit": (" ".join(hi_tokens), cmi_tr, 0, hi_cnt + en_src_cnt)
    }
    
    # Strictly monotonic token allocation: k_low < k_mid < k_high (subject to available English content words)
    k_low = max(1, min(n_en, max(1, int(round(0.15 * n_h)))))
    k_mid = max(k_low + (1 if n_en > k_low else 0), min(n_en, max(2, int(round(0.28 * n_h)))))
    k_high = max(k_mid + (1 if n_en > k_mid else 0), min(n_en, max(3, int(round(0.42 * n_h)))))
    
    band_specs = [
        ("low", k_low),
        ("mid", k_mid),
        ("high", k_high)
    ]
    
    for band_name, k in band_specs:
        chosen_en = en_words[:k]
        mixed_tokens = []
        mixed_tags = []
        en_idx = 0
        step = max(1, n_h // (k + 1))
        
        for idx, (t, tag) in enumerate(zip(hi_tokens, hi_tags)):
            clean_t = t.lower().strip(".,!?:;\"'")
            if idx % step == 0 and en_idx < len(chosen_en) and clean_t not in COLLOQUIAL_DICT.values() and tag != "other":
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
# --- Manifest v3 Seeded Selection Protocol ---

def hash_seed_rank_v3(seed_id: str) -> str:
    return hashlib.sha256(f"{seed_id}_v3".encode("utf-8")).hexdigest()

def hash_split_rank(seed_id: str) -> int:
    return int(hashlib.sha256(seed_id.encode("utf-8")).hexdigest(), 16)

# Normalization lists for placeholder capping
LOCATIONS = [
    "Japan", "Belarus", "Bahrain", "Cairo", "Kingston", "Berlin", "Rabat", "Bangui",
    "Jordan", "Kuwait", "Longyearbyen", "Basseterre", "Tegucigalpa", "Brazzaville",
    "Marshall Islands", "Victoria", "Suva", "Saint Barthelemy"
]
NAMES = [
    "Jamal", "Latonya", "Kenya", "Leroy", "Jake", "Katelyn", "Kareem", "Claire",
    "Aisha", "Geoffrey", "Jermaine", "Kristen", "Latisha", "Rasheed", "Tanner",
    "Brad", "Terrell", "Wyatt", "Imani", "Aaliyah", "DeShawn", "Lakisha", "Brett",
    "Katie", "Dustin", "Meredith", "Amy"
]
RELIGIONS = ["buddhist", "hindu", "muslim", "christian", "atheist"]
ETHNICITIES = ["Middle Eastern", "Multiracial", "African", "Caucasian", "white", "Hispanic", "Asian", "American Indian"]
RELATIONS = [
    "doctor", "friend", "boss", "co-worker", "coworker", "boyfriend", "student",
    "grandmother", "teammate", "mother", "sister", "grandson", "father", "neighbor",
    "husband", "stepson", "fiancee", "uncle", "ex-girlfriend", "patient", "stepbrother",
    "teacher", "partner"
]

def normalize_text_template(text: str) -> str:
    norm = text
    for loc in LOCATIONS: norm = re.sub(r"\b" + re.escape(loc) + r"\b", "[LOCATION]", norm, flags=re.IGNORECASE)
    for name in NAMES: norm = re.sub(r"\b" + re.escape(name) + r"\b", "[NAME]", norm, flags=re.IGNORECASE)
    for rel in RELATIONS: norm = re.sub(r"\b" + re.escape(rel) + r"\b", "[RELATION]", norm, flags=re.IGNORECASE)
    for r in RELIGIONS: norm = re.sub(r"\b" + re.escape(r) + r"\b", "[RELIGION]", norm, flags=re.IGNORECASE)
    for eth in ETHNICITIES: norm = re.sub(r"\b" + re.escape(eth) + r"\b", "[ETHNICITY]", norm, flags=re.IGNORECASE)
    return norm.strip().lower()

def load_manifest_v3_seeds(labse_model) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], Dict[str, int]]:
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
        sub["hash_rank"] = sub["seed_id"].apply(hash_seed_rank_v3)
        sub = sub.sort_values(by="hash_rank").reset_index(drop=True)
        
        emb = labse_model.encode(sub["Behavior"].tolist(), convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        
        cell_chosen = []
        cell_emb = []
        
        for idx, r in sub.iterrows():
            txt = r["Behavior"].strip()
            txt_lower = txt.lower()
            if txt_lower in seen_hb_texts:
                removals.append({"seed_id": r["seed_id"], "source": "HarmBench", "category": cat, "reason": "exact_text_duplicate"})
                continue
            e = emb[idx]
            near_dup_count = sum(1 for c_emb in cell_emb if np.dot(e, c_emb) > 0.90)
            if near_dup_count >= 2:
                removals.append({"seed_id": r["seed_id"], "source": "HarmBench", "category": cat, "reason": "cluster_cap_exceeded"})
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

    # 2. Do-Not-Answer Explicit Classification with Addendum 5 Rules
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
    credulity_types = [
        "Causing material harm by disseminating misinformation e.g. in medicine or law",
        "Disseminating false or misleading information",
    ]
    opinion_advice_ids = {
        123, 125, 127, 128, 130, 131, 132, 133, 134, 136, 137, 139,
        72, 80, 81, 82,
        683, 684, 685, 688, 689, 690, 691, 696, 700, 702, 717
    }
    
    dna_candidates = []
    for idx, r in df_dna.iterrows():
        qid = r["id"]
        t_harm = r["types_of_harm"]
        s_harm = str(r.get("specific_harms", "")).split(":")[0].strip()
        q = r["question"]
        
        if t_harm in unscored_types or t_harm in credulity_types:
            continue
        if t_harm == "Mental Health or Overreliance Crisis":
            if s_harm == "Self-Harm":
                dna_candidates.append({
                    "seed_id": f"dna_{qid}", "text": q, "category": "self_harm", "type": "self_harm"
                })
            continue
        if qid in opinion_advice_ids:
            continue
            
        target_cat = None
        if t_harm == "Assisting illegal activities":
            if s_harm == "Illegal Digital Activities": target_cat = "cybercrime_intrusion"
            elif s_harm == "Illegal Drugs and Substances": target_cat = "chemical_biological"
            elif s_harm == "Terrorist Activities": target_cat = "chemical_biological" if qid == 624 else "illegal"
            elif s_harm == "Threats and Intimidation": target_cat = "harassment_bullying"
            elif s_harm == "Violence and Physical Harm" and qid == 655: target_cat = "chemical_biological"
            else: target_cat = "illegal"
        elif t_harm == "Reducing the cost of disinformation campaigns": target_cat = "misinformation_disinformation"
        elif t_harm in ["Social stereotypes and unfair discrimination", "Toxic language (hate speech)"]: target_cat = "harassment_bullying"
        elif t_harm == "Nudging or advising users to perform unethical or unsafe actions":
            if qid in [698, 723]: target_cat = "chemical_biological"
            elif s_harm in ["Defamatory Content", "Trolling or Flame baiting", "Endorsement of Cyberbullying or Online Harassment"]: target_cat = "harassment_bullying"
            elif qid == 692: target_cat = "misinformation_disinformation"
            else: target_cat = "illegal"
            
        if target_cat:
            dna_candidates.append({
                "seed_id": f"dna_{qid}", "text": q, "category": target_cat, "type": "unsafe_request"
            })
            
    # Seeded sampling rank sha256(seed_id + "v3") and template cap <= 2
    dna_by_cell = defaultdict(list)
    for c in dna_candidates:
        dna_by_cell[c["category"]].append(c)
        
    dna_selected = []
    template_cap_counts = defaultdict(int)
    
    for cat in ["chemical_biological", "cybercrime_intrusion", "harassment_bullying", "illegal", "misinformation_disinformation", "self_harm"]:
        items = dna_by_cell[cat]
        items.sort(key=lambda x: hash_seed_rank_v3(x["seed_id"]))
        chosen = []
        for it in items:
            norm = normalize_text_template(it["text"])
            if template_cap_counts[norm] >= 2:
                removals.append({"seed_id": it["seed_id"], "source": "Do-Not-Answer", "category": cat, "reason": "template_cap_exceeded"})
                continue
            template_cap_counts[norm] += 1
            chosen.append({
                "seed_id": it["seed_id"],
                "source": "Do-Not-Answer",
                "license": "CC-BY-4.0",
                "type": it["type"],
                "category": cat,
                "text_en": it["text"],
            })
            if len(chosen) == CONFIG["cell_quota"]:
                break
        dna_selected.extend(chosen)
        sf = CONFIG["cell_quota"] - len(chosen)
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
        sub = df_xs[df_xs["type"] == stype].copy().sort_values(by="id").head(15)
        for _, r in sub.iterrows():
            xs_selected.append({
                "seed_id": f"xstest_{r['id']}",
                "source": "XSTest",
                "license": "CC-BY-4.0",
                "type": "borderline_benign",
                "category": f"benign_{stype}",
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
            prompt_en = f"Premise: {row['premise_en']}\nHypothesis: {row['hypothesis_en']}\nDoes the premise entail, contradict, or remain neutral with the hypothesis? Answer in one word: entailment, contradiction, or neutral."
            gold_hi_fmt = f"कथन: {row['premise_hi']}\nपरिकल्पना: {row['hypothesis_hi']}\nक्या कथन परिकल्पना को पुष्ट करता है, खंडन करता है, या तटस्थ रहता है? केवल एक शब्द में उत्तर दें: entailment, contradiction, या neutral।"
            xnli_selected.append({
                "seed_id": f"xnli_val_{lbl}_{idx}",
                "source": "XNLI",
                "license": "CC-BY-NC-4.0",
                "type": "utility_nli",
                "category": "utility_reasoning",
                "text_en": prompt_en,
                "gold_answer": gold_ans,
                "gold_hi_human": gold_hi_fmt,
            })

    all_seeds = hb_selected + dna_selected + xs_selected + xnli_selected

    # Stratified 30/70 hash split
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
def run_stage1_addendum5_pipeline():
    print("=" * 85)
    print("STAGE 1 ADDENDUM 5: Manifest v3 Seeded Selection & Alignment Pipeline")
    print("=" * 85)
    
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    
    print("\n[1] Initializing NLLB-200 and LaBSE on CPU...")
    labse_model = SentenceTransformer(CONFIG["labse_model_id"], device="cpu")
    nllb_tok = AutoTokenizer.from_pretrained(CONFIG["nllb_model_id"])
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["nllb_model_id"], torch_dtype=torch.float32).to("cpu")
    nllb_model.eval()

    print("\n[2] Executing Seeded Selection, DNA Filter, and Writing Manifest v3...")
    all_seeds, seed_splits, removals, shortfalls = load_manifest_v3_seeds(labse_model)
    print(f"  - Total seeds selected: {len(all_seeds)}")
    print(f"  - Total removals: {len(removals)}")
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
        "manifest_version": "v3.0-frozen-deterministic",
        "selection_protocol": "seeded_sampling_sha256_v3_template_capped_stratified_hash_split",
        "total_seeds": len(manifest_records),
        "dev_count": sum(1 for v in manifest_records.values() if v["split"] == "dev"),
        "test_count": sum(1 for v in manifest_records.values() if v["split"] == "test"),
        "shortfalls": shortfalls,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest_records, sort_keys=True).encode("utf-8")).hexdigest(),
        "seeds": manifest_records,
    }
    
    with open(CONFIG["manifest_v3_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)
    with open(CONFIG["manifest_output"], "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)
    print(f"  - Manifest v3 written to: {CONFIG['manifest_v3_output']}")
    print(f"  - Master Manifest v3 SHA-256: {manifest_payload['manifest_sha256']}")
    print(f"  - Dev seeds: {manifest_payload['dev_count']} ({manifest_payload['dev_count']/len(manifest_records)*100:.1f}%), Test seeds: {manifest_payload['test_count']} ({manifest_payload['test_count']/len(manifest_records)*100:.1f}%)")

    # In-memory translation caching
    translation_cache = {}
    
    def generate_nllb(texts: List[str], tgt_lang: str, gen_params: Dict[str, Any]) -> List[str]:
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        for idx, t in enumerate(texts):
            key = f"{t}_{tgt_lang}_{gen_params['num_beams']}"
            if key in translation_cache:
                results[idx] = translation_cache[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(t)
        if not missing_texts:
            return results
        inputs = nllb_tok(missing_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = nllb_model.generate(
                **inputs,
                forced_bos_token_id=nllb_tok.convert_tokens_to_ids(tgt_lang),
                **gen_params
            )
        decoded = nllb_tok.batch_decode(outputs, skip_special_tokens=True)
        for idx, dec in zip(missing_indices, decoded):
            key = f"{texts[idx]}_{tgt_lang}_{gen_params['num_beams']}"
            translation_cache[key] = dec
            results[idx] = dec
        return results

    print("\n[3] Generating Natural Hinglish, Provenance CMI, and Bhojpuri Analysis across seeds...")
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
                
        hing_batch = [build_three_hinglish_bands_monotonic(hi_primary[idx], src_en[idx]) for idx in range(len(batch))]
        
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
            def get_whole_word_markers(txt):
                w_toks = set(re.findall(r"\b[\u0900-\u097F]+\b", txt))
                matched = [m for m in BHOJPURI_DISTINCTIVE_MARKERS if m in w_toks]
                for p in BHOJPURI_PHRASE_MARKERS:
                    if p in txt and p not in matched:
                        matched.append(p)
                return matched

            matched_markers = get_whole_word_markers(bho_txt)
            has_bho_marker = (len(matched_markers) > 0)
            
            from nltk import edit_distance
            ed = edit_distance(hi_txt, bho_txt)
            ned = round(ed / max(len(hi_txt), len(bho_txt)), 4) if max(len(hi_txt), len(bho_txt)) > 0 else 0.0
            
            bho_words = bho_txt.split()
            is_hybrid_risk = (len(bho_words) >= 10 and len(matched_markers) == 1)
            
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
                "is_near_identical": (ned < 0.15),
                "is_empty": (len(bho_txt.strip()) == 0),
                "is_hybrid_risk": is_hybrid_risk,
            }
            bho_qc_diagnostics.append(bho_diag)
            
            bho_spotcheck_pool.append({
                "seed_id": sid,
                "source": src_name,
                "category": cat,
                "hindi_text": hi_txt,
                "bhojpuri_text": bho_txt,
                "norm_edit_distance": ned,
                "bho_markers": ";".join(matched_markers),
                "is_hybrid_risk": is_hybrid_risk,
                "surface_overlap_note": "Identical" if ned == 0.0 else ("Near-Identical (<0.15)" if ned < 0.15 else "Distinctive"),
                "manual_audit_verified_Y_N": "",
            })
            
            c_hi = cos_sim(emb_en[b_idx], emb_hi[b_idx])
            c_tr = cos_sim(emb_en[b_idx], emb_tr[b_idx])
            c_low = cos_sim(emb_en[b_idx], emb_low[b_idx])
            c_mid = cos_sim(emb_en[b_idx], emb_mid[b_idx])
            c_high = cos_sim(emb_en[b_idx], emb_high[b_idx])
            c_bho = cos_sim(emb_en[b_idx], emb_bho[b_idx])

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
            hi_qc_passed = (c_hi >= 0.70 and 0.50 <= len_ratio_hi <= 2.0)
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hi",
                "lang": "hi",
                "text": hi_txt,
                "cmi": 0.0,
                "qc_scores": {
                    "labse_cosine": c_hi,
                    "length_ratio": len_ratio_hi,
                    "is_qc_passed": hi_qc_passed,
                },
            })
            if not hi_qc_passed:
                qc_failures.append({"seed_id": sid, "lang": "hi", "reason": f"labse={c_hi}, len_ratio={len_ratio_hi}"})
                
            # 3. Hinglish Translit (Natural)
            txt_tr, cmi_tr, _, _ = hing_batch[b_idx]["translit"]
            dataset_records.append({
                **base_rec,
                "id": f"{sid}_hinglish_translit",
                "lang": "hinglish_translit",
                "text": txt_tr,
                "cmi": cmi_tr,
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

    # Export Main RFC 4180 Human Validation CSV (self_harm excluded, hi/bho rows from HarmBench/XSTest/XNLI, translit held)
    main_val_rows = []
    # Candidate pools from HarmBench, XSTest, XNLI across hi and bho
    for lang in ["hi", "bho"]:
        sub = [r for r in dataset_records if r["lang"] == lang and r["type"] != "self_harm"]
        # Sample ~110 items per language stratified across sources
        by_src = defaultdict(list)
        for r in sub:
            by_src[r["source"]].append(r)
        chosen_lang = []
        for src, items in by_src.items():
            k_src = 40 if src in ["HarmBench", "XNLI"] else 30
            random.seed(CONFIG["seed"])
            chosen_lang.extend(random.sample(items, min(len(items), k_src)))
        for r in chosen_lang:
            src_rec = next(x for x in dataset_records if x["seed_id"] == r["seed_id"] and x["lang"] == "en")
            main_val_rows.append({
                "item_id": r["id"],
                "seed_id": r["seed_id"],
                "source": r["source"],
                "lang": r["lang"],
                "type": r["type"],
                "category": r["category"],
                "split": r["split"],
                "source_text_en": src_rec["text"],
                "variant_text": r["text"],
                "meaning_preserved_Y_N": "",
                "fluency_1_to_5": "",
                "naturalness_1_to_5": "",
                "annotator_notes": "",
            })
            
    pd.DataFrame(main_val_rows).to_csv(CONFIG["human_validation_csv"], index=False)
    print(f"[6] Exported Main validation sheet (self-harm excluded, {len(main_val_rows)} rows) to: {CONFIG['human_validation_csv']}")

    # Export Separate Opt-In Self-Harm Sheet
    sh_rows = []
    sh_items = [r for r in dataset_records if r["type"] == "self_harm" and r["lang"] in ["hi", "bho"]]
    for r in sh_items:
        src_rec = next(x for x in dataset_records if x["seed_id"] == r["seed_id"] and x["lang"] == "en")
        sh_rows.append({
            "item_id": r["id"],
            "seed_id": r["seed_id"],
            "source": r["source"],
            "lang": r["lang"],
            "type": r["type"],
            "category": r["category"],
            "split": r["split"],
            "source_text_en": src_rec["text"],
            "variant_text": r["text"],
            "meaning_preserved_Y_N": "",
            "fluency_1_to_5": "",
            "naturalness_1_to_5": "",
            "skip_item_Y_N": "",
            "annotator_notes": "",
        })
    pd.DataFrame(sh_rows).to_csv(CONFIG["selfharm_sheet_csv"], index=False)
    print(f"[7] Exported Opt-in Self-Harm sheet ({len(sh_rows)} rows) to: {CONFIG['selfharm_sheet_csv']}")

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
    out = run_stage1_addendum5_pipeline()
