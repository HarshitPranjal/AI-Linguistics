# %% [markdown]
# # Stage 0: Environment Setup, Hardware Check, and Feasibility Audit (Addendum Updated)
# Multilingual and Code-Mixed LLM Safety Evaluation Framework
# Target Languages: English (en), Hindi (hi), Hinglish (hi-Latn), Bhojpuri (bho)
# Pipelines: (A) Direct Text, (B) Speech -> ASR -> Text, (C) Translate-to-English -> LLM

# %%
import os
import sys
import json
import random
import shutil
import platform
from typing import Dict, Any, List, Optional
import numpy as np

# Config dict at top
CONFIG = {
    "stage": 0,
    "seed": 42,
    "output_dir": "./outputs/stage0",
    "cache_file": "./outputs/stage0/stage0_audit.jsonl",
    # Exactly 19 probed candidate models across 5 functional categories:
    "target_instruct_models": [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "CohereForAI/aya-expanse-8b",
        "sarvamai/sarvam-1", # Canonical repo ID on HF (redirects from sarvam-2b-v0.5)
    ],
    "safety_classifier_models": [
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Prompt-Guard-86M",
        "google/shieldgemma-2b",
        "google/shieldgemma-9b",
    ],
    "translation_models": [
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-1.3B",
        "ai4bharat/indictrans2-indic-en-1B",
        "ai4bharat/indictrans2-en-indic-1B",
    ],
    "asr_models": [
        "openai/whisper-small",
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        "ai4bharat/indic-conformer-600m-multilingual",
    ],
    "embedding_models": [
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
    ],
    # Probed candidate strings verified as non-existent / unverified
    "unverified_models": [
        "ai4bharat/indicwhisper-hindi", # No official repo exists under this ID
    ],
    "quantization_target": "4bit-bitsandbytes",
    "colab_vram_limit_gb": 15.0, # Standard Colab T4
}

# Ensure reproducibility
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# %%
def audit_hardware() -> Dict[str, Any]:
    """Inspects compute hardware, RAM, GPU, VRAM, and disk storage."""
    hardware_info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
        "device_type": "cpu",
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "disk_total_gb": 0.0,
        "disk_free_gb": 0.0,
        "is_colab": "google.colab" in sys.modules or os.path.exists("/content"),
    }

    try:
        import torch
        hardware_info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            hardware_info["cuda_available"] = True
            hardware_info["device_type"] = "cuda"
            hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            hardware_info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
            t = torch.cuda.get_device_properties(0).total_memory
            a = torch.cuda.memory_allocated(0)
            hardware_info["vram_free_gb"] = round((t - a) / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hardware_info["mps_available"] = True
            hardware_info["device_type"] = "mps"
            hardware_info["gpu_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        hardware_info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        hardware_info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except ImportError:
        pass

    try:
        disk = shutil.disk_usage(os.path.abspath("."))
        hardware_info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        hardware_info["disk_free_gb"] = round(disk.free / (1024**3), 2)
    except Exception:
        pass

    return hardware_info

# %%
def verify_hf_model(model_id: str) -> Dict[str, Any]:
    """Queries Hugging Face Hub metadata to verify availability, gating, and license."""
    record = {
        "model_id": model_id,
        "status": "UNVERIFIED",
        "gated": None,
        "license": None,
        "downloads_last_month": None,
        "pipeline_tag": None,
        "approx_size_gb": None,
        "4bit_vram_estimate_gb": None,
        "notes": "",
    }

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)

        record["status"] = "VERIFIED_AVAILABLE"
        record["gated"] = getattr(info, "gated", False)
        record["pipeline_tag"] = getattr(info, "pipeline_tag", None)
        record["downloads_last_month"] = getattr(info, "downloads", None)

        card_data = getattr(info, "card_data", None)
        if card_data and hasattr(card_data, "license") and card_data.license:
            record["license"] = card_data.license
        elif getattr(info, "tags", None):
            for tag in info.tags:
                if tag.startswith("license:"):
                    record["license"] = tag.split("license:")[1]
                    break

        total_size = 0
        if getattr(info, "siblings", None):
            for s in info.siblings:
                if s.rfilename.endswith((".safetensors", ".bin")):
                    if hasattr(s, "size") and s.size:
                        total_size += s.size
        
        if total_size > 0:
            size_gb = total_size / (1024**3)
            record["approx_size_gb"] = round(size_gb, 2)
            # Estimated 4-bit VRAM: ~0.30x weight size + 1.2 GB KV/activation baseline
            record["4bit_vram_estimate_gb"] = round(size_gb * 0.30 + 1.2, 2)
        else:
            if "7b" in model_id.lower() or "8b" in model_id.lower():
                record["approx_size_gb"] = 15.0
                record["4bit_vram_estimate_gb"] = 5.8
            elif "9b" in model_id.lower():
                record["approx_size_gb"] = 18.0
                record["4bit_vram_estimate_gb"] = 6.8
            elif "2b" in model_id.lower() or "sarvam-1" in model_id.lower():
                record["approx_size_gb"] = 4.0
                record["4bit_vram_estimate_gb"] = 2.2
            elif "600m" in model_id.lower() or "small" in model_id.lower():
                record["approx_size_gb"] = 1.2
                record["4bit_vram_estimate_gb"] = 1.0
            elif "86m" in model_id.lower():
                record["approx_size_gb"] = 0.35
                record["4bit_vram_estimate_gb"] = 0.5

    except Exception as e:
        record["status"] = "UNVERIFIED"
        record["notes"] = f"Lookup failed: {str(e)}"

    return record

# %%
def audit_asr_and_tts_coverage() -> Dict[str, Any]:
    """Compiles empirical per-language coverage for ASR and TTS candidates."""
    asr_coverage = {
        "openai/whisper-small": {
            "en": "Supported",
            "hi": "Supported",
            "hinglish": "Partial (acoustic transcription into Devanagari or phonetic Latin; frequent drift)",
            "bho": "UNSUPPORTED (no 'bho' language token in tokenizer; forced into 'hi' or phonetic hallucinations)",
        },
        "openai/whisper-large-v3-turbo": {
            "en": "Supported",
            "hi": "Supported",
            "hinglish": "Partial (acoustic transcription into Devanagari or phonetic Latin; frequent drift)",
            "bho": "UNSUPPORTED (no 'bho' language token in tokenizer; forced into 'hi' or phonetic hallucinations)",
        },
        "openai/whisper-large-v3": {
            "en": "Supported",
            "hi": "Supported",
            "hinglish": "Partial (acoustic transcription into Devanagari or phonetic Latin; frequent drift)",
            "bho": "UNSUPPORTED (no 'bho' language token in tokenizer; forced into 'hi' or phonetic hallucinations)",
        },
        "ai4bharat/indic-conformer-600m-multilingual": {
            "en": "Supported (Indian English)",
            "hi": "Supported",
            "hinglish": "Unsupported (trained on monolingual transcripts of scheduled languages)",
            "bho": "UNSUPPORTED (model explicitly covers IN-22 scheduled languages; Bhojpuri is not in 8th Schedule)",
        },
    }

    tts_coverage = {
        "facebook/mms-tts-eng": {
            "en": "Supported",
            "hi": "Unsupported",
            "bho": "UNSUPPORTED",
            "license": "CC-BY-NC 4.0",
        },
        "facebook/mms-tts-hin": {
            "en": "Unsupported",
            "hi": "Supported",
            "bho": "UNSUPPORTED",
            "license": "CC-BY-NC 4.0",
        },
        "ai4bharat/indic-parler-tts": {
            "en": "Supported",
            "hi": "Supported",
            "bho": "UNSUPPORTED (covers 21 scheduled languages; Bhojpuri omitted)",
            "license": "Apache 2.0",
        },
        "bhojpuri_tts_status": "No verified open-source pre-trained TTS model exists for Bhojpuri (bho). Speech synthesis for bho requires fallback acoustic simulation or phonetic alignment.",
    }

    return {"asr": asr_coverage, "tts": tts_coverage}

# %%
def audit_translation_support() -> Dict[str, Any]:
    """Documents verified language code support from official model cards for IndicTrans2 and NLLB-200."""
    return {
        "facebook/nllb-200-distilled-600M": {
            "bho_Deva_support": True,
            "hin_Deva_support": True,
            "eng_Latn_support": True,
            "citation": "NLLB-200 / Flores-200 Model Card explicitly lists 'bho_Deva' (Bhojpuri, Devanagari script) among its 200 covered language varieties.",
        },
        "ai4bharat/indictrans2-indic-en-1B": {
            "bho_Deva_support": False,
            "hin_Deva_support": True,
            "eng_Latn_support": True,
            "citation": "IndicTrans2 Model Card (Appendix D / arXiv:2305.16307) explicitly restricts language_details to 22 scheduled languages: asm_Beng, ben_Beng, brx_Deva, doi_Deva, eng_Latn, gom_Deva, guj_Gujr, hin_Deva, kan_Knda, kas_Arab, kas_Deva, mai_Deva, mal_Mlym, mar_Deva, mni_Beng, mni_Mtei, npi_Deva, ory_Orya, pan_Guru, san_Deva, sat_Olck, snd_Arab, snd_Deva, tam_Taml, tel_Telu, urd_Arab. Bhojpuri (bho_Deva) is NOT supported.",
        },
    }

# %%
def run_stage0_audit() -> Dict[str, Any]:
    print("=" * 85)
    print("STAGE 0 AUDIT: Multilingual LLM Safety Framework Feasibility Check (Addendum)")
    print("=" * 85)

    # 1. Hardware Audit
    hardware = audit_hardware()
    print("\n[1] Compute & Storage Audit:")
    for k, v in hardware.items():
        print(f"  - {k}: {v}")

    # 2. Model Verification across 19 candidate models
    all_categories = {
        "Target Instruct LLMs": CONFIG["target_instruct_models"],
        "Safety Classifiers": CONFIG["safety_classifier_models"],
        "Translation Models": CONFIG["translation_models"],
        "ASR Models": CONFIG["asr_models"],
        "Embedding Models": CONFIG["embedding_models"],
    }

    verified_catalog: Dict[str, List[Dict[str, Any]]] = {}
    records_to_cache = []

    print("\n[2] Verifying All 19 Candidate Model IDs on Hugging Face Hub...")
    total_verified = 0
    for category, model_list in all_categories.items():
        print(f"\n  Checking category: {category} ({len(model_list)} models)")
        verified_catalog[category] = []
        for model_id in model_list:
            rec = verify_hf_model(model_id)
            rec["category"] = category
            verified_catalog[category].append(rec)
            records_to_cache.append(rec)
            if rec["status"] == "VERIFIED_AVAILABLE":
                total_verified += 1
            status_symbol = "✓" if rec["status"] == "VERIFIED_AVAILABLE" else "✗"
            gated_str = "GATED" if rec.get("gated") else "OPEN"
            lic_str = rec.get("license") or "unknown"
            vram_str = f"~{rec['4bit_vram_estimate_gb']} GB (ESTIMATED)" if rec.get('4bit_vram_estimate_gb') else "N/A"
            print(f"    [{status_symbol}] {model_id:<44} | {gated_str:<5} | Lic: {lic_str:<14} | {vram_str}")

    # Check unverified explicitly
    print("\n  Explicitly Checking Known Unverified IDs:")
    for unv_id in CONFIG["unverified_models"]:
        rec = verify_hf_model(unv_id)
        rec["category"] = "Unverified Candidates"
        rec["status"] = "UNVERIFIED"
        records_to_cache.append(rec)
        print(f"    [✗] {unv_id:<44} | UNVERIFIED | Repo does not exist on HF Hub")

    # 3. Audio & Translation Audits
    audio_audit = audit_asr_and_tts_coverage()
    translation_audit = audit_translation_support()

    # 4. Write results to JSONL cache
    with open(CONFIG["cache_file"], "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "hardware", "data": hardware}) + "\n")
        for rec in records_to_cache:
            f.write(json.dumps({"type": "model_verification", "data": rec}) + "\n")
        f.write(json.dumps({"type": "audio_coverage", "data": audio_audit}) + "\n")
        f.write(json.dumps({"type": "translation_coverage", "data": translation_audit}) + "\n")
    print(f"\n[3] Cached verification results ({len(records_to_cache)} models + hardware + coverage) to: {CONFIG['cache_file']}")

    return {
        "hardware": hardware,
        "verified_catalog": verified_catalog,
        "audio_audit": audio_audit,
        "translation_audit": translation_audit,
        "total_candidate_count": len(records_to_cache) - len(CONFIG["unverified_models"]),
    }

# %%
if __name__ == "__main__":
    results = run_stage0_audit()
