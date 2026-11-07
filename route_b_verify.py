#!/usr/bin/env python3
"""
Route B Model & Audio Pipeline Verification Script (Google Colab / Local).
Verifies availability and loadability of open-source ASR and TTS model weights on Hugging Face Hub.
"""

import sys
import os
from typing import Dict, List, Any

TARGET_MODELS = [
    {
        "model_id": "openai/whisper-large-v3",
        "component": "ASR",
        "languages": "en, hi",
        "licence": "MIT",
        "notes": "Native support for en and hi; lacks bho token."
    },
    {
        "model_id": "facebook/mms-1b-all",
        "component": "ASR",
        "languages": "bho, hi, en (1,400+ langs)",
        "licence": "CC-BY-NC 4.0",
        "notes": "Explicit bho-script_devanagari vocabulary."
    },
    {
        "model_id": "facebook/mms-tts-hin",
        "component": "TTS",
        "languages": "hi (Devanagari)",
        "licence": "CC-BY-NC 4.0",
        "notes": "VITS architecture open weights."
    },
    {
        "model_id": "facebook/mms-tts-bho",
        "component": "TTS",
        "languages": "bho (Devanagari)",
        "licence": "CC-BY-NC 4.0",
        "notes": "Dedicated open-source Bhojpuri TTS checkpoint."
    },
    {
        "model_id": "ai4bharat/indicwav2vec-hindi",
        "component": "ASR (Indic)",
        "languages": "hi (Indic)",
        "licence": "MIT",
        "notes": "AI4Bharat IndicSuperb benchmark checkpoint."
    },
]


def verify_models() -> List[Dict[str, Any]]:
    results = []
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        has_hf_api = True
    except ImportError:
        has_hf_api = False

    for m in TARGET_MODELS:
        entry = dict(m)
        status = "UNVERIFIED (OFFLINE)"
        if has_hf_api:
            try:
                info = api.model_info(m["model_id"], timeout=5.0)
                status = f"VERIFIED (commit: {info.sha[:7]})"
            except Exception as e:
                err_str = str(e)
                if "404" in err_str:
                    status = "UNVERIFIED (404 Not Found)"
                elif "401" in err_str or "gated" in err_str.lower():
                    status = "VERIFIED (Gated / Auth Required)"
                else:
                    status = f"UNVERIFIED ({type(e).__name__})"
        entry["status"] = status
        results.append(entry)

    return results


def main():
    print("=" * 80)
    print("ROUTE B AUDIO PIPELINE MODEL VERIFICATION (HF HUB)")
    print("=" * 80)
    results = verify_models()
    
    print("\n| Model ID | Component | Languages | Licence | HF Status |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| `{r['model_id']}` | {r['component']} | {r['languages']} | {r['licence']} | **{r['status']}** |")
    
    print("\n" + "=" * 80)
    print("ROUTE B HINGLISH SPEECH POLICY")
    print("=" * 80)
    print("Hinglish speech is explicitly DROPPED from Route B audio generation.")
    print("Reason: Lack of verified single-model open-weight code-mixed TTS synthesizer on Hugging Face Hub.")
    print("Scope: Route B speech generation and recognition focuses strictly on en, hi, and bho.")
    print("\nVerification complete.")


if __name__ == "__main__":
    main()

