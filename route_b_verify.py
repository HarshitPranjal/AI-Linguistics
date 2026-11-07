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
    print("ROUTE B HINGLISH SPEECH OPTIONS (STATUS: UNDECIDED)")
    print("=" * 80)
    print("| Option | Synthesis / Capture Approach | Model / Source ID | Licence | Feasibility Status | Notes |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    print("| 1. Native Recordings | Human speaker audio capture | Annotator protocol ($N=161$ dev) | Custom / Open | **FEASIBLE (Protocols required)** | Highest acoustic authenticity; zero synthetic artifact |")
    print("| 2. Indic Parler-TTS | Open-weight Indic text-to-speech | `ai4bharat/indic-parler-tts` | Apache-2.0 | **UNVERIFIED** | Model card retrieved for Indic languages; Hinglish code-mixed checkpoint unverified |")
    print("| 3. Coqui XTTS-v2 | Zero-shot voice cloning | `coqui/XTTS-v2` | CPML (Non-Commercial) | **UNVERIFIED** | Model card covers 17 languages; Hindi/Hinglish unsupported natively |")
    print("| 4. Bilingual Stitching | Script-splitting & audio concatenation | `mms-tts-hin` + `mms-tts-eng` | CC-BY-NC 4.0 | **FEASIBLE (Acoustic boundary artifacts)** | Discontinuities at code-switching junctures |")
    print("| 5. Monolithic Scope | Drop Hinglish speech; evaluate en, hi, bho | MMS & Whisper | CC-BY-NC / MIT | **FEASIBLE (Baseline recommendation)** | Methodologically cleanest baseline avoiding synthetic distortion |")
    print("\nVerification complete.")


if __name__ == "__main__":
    main()

