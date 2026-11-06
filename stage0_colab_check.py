# -*- coding: utf-8 -*-
"""stage0_colab_check.py

Stage 0 Hardware, Quantization, and VRAM Feasibility Benchmark for Google Colab (Tesla T4).
This standalone script tests hardware environment, 4-bit NF4 quantization with fp16 compute,
and measures peak VRAM during 256-token generation across all shortlisted safety & Indic models.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Open a new Google Colab notebook: Runtime -> Change runtime type -> T4 GPU.
2. In the first cell, install dependencies:
   !pip install -q --upgrade torch torchvision torchaudio
   !pip install -q transformers accelerate bitsandbytes sentencepiece protobuf huggingface_hub
3. Set your Hugging Face token (required for gated models: Llama-3, Gemma-2, Llama-Guard-3):
   import os
   os.environ["HF_TOKEN"] = "your_hf_token_here"
4. Copy and run this script, or upload and execute:
   !python stage0_colab_check.py
"""

# %% [markdown]
# ### Cell 1: Environment and Hardware Verification

# %%
import os
import sys
import gc
import time
import json
import torch

def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

print_banner("1. ENVIRONMENT & HARDWARE VERIFICATION")
print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available : {cuda_available}")

if not cuda_available:
    print("\n[ERROR] CUDA is not available! Please enable GPU in Colab: Runtime -> Change runtime type -> T4 GPU.")
    print("Exiting benchmark.")
    sys.exit(1)

device_id = 0
gpu_name = torch.cuda.get_device_name(device_id)
gpu_props = torch.cuda.get_device_properties(device_id)
total_vram_gb = gpu_props.total_memory / (1024 ** 3)
compute_capability = f"{gpu_props.major}.{gpu_props.minor}"

print(f"Device Name    : {gpu_name}")
print(f"Compute Cap.   : sm_{compute_capability}")
print(f"Total VRAM     : {total_vram_gb:.2f} GB")

# Package versions
packages = ["transformers", "accelerate", "bitsandbytes", "sentencepiece", "huggingface_hub"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"Package {pkg:15s}: {ver}")
    except ImportError:
        print(f"Package {pkg:15s}: [NOT INSTALLED]")

# Hugging Face Authentication Check
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    try:
        from huggingface_hub import HfFolder
        hf_token = HfFolder.get_token()
    except Exception:
        pass

if hf_token:
    print(f"Hugging Face Token: Detected (ends with ...{hf_token[-4:] if len(hf_token) >= 4 else '****'})")
else:
    print("[WARNING] No HF_TOKEN detected! Access to gated models (Llama-3-8B, Gemma-2-9B, Llama-Guard-3) will fail.")
    print("Please set your token: export HF_TOKEN='your_token' or in python: os.environ['HF_TOKEN'] = '...'")

# %% [markdown]
# ### Cell 2: Quantization Configuration Setup

# %%
print_banner("2. 4-BIT NF4 QUANTIZATION CONFIGURATION")

try:
    from transformers import BitsAndBytesConfig
    bnb_config_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    print("BitsAndBytesConfig successfully instantiated:")
    print("  - load_in_4bit          : True")
    print("  - bnb_4bit_compute_dtype: torch.float16")
    print("  - bnb_4bit_quant_type   : 'nf4'")
    print("  - bnb_4bit_use_double_quant: True")
except Exception as e:
    print(f"[ERROR] Failed to import BitsAndBytesConfig: {e}")
    sys.exit(1)

# %% [markdown]
# ### Cell 3: Benchmark Runner Function

# %%
def cleanup_memory():
    """Aggressively collect garbage and clear PyTorch CUDA caching allocator."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

def benchmark_generation_vram(
    model_id: str,
    prompt: str = "Explain the fundamental principles of data privacy in Hindi and English.",
    max_new_tokens: int = 256,
    is_gemma2: bool = False,
    is_guard: bool = False
) -> dict:
    """Load model in 4-bit NF4, generate 256 tokens, record peak allocated & reserved VRAM."""
    cleanup_memory()
    start_vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    start_vram_res = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"\n---> Benchmarking: {model_id}")
    result = {
        "model_id": model_id,
        "load_status": "FAILED",
        "tokens_generated": 0,
        "load_time_sec": 0.0,
        "gen_time_sec": 0.0,
        "tokens_per_sec": 0.0,
        "peak_vram_alloc_gb": 0.0,
        "peak_vram_res_gb": 0.0,
        "fits_t4": False,
        "error": None
    }

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading kwargs
        model_kwargs = {
            "quantization_config": bnb_config_nf4,
            "device_map": "auto",
            "token": hf_token,
            "trust_remote_code": True
        }

        # CRITICAL: Gemma-2 requires eager attention on T4 to prevent OOM/Flash-Attn kernel crashes
        if is_gemma2:
            print("     [NOTE] Applying attn_implementation='eager' for Gemma-2 architecture on T4.")
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        load_time = time.time() - t0
        result["load_time_sec"] = round(load_time, 2)
        print(f"     Model loaded in {load_time:.2f}s")

        # Format input
        if hasattr(tokenizer, "apply_chat_template") and not is_guard:
            messages = [{"role": "user", "content": prompt}]
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_input = prompt

        inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

        # Reset peak stats right before inference
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_time = time.time() - t1
        num_new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved() / (1024 ** 3)

        result["load_status"] = "SUCCESS"
        result["tokens_generated"] = int(num_new_tokens)
        result["gen_time_sec"] = round(gen_time, 2)
        result["tokens_per_sec"] = round(num_new_tokens / max(gen_time, 1e-4), 2)
        result["peak_vram_alloc_gb"] = round(peak_alloc, 3)
        result["peak_vram_res_gb"] = round(peak_res, 3)
        result["fits_t4"] = peak_res <= (total_vram_gb * 0.95)

        print(f"     Generation: {num_new_tokens} tokens in {gen_time:.2f}s ({result['tokens_per_sec']} tok/s)")
        print(f"     Peak VRAM Allocated: {peak_alloc:.2f} GB")
        print(f"     Peak VRAM Reserved : {peak_res:.2f} GB / {total_vram_gb:.2f} GB")
        print(f"     Fit Margin         : {total_vram_gb - peak_res:.2f} GB remaining")

    except Exception as e:
        result["error"] = str(e)
        print(f"     [FAILED] Error: {e}")
    finally:
        # Cleanup
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        cleanup_memory()

    return result

# %% [markdown]
# ### Cell 4: Candidate Generation Models Shortlist Run

# %%
print_banner("3. CANDIDATE GENERATION & GUARD MODELS SHORTLIST BENCHMARK")

candidate_models = [
    {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "is_gemma2": False,
        "is_guard": False,
        "desc": "Meta Llama-3.1 8B Instruct primary alignment baseline"
    },
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "is_gemma2": False,
        "is_guard": False,
        "desc": "Qwen 2.5 7B Instruct multilingual baseline"
    },
    {
        "model_id": "google/gemma-2-9b-it",
        "is_gemma2": True,
        "is_guard": False,
        "desc": "Google Gemma-2 9B IT with eager attention"
    },
    {
        "model_id": "CohereForAI/aya-expanse-8b",
        "is_gemma2": False,
        "is_guard": False,
        "desc": "Cohere For AI Aya Expanse 8B multilingual model"
    },
    {
        "model_id": "meta-llama/Llama-Guard-3-8B",
        "is_gemma2": False,
        "is_guard": True,
        "desc": "Llama Guard 3 8B safety classifier head"
    }
]

benchmark_results = []
for m in candidate_models:
    res = benchmark_generation_vram(
        model_id=m["model_id"],
        prompt="Explain the water cycle in simple terms in Hindi and English.",
        max_new_tokens=256,
        is_gemma2=m["is_gemma2"],
        is_guard=m["is_guard"]
    )
    res["desc"] = m["desc"]
    benchmark_results.append(res)

# %% [markdown]
# ### Cell 5: Specialized Translation Model Load Checks

# %%
print_banner("4. TRANSLATION MODELS LOAD CHECKS (NLLB & IndicTrans2)")

translation_checks = []

# Test NLLB-200-3.3B in fp16
print("\n---> Testing: facebook/nllb-200-3.3B (fp16)")
cleanup_memory()
nllb_res = {"model_id": "facebook/nllb-200-3.3B", "status": "FAILED", "vram_gb": 0.0, "error": None}
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    t0 = time.time()
    nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    nllb_mod = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    load_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
    nllb_res["status"] = "SUCCESS"
    nllb_res["vram_gb"] = round(peak_vram, 2)
    nllb_res["load_time"] = round(load_time, 2)
    print(f"     NLLB-200-3.3B loaded in fp16 in {load_time:.2f}s. Peak Reserved VRAM: {peak_vram:.2f} GB")
except Exception as e:
    nllb_res["error"] = str(e)
    print(f"     [FAILED] NLLB-200-3.3B load: {e}")
finally:
    try:
        del nllb_mod
    except Exception:
        pass
    try:
        del nllb_tok
    except Exception:
        pass
    cleanup_memory()
translation_checks.append(nllb_res)

# Test IndicTrans2-1B load verification
print("\n---> Testing: ai4bharat/indictrans2-indic-en-1B")
cleanup_memory()
it2_res = {"model_id": "ai4bharat/indictrans2-indic-en-1B", "status": "FAILED", "vram_gb": 0.0, "error": None}
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    t0 = time.time()
    it2_tok = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
    it2_mod = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-indic-en-1B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3)
    it2_res["status"] = "SUCCESS"
    it2_res["vram_gb"] = round(peak_vram, 2)
    it2_res["load_time"] = round(load_time, 2)
    print(f"     IndicTrans2-1B loaded in fp16 in {load_time:.2f}s. Peak Reserved VRAM: {peak_vram:.2f} GB")
except Exception as e:
    it2_res["error"] = str(e)
    print(f"     [NOTE] IndicTrans2 load: {e}")
    print("     (IndicTrans2 often requires specific IndicTransTokenizer or fairseq architecture)")
finally:
    try:
        del it2_mod
    except Exception:
        pass
    try:
        del it2_tok
    except Exception:
        pass
    cleanup_memory()
translation_checks.append(it2_res)

# %% [markdown]
# ### Cell 6: AI4Bharat Transliteration Library Install and Test (20 Sentences)

# %%
print_banner("5. AI4BHARAT TRANSLITERATION TEST (20 SENTENCES)")

try:
    import subprocess
    print("Attempting to load or install ai4bharat-transliteration...")
    try:
        from ai4bharat.transliteration import XlitEngine
    except ImportError:
        print("ai4bharat-transliteration not installed. Running pip install...")
        install_res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ai4bharat-transliteration"],
            capture_output=True,
            text=True
        )
        if install_res.returncode == 0:
            print("Successfully installed ai4bharat-transliteration.")
            from ai4bharat.transliteration import XlitEngine
        else:
            raise ImportError(f"Installation failed: {install_res.stderr.strip()}")

    print("Initializing XlitEngine for Hindi (hi)...")
    engine = XlitEngine("hi", beam_width=4, rescore=False)

    test_sentences = [
        "namaste aap kaise hain?",
        "computer aur internet aaj ke samay mein bahut mahatvapurna hain.",
        "cyber suraksha ke niyam kya hain?",
        "kya aap mujhe ek email draft karne mein madad kar sakte hain?",
        "bank khate mein paise transfer karne ke tarike bataiye.",
        "doctor se online appointment kaise book karein?",
        "mobile phone ki battery life kaise badhayein?",
        "bharatiya samvidhan ke maulik adhikar kaun se hain?",
        "data privacy aur suraksha mein kya antar hai?",
        "artificial intelligence ka bhavishya kya hai?",
        "password reset karne ke liye kya karein?",
        "online shopping karte samay kin baaton ka dhyan rakhein?",
        "social media par farzi khabron se kaise bachein?",
        "python programming language kaise seekhein?",
        "machine learning algorithm kaise kaam karte hain?",
        "cloud computing ke kya fayde hain?",
        "digital payment ke surakshit tarike kya hain?",
        "software update karna kyon zaroori hai?",
        "kya aap is paragraph ka anuvad kar sakte hain?",
        "aaj ke mukhya samachar kya hain?"
    ]

    print(f"\nExecuting transliteration across {len(test_sentences)} sentences:")
    for idx, sent in enumerate(test_sentences, 1):
        out = engine.translit_sentence(sent)
        print(f"[{idx:02d}] Source : {sent}")
        print(f"     Translit: {out}")
    print("\n[SUCCESS] AI4Bharat transliteration test successfully completed on 20 sentences.")

except Exception as e:
    print(f"\n[NOTE / WARNING] AI4Bharat transliteration check: {e}")
    print("Detailed dependency notice: ai4bharat-transliteration may require specific onnxruntime/cython or model assets.")

# %% [markdown]
# ### Cell 7: Summary Report & Verification Table

# %%
print_banner("6. T4 VRAM AUDIT SUMMARY REPORT")

header = f"{'Model ID':<35} | {'Status':<7} | {'Alloc GB':<8} | {'Peak Res GB':<11} | {'Remaining':<9} | {'Fit T4'}"
print(header)
print("-" * len(header))

for r in benchmark_results:
    mid = r["model_id"].split("/")[-1][:33]
    st = r["load_status"]
    alloc = f"{r['peak_vram_alloc_gb']:.2f}" if st == "SUCCESS" else "N/A"
    res = f"{r['peak_vram_res_gb']:.2f}" if st == "SUCCESS" else "N/A"
    rem = f"{total_vram_gb - r['peak_vram_res_gb']:.2f} GB" if st == "SUCCESS" else "N/A"
    fits = "YES" if r.get("fits_t4") else ("NO" if st == "SUCCESS" else "ERR")
    print(f"{mid:<35} | {st:<7} | {alloc:<8} | {res:<11} | {rem:<9} | {fits}")

print("\nTranslation Models:")
for r in translation_checks:
    mid = r["model_id"].split("/")[-1][:33]
    st = r["status"]
    res = f"{r['vram_gb']:.2f}" if st == "SUCCESS" else "N/A"
    print(f"{mid:<35} | {st:<7} | {'-':<8} | {res:<11} | {'-':<9} | {st == 'SUCCESS'}")

# Export JSON summary for documentation
summary_data = {
    "gpu_name": gpu_name,
    "total_vram_gb": total_vram_gb,
    "cuda_version": torch.version.cuda,
    "compute_capability": compute_capability,
    "models": benchmark_results,
    "translation_models": translation_checks
}

with open("stage0_colab_results.json", "w") as f:
    json.dump(summary_data, f, indent=2)

print("\nBenchmark results saved to stage0_colab_results.json")
print("Stage 0 Colab check complete.")

