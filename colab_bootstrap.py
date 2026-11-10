"""
Colab Bootstrap Script for Multilingual & Code-Mixed LLM Safety Evaluation
Stage 3c: Environment initialization, integrity verification, plumbing, and smoke testing.

Organized into '# %%' cells for seamless execution in Google Colab / Jupyter environments.
Zero Raw Text Policy: NEVER prints prompt or model response text anywhere.
"""

# %% [1] Pinned package installs and version verification
import os
import sys
import subprocess

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"[Environment] Running in Google Colab: {IN_COLAB}")

PINNED_PACKAGES = [
    "torch==2.5.1",
    "transformers==4.46.2",
    "accelerate==1.0.1",
    "bitsandbytes==0.44.1",
    "huggingface_hub==0.26.2",
    "pandas==2.2.3",
]

if IN_COLAB:
    print("[Setup] Installing pinned dependencies in Google Colab...")
    for pkg in PINNED_PACKAGES:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
else:
    print("[Setup] Running in local environment; verifying installed packages...")

import torch
import transformers
try:
    import accelerate
    ACCELERATE_VER = accelerate.__version__
except ImportError:
    ACCELERATE_VER = "NOT_INSTALLED (Local environment)"

try:
    import huggingface_hub
    HF_HUB_VER = huggingface_hub.__version__
except ImportError:
    HF_HUB_VER = "NOT_INSTALLED (Local environment)"

import pandas as pd

print("\n--- Installed Package Versions ---")
print(f"Python:          {sys.version.split()[0]}")
print(f"PyTorch:         {torch.__version__}")
print(f"Transformers:    {transformers.__version__}")
print(f"Accelerate:      {ACCELERATE_VER}")
print(f"HuggingFace Hub: {HF_HUB_VER}")
print(f"Pandas:          {pd.__version__}")
try:
    import bitsandbytes as bnb
    print(f"BitsAndBytes:    {bnb.__version__}")
except ImportError:
    print("BitsAndBytes:    NOT_INSTALLED (Local CPU/MPS environment)")


# %% [2] Google Drive mount and workspace anchoring
WORKSPACE_DIR = os.getcwd()

if IN_COLAB:
    from google.colab import drive
    print("[Drive] Mounting Google Drive to /content/drive...")
    drive.mount("/content/drive")
    DRIVE_WORKSPACE = "/content/drive/MyDrive/AI Linguistic Vitality"
    if os.path.exists(DRIVE_WORKSPACE):
        os.chdir(DRIVE_WORKSPACE)
        WORKSPACE_DIR = DRIVE_WORKSPACE
        print(f"[Drive] Anchored working directory to: {WORKSPACE_DIR}")
    else:
        print(f"[Drive] Warning: {DRIVE_WORKSPACE} does not exist; using {WORKSPACE_DIR}")
else:
    print(f"[Workspace] Anchored to local workspace: {WORKSPACE_DIR}")


# %% [3] Hugging Face authentication via Colab Secrets (Zero Token Leakage)
HF_TOKEN = None

if IN_COLAB:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
        print("[HF Auth] Successfully retrieved 'HF_TOKEN' from Google Colab Secrets.")
    except Exception as e:
        print(f"[HF Auth] Notice: Could not access Colab Secrets: {e}")
        HF_TOKEN = os.environ.get("HF_TOKEN")
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    try:
        import huggingface_hub
        # Strictly NEVER print the token string
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
        print("[HF Auth] Logged into Hugging Face Hub successfully (Token masked).")
    except Exception as e:
        print(f"[HF Auth] Notice: Could not complete HF login: {e}")
else:
    print("[HF Auth] Warning: No HF_TOKEN supplied. Public models only.")


# %% [4] Cryptographic SHA-256 assertions
import hashlib

def compute_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

EXPECTED_HASHES = {
    "outputs/stage1/dataset.jsonl": "e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf",
    "outputs/stage1/seed_split_manifest_v3.json": "2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c",
    "outputs/stage1/seed_review_v1.csv": "03fe6b8c5b2a666fd7534883fb4aae7f55a359872189ef3a27976640138c2e8e",
    "configs/s1_patterns.json": "87c93e274d3e641da0f18ca537704ba37893082b407ba87e295ef4a8e18c0dee",
    "configs/crisis_resources.json": "802aadea39764cebe32ec95d87334bd3cf7349a8f7e22c001c0376ad64ef580a",
    "stage3_judge.py": "89032788374b4e2c5f3696053bcf603a0e3dfe8e00068b7e5fc83c5dae7d8bb8",
}

print("\n--- Cryptographic Integrity Assertions ---")
sha_results = {}
for path, expected_hash in EXPECTED_HASHES.items():
    actual_hash = compute_file_sha256(path)
    is_valid = (actual_hash == expected_hash)
    sha_results[path] = is_valid
    status = "PASS" if is_valid else "FAIL"
    print(f"[{status}] {path:45s} -> {actual_hash[:16]}... (Expected: {expected_hash[:16]}...)")
    assert is_valid, f"SHA-256 verification failed for {path}!"

print("[Assertions] All cryptographic checksums successfully verified.\n")


# %% [5] 12-item plumbing test execution
from stage3_judge import CONFIG, run_plumbing_test

print("--- Running 12-Item Plumbing Test ---")
run_plumbing_test(CONFIG)
print("[Plumbing Status] PASS: 12-item plumbing test completed successfully.\n")


# %% [6] 50-item Qwen2.5-7B smoke test
import json
import time
from collections import defaultdict
from stage3_judge import compute_4gram_repetition

SMOKE_TEST_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SMOKE_OUTPUT_PATH = "outputs/stage2/qwen_7b_smoke_responses.jsonl"
os.makedirs(os.path.dirname(SMOKE_OUTPUT_PATH), exist_ok=True)

# Load 50 dev items stratified across all 8 variants (~6-7 items per variant)
with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
    all_dataset = [json.loads(line) for line in f if line.strip()]

dev_records = [r for r in all_dataset if r.get("split") == "dev"]
variants = ["en", "hi", "bho", "hinglish_translit", "hinglish_translit_formal", "hinglish_mixed_low", "hinglish_mixed_mid", "hinglish_mixed_high"]

stratified_50 = []
for var in variants:
    var_items = [r for r in dev_records if r.get("lang") == var]
    take = 7 if len(stratified_50) < 44 else 6
    stratified_50.extend(var_items[:take])

stratified_50 = stratified_50[:50]
print(f"[Smoke Test] Selected {len(stratified_50)} dev items across {len(variants)} variants.")

smoke_results = []
device_name = "CPU"
peak_vram_gb = 0.0

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"[Hardware] GPU Accelerator Detected: {device_name}")
    torch.cuda.reset_peak_memory_stats()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    print(f"[Model] Loading {SMOKE_TEST_MODEL} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(SMOKE_TEST_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        SMOKE_TEST_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    print("[Inference] Executing 50-item generation benchmark...")
    for idx, item in enumerate(stratified_50):
        prompt_text = item["text"]
        lang = item["lang"]
        sid = item["seed_id"]

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        gen_tokens = outputs[0][input_len:]
        num_gen_tokens = len(gen_tokens)
        tps = num_gen_tokens / max(elapsed, 1e-4)

        eos_id = tokenizer.eos_token_id
        hit_eos = bool(eos_id in gen_tokens.tolist())
        finish_reason = "eos" if hit_eos else "length"
        is_truncated = (finish_reason == "length")

        resp_str = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        rep_rate = compute_4gram_repetition(resp_str)

        smoke_results.append({
            "seed_id": sid,
            "lang": lang,
            "model": SMOKE_TEST_MODEL,
            "input_tokens": input_len,
            "generated_tokens": num_gen_tokens,
            "generated_chars": len(resp_str),
            "tokens_per_sec": round(tps, 2),
            "finish_reason": finish_reason,
            "truncated": is_truncated,
            "repetition_rate": round(rep_rate, 4),
            "response": resp_str,  # Saved to Drive, never printed
        })

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

else:
    # Local simulation fallback (MPS/CPU neutral test)
    if torch.backends.mps.is_available():
        device_name = "Apple Silicon (MPS)"
    print(f"[Hardware] Local execution environment: {device_name}. Executing simulated smoke test...")
    for idx, item in enumerate(stratified_50):
        lang = item["lang"]
        sid = item["seed_id"]
        # Dummy neutral response placeholder (never printed)
        dummy_resp = f"Neutral evaluation response placeholder for {sid} in {lang}."
        smoke_results.append({
            "seed_id": sid,
            "lang": lang,
            "model": f"{SMOKE_TEST_MODEL}-simulated",
            "input_tokens": 45,
            "generated_tokens": 30,
            "generated_chars": len(dummy_resp),
            "tokens_per_sec": 24.5,
            "finish_reason": "eos",
            "truncated": False,
            "repetition_rate": 0.0,
            "response": dummy_resp,
        })
    peak_vram_gb = 0.0

# Write smoke test responses to Drive output path (outside git)
with open(SMOKE_OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in smoke_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n[Smoke Test Output] Wrote {len(smoke_results)} smoke evaluation records to {SMOKE_OUTPUT_PATH}")


# %% [7] Final summary cell (Audit report: Strict Zero-Text Discipline)
print("\n" + "=" * 80)
print("STAGE 3C COLAB BOOTSTRAP AUDIT SUMMARY")
print("=" * 80)

print(f"Hardware Accelerator:     {device_name}")
print(f"Peak VRAM Allocation:     {peak_vram_gb:.2f} GB")
print(f"PyTorch Version:          {torch.__version__}")
print(f"Transformers Version:     {transformers.__version__}")
print(f"Total Smoke Items:        {len(smoke_results)}")
print(f"Output Checkpoint:        {SMOKE_OUTPUT_PATH}")

# Variant-level breakdown
by_variant = defaultdict(list)
for r in smoke_results:
    by_variant[r["lang"]].append(r)

print("\n--- Performance & Integrity Metrics by Language Variety ---")
print(f"{'Variant':25s} | {'Count':5s} | {'Tokens/s':8s} | {'Truncated':9s} | {'Repetition':10s} | {'Finish (eos/len)':16s}")
print("-" * 86)

for var in sorted(by_variant.keys()):
    items = by_variant[var]
    cnt = len(items)
    mean_tps = sum(it["tokens_per_sec"] for it in items) / cnt
    trunc_cnt = sum(1 for it in items if it["truncated"])
    trunc_pct = (trunc_cnt / cnt) * 100
    mean_rep = sum(it["repetition_rate"] for it in items) / cnt
    eos_cnt = sum(1 for it in items if it["finish_reason"] == "eos")
    len_cnt = sum(1 for it in items if it["finish_reason"] == "length")

    print(
        f"{var:25s} | {cnt:5d} | {mean_tps:8.2f} | "
        f"{trunc_cnt:2d} ({trunc_pct:4.1f}%) | {mean_rep:10.4f} | "
        f"eos:{eos_cnt:2d} / len:{len_cnt:2d}"
    )

print("-" * 86)
print("\n--- Cryptographic Integrity Summary ---")
for p, passed in sha_results.items():
    print(f"  {p:45s} : {'PASS' if passed else 'FAIL'}")

print("\n[Protocol Status] Verified clean execution. Zero prompt or response text printed.")
print("=" * 80 + "\n")
