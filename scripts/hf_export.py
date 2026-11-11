#!/usr/bin/env python3
"""
Hugging Face Hub Export & Collection Builder.
Automates uploading the AI Linguistic Vitality benchmark dataset and creating
the public Hugging Face Dataset & Model Collections for official submission.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, create_repo


EXPECTED_DATASET_SHA = "e599106a208b75bb61f73c49874d61eb6ed8c4dd2931f840b347242ccc7bfcbf"
EXPECTED_MANIFEST_SHA = "2d781b131e94694f576b024533fbfa4cdb88980e50dabcf9496dceeb55e6ee0c"


def verify_file_sha(filepath: str, expected_sha: str) -> bool:
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return False
    h = hashlib.sha256(open(filepath, "rb").read()).hexdigest()
    if h != expected_sha:
        print(f"[Mismatch] {filepath}: got {h}, expected {expected_sha}")
        return False
    print(f"[Verified] {filepath} -> {h[:16]}... (PASS)")
    return True


def upload_dataset(api: HfApi, repo_id: str):
    """Uploads the frozen dataset, manifests, and dataset card to Hugging Face."""
    print(f"\n[Dataset Upload] Target Repository: {repo_id}")
    
    # 1. Verify cryptographic integrity before uploading
    assert verify_file_sha("outputs/stage1/dataset.jsonl", EXPECTED_DATASET_SHA), "Dataset SHA mismatch!"
    assert verify_file_sha("outputs/stage1/seed_split_manifest_v3.json", EXPECTED_MANIFEST_SHA), "Manifest SHA mismatch!"

    # 2. Create the dataset repository (public)
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    # 3. Upload dataset card as README.md
    print("[Upload] Uploading README.md (Dataset Card)...")
    api.upload_file(
        path_or_fileobj="outputs/reports/hf_dataset_card.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )

    # 4. Upload frozen dataset
    print("[Upload] Uploading dataset.jsonl...")
    api.upload_file(
        path_or_fileobj="outputs/stage1/dataset.jsonl",
        path_in_repo="dataset.jsonl",
        repo_id=repo_id,
        repo_type="dataset"
    )

    # 5. Upload manifests & configs
    for path, target in [
        ("outputs/stage1/seed_split_manifest_v3.json", "seed_split_manifest_v3.json"),
        ("outputs/stage1/seed_review_v1.csv", "seed_review_v1.csv"),
        ("configs/s1_patterns.json", "configs/s1_patterns.json"),
        ("configs/crisis_resources.json", "configs/crisis_resources.json"),
    ]:
        if os.path.exists(path):
            print(f"[Upload] Uploading {target}...")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=target,
                repo_id=repo_id,
                repo_type="dataset"
            )

    print(f"[Success] Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def create_model_collection(api: HfApi, namespace: str):
    """Creates the public model collection on Hugging Face."""
    print(f"\n[Model Collection] Creating collection under namespace: {namespace}")
    title = "AI Linguistic Vitality: Evaluated Models & Judges"
    desc = (
        "Public model collection containing target models, safety judges, and translation encoders "
        "evaluated in the AI Linguistic Vitality benchmark."
    )
    
    try:
        col = api.create_collection(
            title=title,
            description=desc,
            namespace=namespace
        )
        print(f"[Created] Collection: {col.title} ({col.url})")

        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it",
            "meta-llama/Llama-Guard-3-8B",
            "facebook/nllb-200-3.3B",
            "sentence-transformers/LaBSE"
        ]

        for m in models:
            try:
                api.add_collection_item(
                    collection_slug=col.slug,
                    item_id=m,
                    item_type="model"
                )
                print(f"  + Added model: {m}")
            except Exception as e:
                print(f"  ! Warning adding {m}: {e}")

        print(f"[Success] Model Collection URL: {col.url}")
    except Exception as e:
        print(f"[Notice] If collection already exists, view at: https://huggingface.co/{namespace}")
        print(f"Details: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Export & Submission Helper")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (e.g. username/ai-linguistic-vitality)")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face API Write Token")
    parser.add_argument("--create-collection", action="store_true", help="Also create public model collection")
    args = parser.parse_args()

    if not args.token:
        print("[Error] No Hugging Face token provided. Pass --token or set HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi(token=args.token)
    upload_dataset(api, args.repo_id)

    if args.create_collection:
        namespace = args.repo_id.split("/")[0]
        create_model_collection(api, namespace)

    print("\n================================================================================")
    print("SUBMISSION CHECKLIST (Deadline: Nov 15, 11:59 PM AoE):")
    print(f"1. Hugging Face Dataset: https://huggingface.co/datasets/{args.repo_id} (Ensure Public)")
    print(f"2. Hugging Face Models: Public Collection created in namespace '{args.repo_id.split('/')[0]}'")
    print(f"3. Hugging Face Article: Publish text from outputs/reports/huggingface_article.md")
    print(f"4. GitHub Repository: Public repo with README.md")
    print("================================================================================")


if __name__ == "__main__":
    main()

