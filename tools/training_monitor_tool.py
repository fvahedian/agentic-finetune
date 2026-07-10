# tools/training_monitor_tool.py
"""
Training Monitor Tool — Phase 3 of the Agentic Fine-tuning Pipeline

Checks HuggingFace Hub for fine-tuned model status.
Reports whether training is needed or complete.
The orchestrator uses this to decide:
  - if no model exists → training needed
  - if model exists → load metrics and evaluate
"""

import os
import json
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = "fvahedian"
FINETUNED_REPO = f"{HF_USERNAME}/qwen3-rag-faithful-lora"


def check_model_exists(repo_id: str = FINETUNED_REPO) -> bool:
    """
    Check if fine-tuned model exists on HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo id to check

    Returns:
        True if model exists, False otherwise
    """
    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        print(f"  ⚠️  Error checking repo: {e}")
        return False


def get_training_metrics(repo_id: str = FINETUNED_REPO) -> dict:
    """
    Load training metrics from HuggingFace Hub if available.

    Args:
        repo_id: HuggingFace repo id

    Returns:
        dict with training metrics or empty dict
    """
    try:
        metrics_path = hf_hub_download(
            repo_id=repo_id,
            filename="training_metrics.json",
            token=os.getenv("HF_TOKEN")
        )
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_model_card(repo_id: str = FINETUNED_REPO) -> str:
    """
    Load model card from HuggingFace Hub if available.

    Args:
        repo_id: HuggingFace repo id

    Returns:
        model card text or empty string
    """
    try:
        card_path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            token=os.getenv("HF_TOKEN")
        )
        with open(card_path) as f:
            return f.read()
    except Exception:
        return ""


def check_datasets_ready() -> dict:
    """
    Check if training datasets are available on HuggingFace Hub.

    Returns:
        dict showing which datasets are ready
    """
    api = HfApi(token=os.getenv("HF_TOKEN"))

    sft_repo = f"{HF_USERNAME}/qwen3-rag-faithful-sft"
    dpo_repo = f"{HF_USERNAME}/qwen3-rag-faithful-dpo"

    sft_ready = False
    dpo_ready = False

    try:
        api.repo_info(repo_id=sft_repo, repo_type="dataset")
        sft_ready = True
    except RepositoryNotFoundError:
        pass

    try:
        api.repo_info(repo_id=dpo_repo, repo_type="dataset")
        dpo_ready = True
    except RepositoryNotFoundError:
        pass

    return {
        "sft_dataset_ready": sft_ready,
        "dpo_dataset_ready": dpo_ready,
        "sft_repo": sft_repo,
        "dpo_repo": dpo_repo
    }


def run_training_monitor() -> dict:
    """
    Full training status check.

    Returns:
        status report dict with recommendation
    """
    print(f"{'='*60}")
    print(f"  TRAINING MONITOR")
    print(f"  Checking HuggingFace Hub...")
    print(f"{'='*60}\n")

    # Check datasets
    print("Checking datasets...")
    dataset_status = check_datasets_ready()
    print(f"  SFT dataset : {'✅ ready' if dataset_status['sft_dataset_ready'] else '❌ not found'}")
    print(f"  DPO dataset : {'✅ ready' if dataset_status['dpo_dataset_ready'] else '❌ not found'}")

    # Check model
    print(f"\nChecking fine-tuned model ({FINETUNED_REPO})...")
    model_exists = check_model_exists()

    if not model_exists:
        print(f"  ❌ No fine-tuned model found")
        print(f"  → Training needed")

        status = {
            "model_exists": False,
            "training_needed": True,
            "dataset_status": dataset_status,
            "training_metrics": {},
            "recommendation": "TRAIN — no fine-tuned model found on HuggingFace Hub",
            "colab_notebook": "training/colab_training.ipynb",
            "sft_dataset": dataset_status["sft_repo"],
            "dpo_dataset": dataset_status["dpo_repo"]
        }

    else:
        print(f"  ✅ Fine-tuned model found: {FINETUNED_REPO}")

        # Load metrics if available
        metrics = get_training_metrics()
        if metrics:
            print(f"\n  Training metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
        else:
            print(f"  ⚠️  No training metrics found in repo")

        status = {
            "model_exists": True,
            "training_needed": False,
            "dataset_status": dataset_status,
            "training_metrics": metrics,
            "recommendation": "EVALUATE — fine-tuned model found, proceed to evaluation",
            "model_repo": FINETUNED_REPO
        }

    # Save report
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/training_monitor_report.json", "w") as f:
        json.dump(status, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  STATUS: {status['recommendation']}")
    print(f"{'='*60}")
    print(f"\n  Report saved to eval/results/training_monitor_report.json")

    return status


if __name__ == "__main__":
    status = run_training_monitor()