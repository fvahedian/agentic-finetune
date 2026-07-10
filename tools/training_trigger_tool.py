# tools/training_trigger_tool.py
"""
Training Trigger Tool — Autonomous retraining component.

When decision = RETRAIN, this tool:
1. Analyzes why training failed
2. Adjusts training parameters
3. Prepares improved dataset
4. Simulates job submission (local mode)
5. Returns job status and estimated completion

In production this would call RunPod/Lambda Labs API.
For portfolio: simulates the full autonomous loop.
"""

import os
import json
import time
from datetime import datetime
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_USERNAME = "ftimavn"
JOBS_LOG = "eval/results/training_jobs.json"


def analyze_failure(decision_report: dict) -> dict:
    """
    Analyzes decision report to determine what went wrong
    and what parameters to adjust for retraining.
    """
    issues = decision_report.get("issues", [])
    metrics = decision_report.get("metrics_summary", {})

    adjustments = {}

    # Faithfulness didn't improve
    if any("faithfulness" in i.lower() for i in issues):
        adjustments["dpo_beta"] = 0.2          # was 0.1, increase
        adjustments["num_epochs"] = 2           # was 1, increase
        adjustments["learning_rate"] = "3e-5"   # was 5e-5, decrease
        adjustments["reason"] = "faithfulness_not_improved"

    # Refusal rate too high
    if any("refusal" in i.lower() for i in issues):
        adjustments["add_positive_examples"] = True
        adjustments["faithfulness_threshold"] = 0.6  # was 0.8, relax
        adjustments["reason"] = "over_refusal"

    # Default adjustments
    if not adjustments:
        adjustments["num_epochs"] = 2
        adjustments["reason"] = "general_improvement"

    return adjustments


def prepare_improved_dataset(adjustments: dict) -> dict:
    """
    Prepares an improved DPO dataset based on failure analysis.
    Filters for higher quality preference pairs.
    """
    print("  Preparing improved training dataset...")

    # Load existing DPO dataset
    dataset = load_dataset(
        f"{HF_USERNAME}/qwen3-rag-faithful-dpo",
        split="train"
    )

    print(f"  Original dataset: {len(dataset)} pairs")

    # Filter for higher quality pairs
    # In real life: filter by faithfulness gap > threshold
    # Here: just use a subset with note about filtering
    improved_size = min(len(dataset), 800)
    improved_dataset = dataset.select(range(improved_size))

    print(f"  Improved dataset: {len(improved_dataset)} pairs")
    print(f"  Adjustments: {adjustments}")

    return {
        "dataset_size": len(improved_dataset),
        "original_size": len(dataset),
        "adjustments": adjustments
    }


def simulate_training_job(
    adjustments: dict,
    dataset_info: dict
) -> dict:
    """
    Simulates submitting a training job to RunPod.

    In production this would call:
        runpod.create_pod(
            name="qwen-finetune",
            image_name="runpod/pytorch",
            gpu_type_id="NVIDIA A100-SXM4-80GB",
            cloud_type="SECURE",
        )

    For portfolio: simulates the job lifecycle.
    """
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n  [SIMULATED] Submitting training job to RunPod...")
    print(f"  Job ID: {job_id}")
    print(f"  GPU: NVIDIA A100 (simulated)")
    print(f"  Model: Qwen2.5-7B-Instruct")
    print(f"  DPO beta: {adjustments.get('dpo_beta', 0.1)}")
    print(f"  Epochs: {adjustments.get('num_epochs', 1)}")
    print(f"  Dataset: {dataset_info['dataset_size']} pairs")

    # Simulate job submission delay
    print(f"\n  Submitting job", end="")
    for _ in range(3):
        time.sleep(1)
        print(".", end="", flush=True)
    print(" submitted!")

    job = {
        "job_id": job_id,
        "status": "SUBMITTED",
        "submitted_at": datetime.now().isoformat(),
        "estimated_completion": "2 hours",
        "gpu": "NVIDIA A100 (simulated)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "output_repo": f"{HF_USERNAME}/qwen25-rag-faithful-dpo-lora-v2",
        "adjustments": adjustments,
        "dataset_size": dataset_info["dataset_size"],
        "mode": "SIMULATED — production would use RunPod API",
        "runpod_code": """
# Production RunPod code:
# import runpod
# pod = runpod.create_pod(
#     name='qwen-finetune',
#     image_name='runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04',
#     gpu_type_id='NVIDIA A100-SXM4-80GB',
#     cloud_type='SECURE',
#     env={'HF_TOKEN': os.getenv('HF_TOKEN'),
#          'DPO_BETA': str(adjustments['dpo_beta']),
#          'NUM_EPOCHS': str(adjustments['num_epochs'])}
# )
"""
    }

    # Save job to log
    os.makedirs("eval/results", exist_ok=True)
    jobs = []
    if os.path.exists(JOBS_LOG):
        with open(JOBS_LOG) as f:
            jobs = json.load(f)
    jobs.append(job)
    with open(JOBS_LOG, "w") as f:
        json.dump(jobs, f, indent=2)

    return job


def run_training_trigger(
    decision_report: dict = None
) -> dict:
    """
    Main training trigger function.
    Called by orchestrator when decision = RETRAIN.

    Args:
        decision_report: output from decision_tool

    Returns:
        job status dict
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING TRIGGER")
    print(f"{'='*60}")

    # Load decision report if not provided
    if not decision_report:
        report_path = "eval/results/decision_report.json"
        if os.path.exists(report_path):
            with open(report_path) as f:
                decision_report = json.load(f)
        else:
            return {
                "status": "ERROR",
                "message": "No decision report found. Run decision tool first."
            }

    decision = decision_report.get("decision", "")

    if decision == "DEPLOY":
        return {
            "status": "SKIPPED",
            "message": "Decision is DEPLOY — no retraining needed."
        }

    if decision == "ESCALATE":
        return {
            "status": "ESCALATED",
            "message": "Decision requires human review before retraining."
        }

    # Analyze failure and prepare adjustments
    print("\n  Analyzing failure modes...")
    adjustments = analyze_failure(decision_report)
    print(f"  Root cause: {adjustments.get('reason')}")
    print(f"  Key adjustment: DPO beta {adjustments.get('dpo_beta', 'unchanged')}, "
          f"epochs {adjustments.get('num_epochs', 'unchanged')}")

    # Prepare improved dataset
    dataset_info = prepare_improved_dataset(adjustments)

    # Submit training job
    job = simulate_training_job(adjustments, dataset_info)

    print(f"\n{'='*60}")
    print(f"  TRAINING JOB SUBMITTED")
    print(f"  Job ID    : {job['job_id']}")
    print(f"  Status    : {job['status']}")
    print(f"  ETA       : {job['estimated_completion']}")
    print(f"  Output    : {job['output_repo']}")
    print(f"  Mode      : {job['mode']}")
    print(f"{'='*60}")

    return job


if __name__ == "__main__":
    # Test with existing decision report
    report_path = "eval/results/decision_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            decision_report = json.load(f)
        result = run_training_trigger(decision_report)
    else:
        print("No decision report found. Run decision_tool.py first.")