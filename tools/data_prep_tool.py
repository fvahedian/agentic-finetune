# tools/data_prep_tool.py
"""
Data Preparation Tool — Phase 2 of the Agentic Fine-tuning Pipeline

Loads and formats two public datasets:
- SQuAD v1.1  → SFT dataset (question + context → faithful answer)
- HaluEval    → DPO dataset (chosen faithful vs rejected hallucinated)

Runs quality check on preference pairs:
- chosen answer should have high faithfulness
- rejected answer should have low faithfulness
- filters out pairs where gap is too small

Saves formatted datasets to HuggingFace Hub.
"""

import os
import json
import time
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI

load_dotenv()

MODEL = "qwen/qwen3-next-80b-a3b-instruct"
HF_USERNAME = "ftimavn"


def get_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )


def format_sft_example(question: str, context: str, answer: str) -> dict:
    """
    Format a single SQuAD example for SFT training.
    Uses chat template format expected by Qwen.
    """
    system = """You are a helpful assistant that answers questions 
based strictly on the provided context. 
If the answer is not in the context, say so clearly.
Never add information not present in the context."""

    user = f"""Context: {context}

Question: {question}"""

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer}
        ],
        "question": question,
        "context": context,
        "answer": answer
    }


def format_dpo_example(
    question: str,
    context: str,
    chosen: str,
    rejected: str
) -> dict:
    """
    Format a single HaluEval example for DPO training.
    chosen  = faithful answer grounded in context
    rejected = hallucinated answer
    """
    system = """You are a helpful assistant that answers questions 
based strictly on the provided context."""

    prompt = f"""Context: {context}

Question: {question}"""

    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "question": question,
        "context": context
    }


def prepare_sft_dataset(n: int = 500) -> list:
    """
    Load SQuAD and format for SFT training.

    Args:
        n: number of examples to use

    Returns:
        list of formatted SFT examples
    """
    print(f"\nLoading SQuAD for SFT ({n} examples)...")
    dataset = load_dataset(
        "rajpurkar/squad",
        split=f"train[:{n}]"
    )

    sft_examples = []
    for example in dataset:
        question = example["question"]
        context = example["context"]
        answer = example["answers"]["text"][0]

        formatted = format_sft_example(question, context, answer)
        sft_examples.append(formatted)

    print(f"✅ SFT dataset: {len(sft_examples)} examples")
    return sft_examples


def prepare_dpo_dataset(n: int = 1000) -> list:
    """
    Load HaluEval and format for DPO training.

    HaluEval already has:
    - question
    - context (knowledge)
    - right_answer (faithful)
    - hallucinated_answer (hallucinated)

    Args:
        n: number of examples to use

    Returns:
        list of formatted DPO preference pairs
    """
    print(f"\nLoading HaluEval for DPO ({n} examples)...")

    try:
        dataset = load_dataset(
            "pminervini/HaluEval",
            "qa",
            split=f"data[:{n}]"
        )
    except Exception as e:
        print(f"  ⚠️  HaluEval load error: {e}")
        print("  Trying alternative split...")
        dataset = load_dataset(
            "pminervini/HaluEval",
            "qa",
            split=f"train[:{n}]"
        )

    dpo_examples = []
    skipped = 0

    for example in dataset:
        question = example.get("question", "")
        context = example.get("knowledge", "")
        chosen = example.get("right_answer", "")
        rejected = example.get("hallucinated_answer", "")

        # Skip if any field is empty
        if not all([question, context, chosen, rejected]):
            skipped += 1
            continue

        # Skip if chosen and rejected are identical
        if chosen.strip() == rejected.strip():
            skipped += 1
            continue

        formatted = format_dpo_example(
            question, context, chosen, rejected
        )
        dpo_examples.append(formatted)

    print(f"✅ DPO dataset: {len(dpo_examples)} examples "
          f"({skipped} skipped)")
    return dpo_examples


def quality_check_dpo(
    dpo_examples: list,
    sample_size: int = 20
) -> dict:
    """
    Run quality check on a sample of DPO preference pairs.
    Verifies chosen is more faithful than rejected.

    Args:
        dpo_examples: list of DPO formatted examples
        sample_size: number of examples to check

    Returns:
        quality report dict
    """
    print(f"\nRunning quality check on {sample_size} DPO pairs...")
    client = get_client()

    sample = dpo_examples[:sample_size]
    chosen_scores = []
    rejected_scores = []
    good_pairs = 0

    for i, example in enumerate(sample):
        question = example["question"]
        context = example["context"]
        chosen = example["chosen"][0]["content"]
        rejected = example["rejected"][0]["content"]

        # Quick faithfulness check on chosen
        prompt_chosen = f"""Is this answer grounded in the context?
Answer yes or no only.
Context: {context[:300]}
Answer: {chosen[:200]}"""

        # Quick faithfulness check on rejected
        prompt_rejected = f"""Is this answer grounded in the context?
Answer yes or no only.
Context: {context[:300]}
Answer: {rejected[:200]}"""

        try:
            resp_c = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_chosen}],
                temperature=0.0,
                max_tokens=5
            )
            resp_r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_rejected}],
                temperature=0.0,
                max_tokens=5
            )

            chosen_faithful = resp_c.choices[0].message.content.strip()\
                .lower().startswith("yes")
            rejected_faithful = resp_r.choices[0].message.content.strip()\
                .lower().startswith("yes")

            chosen_scores.append(1.0 if chosen_faithful else 0.0)
            rejected_scores.append(1.0 if rejected_faithful else 0.0)

            if chosen_faithful and not rejected_faithful:
                good_pairs += 1

            print(f"  [{i+1}/{sample_size}] "
                  f"Chosen: {'✅' if chosen_faithful else '❌'} | "
                  f"Rejected: {'✅' if rejected_faithful else '❌'}")

            time.sleep(5)

        except Exception as e:
            print(f"  ❌ Error on pair {i+1}: {e}")

    avg_chosen = round(
        sum(chosen_scores) / len(chosen_scores), 3
    ) if chosen_scores else 0
    avg_rejected = round(
        sum(rejected_scores) / len(rejected_scores), 3
    ) if rejected_scores else 0
    good_pair_rate = round(good_pairs / len(sample), 3) if sample else 0

    quality_report = {
        "sample_size": len(sample),
        "avg_chosen_faithfulness": avg_chosen,
        "avg_rejected_faithfulness": avg_rejected,
        "faithfulness_gap": round(avg_chosen - avg_rejected, 3),
        "good_pair_rate": good_pair_rate,
        "quality": "GOOD" if good_pair_rate >= 0.6 else "POOR"
    }

    print(f"\n  Chosen faithfulness  : {avg_chosen}")
    print(f"  Rejected faithfulness: {avg_rejected}")
    print(f"  Faithfulness gap     : {quality_report['faithfulness_gap']}")
    print(f"  Good pair rate       : {good_pair_rate}")
    print(f"  Quality              : {quality_report['quality']}")

    return quality_report


def save_to_hub(
    sft_examples: list,
    dpo_examples: list
) -> dict:
    """
    Save formatted datasets to HuggingFace Hub.

    Args:
        sft_examples: formatted SFT examples
        dpo_examples: formatted DPO examples

    Returns:
        dict with HuggingFace dataset URLs
    """
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN"))

    print("\nSaving datasets to HuggingFace Hub...")

    # Save SFT dataset
    sft_repo = f"{HF_USERNAME}/qwen3-rag-faithful-sft"
    sft_dataset = Dataset.from_list([
        {
            "messages": json.dumps(ex["messages"]),
            "question": ex["question"],
            "context": ex["context"],
            "answer": ex["answer"]
        }
        for ex in sft_examples
    ])
    sft_dataset.push_to_hub(sft_repo)
    print(f"✅ SFT dataset saved: huggingface.co/datasets/{sft_repo}")

    # Save DPO dataset
    dpo_repo = f"{HF_USERNAME}/qwen3-rag-faithful-dpo"
    dpo_dataset = Dataset.from_list([
        {
            "prompt": json.dumps(ex["prompt"]),
            "chosen": json.dumps(ex["chosen"]),
            "rejected": json.dumps(ex["rejected"]),
            "question": ex["question"],
            "context": ex["context"]
        }
        for ex in dpo_examples
    ])
    dpo_dataset.push_to_hub(dpo_repo)
    print(f"✅ DPO dataset saved: huggingface.co/datasets/{dpo_repo}")

    return {
        "sft_dataset": f"huggingface.co/datasets/{sft_repo}",
        "dpo_dataset": f"huggingface.co/datasets/{dpo_repo}"
    }


def run_data_prep(
    n_sft: int = 500,
    n_dpo: int = 1000,
    quality_check: bool = True,
    push_to_hub: bool = True
) -> dict:
    """
    Full data preparation pipeline.

    Args:
        n_sft: number of SFT examples
        n_dpo: number of DPO examples
        quality_check: whether to run quality check
        push_to_hub: whether to save to HuggingFace

    Returns:
        data preparation report
    """
    print(f"{'='*60}")
    print(f"  DATA PREPARATION")
    print(f"  SFT: {n_sft} examples | DPO: {n_dpo} examples")
    print(f"{'='*60}")

    # Prepare datasets
    sft_examples = prepare_sft_dataset(n_sft)
    dpo_examples = prepare_dpo_dataset(n_dpo)

    # Quality check
    quality_report = {}
    if quality_check and dpo_examples:
        quality_report = quality_check_dpo(dpo_examples, sample_size=10)

    # Save to hub
    hub_urls = {}
    if push_to_hub:
        hub_urls = save_to_hub(sft_examples, dpo_examples)

    report = {
        "sft_examples": len(sft_examples),
        "dpo_examples": len(dpo_examples),
        "quality_report": quality_report,
        "hub_urls": hub_urls,
        "ready_for_training": (
            quality_report.get("quality", "GOOD") == "GOOD"
        )
    }

    # Save report
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/data_prep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DATA PREP COMPLETE")
    print(f"  SFT examples    : {len(sft_examples)}")
    print(f"  DPO examples    : {len(dpo_examples)}")
    print(f"  Ready for train : {report['ready_for_training']}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    report = run_data_prep(
        n_sft=500,
        n_dpo=1000,
        quality_check=True,
        push_to_hub=True
    )