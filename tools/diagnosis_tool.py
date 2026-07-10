# tools/diagnosis_tool.py
"""
Diagnosis Tool — Phase 1 of the Agentic Fine-tuning Pipeline

Runs 5 metrics on the base model to understand what's failing:
1. Faithfulness    — is the model hallucinating?
2. Answer Relevance — is it answering the question?
3. Refusal Rate    — is it refusing too often?
4. Consistency     — does it give same answer twice?
5. Conciseness     — is the answer appropriately short?

The orchestrator uses this diagnosis to decide
the training strategy before touching any data.
"""

import os
import json
import time
import numpy as np
from groq import Groq
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def get_client():
    return Groq(
        
        api_key=os.getenv("GROQ_API_KEY")
    )


def generate_answer(question: str, context: str, loose: bool = False) -> str:
    """Generate answer using Qwen3 via NVIDIA NIM."""
    client = get_client()

    if loose:
        prompt = f"""Answer the question using the context if helpful,
but you can also use your general knowledge.
Be confident and specific.

Context: {context}
Question: {question}
Answer:"""
    else:
        prompt = f"""Answer the question using ONLY the provided context.
If the answer is not in the context, say "I cannot answer from the context."
Be concise.

Context: {context}
Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


def check_faithfulness(question: str, answer: str, context: str) -> float:
    """Check if answer is grounded in context using claim verification."""
    client = get_client()

    # Extract claims
    claims_prompt = f"""Break this answer into atomic claims.
Return only a JSON array of strings.
Answer: {answer}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": claims_prompt}],
        temperature=0.0,
        max_tokens=200
    )

    try:
        raw = response.choices[0].message.content.strip()
        # handle markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        claims = json.loads(raw.strip())
    except:
        claims = [answer]

    if not claims:
        return 1.0

    # Verify each claim
    supported = 0
    for claim in claims:
        verify_prompt = f"""Is this claim supported by the context?
Answer only yes or no.
Context: {context[:400]}
Claim: {claim}"""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=5
        )
        if resp.choices[0].message.content.strip().lower().startswith("yes"):
            supported += 1

    return round(supported / len(claims), 3)


def check_relevance(question: str, answer: str) -> float:
    """Check if answer addresses the question."""
    client = get_client()

    prompt = f"""On a scale of 1-5, how well does this answer
address the question? Return only a number.

Question: {question}
Answer: {answer}
Score (1-5):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )

    try:
        score = float(response.choices[0].message.content.strip()[0])
        return round(score / 5.0, 3)
    except:
        return 0.5


def check_refusal(answer: str) -> bool:
    """Check if the model refused to answer."""
    refusal_phrases = [
        "i cannot", "i can't", "not in the context",
        "cannot answer", "don't know", "no information",
        "context does not", "context doesn't"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)


def check_consistency(question: str, context: str) -> float:
    """Ask same question twice and measure agreement."""
    client = get_client()

    prompt = f"""Answer using ONLY the context. Be concise.
Context: {context}
Question: {question}
Answer:"""

    answers = []
    for _ in range(2):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        answers.append(response.choices[0].message.content.strip())
        time.sleep(2)

    # Simple consistency — do answers share key words?
    words1 = set(answers[0].lower().split())
    words2 = set(answers[1].lower().split())
    if not words1 or not words2:
        return 0.5
    overlap = len(words1 & words2) / len(words1 | words2)
    return round(overlap, 3)


def run_diagnosis(n_examples: int = 5) -> dict:
    """
    Run full diagnosis on base model.

    Args:
        n_examples: number of SQuAD examples to test on

    Returns:
        diagnosis report with all metrics and failure patterns
    """
    print(f"{'='*60}")
    print(f"  DIAGNOSIS — Base Model (Qwen3)")
    print(f"  Testing on {n_examples} SQuAD examples")
    print(f"{'='*60}\n")

    # Load SQuAD
    dataset = load_dataset(
        "rajpurkar/squad",
        split=f"validation[:{n_examples}]"
    )

    results = []
    faithfulness_scores = []
    relevance_scores = []
    refusal_flags = []
    consistency_scores = []
    answer_lengths = []

    for i, example in enumerate(dataset):
        question = example["question"]
        context = example["context"]
        gold_answer = example["answers"]["text"][0]

        print(f"[{i+1}/{n_examples}] {question[:55]}...")

        try:
            # Generate with loose prompt to expose hallucinations
            answer = generate_answer(question, context, loose=True)
            print(f"  Answer: {answer[:70]}")

            # Metric 1: Faithfulness
            faith = check_faithfulness(question, answer, context)
            faithfulness_scores.append(faith)

            # Metric 2: Relevance
            relevance = check_relevance(question, answer)
            relevance_scores.append(relevance)

            # Metric 3: Refusal
            refused = check_refusal(answer)
            refusal_flags.append(refused)

            # Metric 4: Consistency
            consistency = check_consistency(question, context)
            consistency_scores.append(consistency)

            # Metric 5: Conciseness
            answer_lengths.append(len(answer.split()))

            result = {
                "question": question,
                "context": context[:200],
                "gold_answer": gold_answer,
                "generated_answer": answer,
                "faithfulness": faith,
                "relevance": relevance,
                "refused": refused,
                "consistency": consistency,
                "answer_length": len(answer.split())
            }
            results.append(result)

            print(f"  Faith: {faith} | Rel: {relevance} | "
                  f"Refused: {refused} | Consist: {consistency}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        if i < n_examples - 1:
            print(f"  Waiting 10s...")
            time.sleep(10)

    # Guard against empty results
    if not faithfulness_scores:
        print("❌ No successful examples. Check model name and API key.")
        return {}

    # Compute summary
    avg_faith = round(float(np.mean(faithfulness_scores)), 3)
    avg_relevance = round(float(np.mean(relevance_scores)), 3)
    refusal_rate = round(float(np.mean(refusal_flags)), 3)
    avg_consistency = round(float(np.mean(consistency_scores)), 3)
    avg_length = round(float(np.mean(answer_lengths)), 1)
    hallucination_rate = round(
        sum(1 for f in faithfulness_scores if f < 0.5)
        / len(faithfulness_scores), 3
    )

    # Classify dominant failure
    if hallucination_rate > 0.3:
        failure_type = "high_hallucination"
        strategy = "aggressive — use SFT + DPO with strict faithfulness"
    elif refusal_rate > 0.3:
        failure_type = "over_refusal"
        strategy = "balanced — use SFT only with positive faithful examples"
    elif avg_consistency < 0.5:
        failure_type = "inconsistency"
        strategy = "consistency — use temperature reduction + SFT"
    else:
        failure_type = "mild_hallucination"
        strategy = "gentle — use DPO only"

    diagnosis = {
        "n_examples": len(results),
        "model": MODEL,
        "metrics": {
            "avg_faithfulness": avg_faith,
            "avg_relevance": avg_relevance,
            "refusal_rate": refusal_rate,
            "avg_consistency": avg_consistency,
            "avg_answer_length": avg_length,
            "hallucination_rate": hallucination_rate
        },
        "dominant_failure": failure_type,
        "recommended_strategy": strategy,
        "examples": results
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS REPORT")
    print(f"{'='*60}")
    print(f"  Model             : {MODEL}")
    print(f"  Avg Faithfulness  : {avg_faith}")
    print(f"  Avg Relevance     : {avg_relevance}")
    print(f"  Refusal Rate      : {refusal_rate}")
    print(f"  Avg Consistency   : {avg_consistency}")
    print(f"  Avg Answer Length : {avg_length} words")
    print(f"  Hallucination Rate: {hallucination_rate}")
    print(f"\n  Dominant Failure  : {failure_type}")
    print(f"  Strategy          : {strategy}")
    print(f"{'='*60}")

    # Save report
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/diagnosis_report.json", "w") as f:
        json.dump(diagnosis, f, indent=2)
    print(f"\n  Report saved to eval/results/diagnosis_report.json")

    return diagnosis


if __name__ == "__main__":
    diagnosis = run_diagnosis(n_examples=5)