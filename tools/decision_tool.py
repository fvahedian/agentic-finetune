# tools/decision_tool.py
"""
Decision Tool — Final step of the Agentic Fine-tuning Pipeline

Takes the evaluation report and makes a deployment decision:
- DEPLOY    : improvement above threshold, all metrics acceptable
- RETRAIN   : improvement below threshold, more training needed
- ADJUST    : specific metric degraded, adjust dataset and retry
- ESCALATE  : conflicting signals, needs human review

This is the genuinely agentic piece — the orchestrator
reasons about multi-metric tradeoffs and picks a strategy.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# Thresholds
MIN_FAITHFULNESS_IMPROVEMENT = 0.05
MAX_REFUSAL_RATE = 0.30
MIN_RELEVANCE = 0.60
MAX_HALLUCINATION_RATE = 0.20


def make_decision(evaluation_report: dict) -> dict:
    """
    Makes deployment decision based on evaluation report.

    Args:
        evaluation_report: output from evaluation_tool.run_evaluation()

    Returns:
        decision dict with action and reasoning
    """
    if not evaluation_report:
        return {
            "decision": "ESCALATE",
            "reasoning": "No evaluation report available.",
            "action": "Run evaluation tool first."
        }

    base = evaluation_report.get("base_model", {})
    finetuned = evaluation_report.get("finetuned_model", {})
    improvement = evaluation_report.get("improvement", {})

    faith_delta = improvement.get("faithfulness_delta", 0)
    halluc_reduction = improvement.get("hallucination_reduction", 0)
    finetuned_refusal = finetuned.get("refusal_rate", 0)
    finetuned_relevance = finetuned.get("avg_relevance", 0)
    finetuned_halluc = finetuned.get("hallucination_rate", 1)

    reasoning = []
    issues = []

    # Check faithfulness improvement
    if faith_delta >= MIN_FAITHFULNESS_IMPROVEMENT:
        reasoning.append(
            f"✅ Faithfulness improved by {faith_delta:+.3f}"
        )
    else:
        issues.append(
            f"❌ Faithfulness improvement {faith_delta:+.3f} "
            f"below threshold {MIN_FAITHFULNESS_IMPROVEMENT}"
        )

    # Check hallucination rate
    if finetuned_halluc <= MAX_HALLUCINATION_RATE:
        reasoning.append(
            f"✅ Hallucination rate {finetuned_halluc} "
            f"within threshold {MAX_HALLUCINATION_RATE}"
        )
    else:
        issues.append(
            f"❌ Hallucination rate {finetuned_halluc} "
            f"above threshold {MAX_HALLUCINATION_RATE}"
        )

    # Check refusal rate
    if finetuned_refusal <= MAX_REFUSAL_RATE:
        reasoning.append(
            f"✅ Refusal rate {finetuned_refusal} acceptable"
        )
    else:
        issues.append(
            f"⚠️  Refusal rate {finetuned_refusal} too high — "
            f"model over-refuses, needs balanced training"
        )

    # Check relevance
    if finetuned_relevance >= MIN_RELEVANCE:
        reasoning.append(
            f"✅ Relevance {finetuned_relevance} above minimum"
        )
    else:
        issues.append(
            f"❌ Relevance {finetuned_relevance} "
            f"dropped below {MIN_RELEVANCE}"
        )

    # Make decision
    if not issues:
        decision = "DEPLOY"
        action = (
            "All metrics passed. Deploy fine-tuned model to FastAPI service."
        )
    elif len(issues) == 1 and "refusal" in issues[0]:
        decision = "ADJUST"
        action = (
            "Refusal rate too high. Add more positive faithful examples "
            "to SFT dataset and retrain with lower faithfulness threshold."
        )
    elif faith_delta < 0:
        decision = "RETRAIN"
        action = (
            "Faithfulness degraded. Increase DPO beta, "
            "filter higher quality preference pairs, retrain."
        )
    elif len(issues) >= 2:
        decision = "ESCALATE"
        action = (
            "Multiple metrics failed. Results unclear — "
            "needs human review before deployment."
        )
    else:
        decision = "RETRAIN"
        action = (
            "Improvement below threshold. "
            "Run more epochs or increase training data."
        )

    result = {
        "decision": decision,
        "action": action,
        "reasoning": reasoning,
        "issues": issues,
        "metrics_summary": {
            "faithfulness_delta": faith_delta,
            "hallucination_reduction": halluc_reduction,
            "finetuned_refusal_rate": finetuned_refusal,
            "finetuned_relevance": finetuned_relevance,
            "finetuned_hallucination_rate": finetuned_halluc
        }
    }

    # Print decision
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT DECISION")
    print(f"{'='*60}")
    print(f"\n  Decision : {decision}")
    print(f"  Action   : {action}")
    print(f"\n  Reasoning:")
    for r in reasoning:
        print(f"    {r}")
    if issues:
        print(f"\n  Issues:")
        for issue in issues:
            print(f"    {issue}")
    print(f"{'='*60}")

    # Save
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/decision_report.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Report saved to eval/results/decision_report.json")

    return result


if __name__ == "__main__":
    # Load evaluation report
    report_path = "eval/results/evaluation_report.json"

    if not os.path.exists(report_path):
        print("❌ No evaluation report found.")
        print("   Run evaluation_tool.py first.")
    else:
        with open(report_path) as f:
            evaluation_report = json.load(f)
        decision = make_decision(evaluation_report)