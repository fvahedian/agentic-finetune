# agent/finetune_agent.py
"""
NeMo ReAct Orchestrator for the Agentic Fine-tuning Pipeline.

Registers all six tools as NeMo components and wires them
into a ReAct agent that reasons through the full pipeline:
1. Diagnose base model
2. Check data and training status
3. Evaluate base vs fine-tuned
4. Decide: DEPLOY / RETRAIN / ADJUST / ESCALATE
5. Trigger retraining if needed
6. Deploy if quality passes
"""

import sys
import os

# Add project root to path so tools can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from tools.diagnosis_tool import run_diagnosis
from tools.training_monitor_tool import run_training_monitor
from tools.evaluation_tool import run_evaluation
from tools.decision_tool import make_decision
from tools.training_trigger_tool import run_training_trigger
from tools.deployment_tool import run_deployment


# ── Tool 1: Diagnosis ─────────────────────────────────────────
class DiagnosisToolConfig(FunctionBaseConfig, name="diagnosis_tool"):
    """
    Diagnoses base model failure modes before training.
    Runs 5 metrics: faithfulness, relevance, refusal rate,
    consistency, conciseness.
    Returns failure pattern and recommended training strategy.
    Run this first before any training decisions.
    """
    n_examples: int = Field(
        default=5,
        description="Number of test examples to diagnose on."
    )


@register_function(
    config_type=DiagnosisToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def diagnosis_tool_function(
    config: DiagnosisToolConfig,
    builder: Builder
):
    async def _diagnose(task: str = "diagnose base model") -> str:
        """
        Runs 5 evaluation metrics on the base model to identify
        failure patterns and recommend a training strategy.
        Always run this first before deciding on training approach.
        Returns: failure_type, recommended_strategy, metric scores.
        """
        result = run_diagnosis(n_examples=config.n_examples)
        return json.dumps(result)

    yield FunctionInfo.from_fn(_diagnose, description=_diagnose.__doc__)


# ── Tool 2: Training Monitor ──────────────────────────────────
class TrainingMonitorConfig(FunctionBaseConfig, name="training_monitor_tool"):
    """
    Checks HuggingFace Hub for fine-tuned model and dataset status.
    Returns whether training is needed or model is ready to evaluate.
    Run this after diagnosis to check if training has been done.
    """
    pass


@register_function(
    config_type=TrainingMonitorConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def training_monitor_function(
    config: TrainingMonitorConfig,
    builder: Builder
):
    async def _check_training(task: str = "check training status") -> str:
        """
        Checks HuggingFace Hub for fine-tuned model availability.
        Also verifies SFT and DPO datasets are ready for training.
        Returns: model_exists, training_needed, dataset_status,
        recommendation (TRAIN or EVALUATE).
        """
        result = run_training_monitor()
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _check_training,
        description=_check_training.__doc__
    )


# ── Tool 3: Evaluation ────────────────────────────────────────
class EvaluationToolConfig(FunctionBaseConfig, name="evaluation_tool"):
    """
    Compares base model vs fine-tuned model on 5 metrics.
    Faithfulness, relevance, refusal rate, consistency, conciseness.
    Run after training is complete to measure improvement.
    """
    n_examples: int = Field(
        default=5,
        description="Number of test examples per model."
    )


@register_function(
    config_type=EvaluationToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def evaluation_tool_function(
    config: EvaluationToolConfig,
    builder: Builder
):
    async def _evaluate(task: str = "evaluate models") -> str:
        """
        Runs before/after comparison of base vs fine-tuned model.
        Measures faithfulness delta, hallucination reduction,
        relevance, refusal rate changes across both models.
        Returns comparison report with verdict: IMPROVED or NO_IMPROVEMENT.
        """
        result = run_evaluation(n_examples=config.n_examples)
        return json.dumps(result)

    yield FunctionInfo.from_fn(_evaluate, description=_evaluate.__doc__)


# ── Tool 4: Decision ──────────────────────────────────────────
class DecisionToolConfig(FunctionBaseConfig, name="decision_tool"):
    """
    Makes deployment decision based on evaluation results.
    Reasons about multi-metric tradeoffs and returns:
    DEPLOY, RETRAIN, ADJUST, or ESCALATE with reasoning.
    Run this last after evaluation to get final recommendation.
    """
    pass


@register_function(
    config_type=DecisionToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def decision_tool_function(
    config: DecisionToolConfig,
    builder: Builder
):
    async def _decide(task: str = "make deployment decision") -> str:
        """
        Reads evaluation report and makes deployment decision.
        Reasons about faithfulness improvement, hallucination rate,
        refusal rate, and relevance tradeoffs.
        Returns: DEPLOY (all metrics pass), RETRAIN (improvement
        below threshold), ADJUST (specific metric failed),
        or ESCALATE (conflicting signals need human review).
        """
        report_path = "eval/results/evaluation_report.json"
        if not os.path.exists(report_path):
            return json.dumps({
                "decision": "ESCALATE",
                "reasoning": "No evaluation report found. Run evaluation first."
            })
        with open(report_path) as f:
            evaluation_report = json.load(f)
        result = make_decision(evaluation_report)
        return json.dumps(result)

    yield FunctionInfo.from_fn(_decide, description=_decide.__doc__)


# ── Tool 5: Training Trigger ──────────────────────────────────
class TrainingTriggerConfig(FunctionBaseConfig, name="training_trigger_tool"):
    """
    Autonomously triggers retraining when decision = RETRAIN.
    Analyzes failure, adjusts hyperparameters, prepares improved
    dataset, and submits training job to RunPod (simulated).
    Call this when decision_tool returns RETRAIN.
    """
    pass


@register_function(
    config_type=TrainingTriggerConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def training_trigger_function(
    config: TrainingTriggerConfig,
    builder: Builder
):
    async def _trigger_training(
        task: str = "trigger retraining"
    ) -> str:
        """
        Triggers autonomous retraining when fine-tuning did not improve.
        Analyzes failure mode, adjusts DPO beta and epochs, prepares
        improved dataset, submits job to RunPod GPU cloud (simulated).
        Returns: job_id, status, ETA, adjusted parameters.
        Call this when decision_tool returns RETRAIN decision.
        """
        result = run_training_trigger()
        return json.dumps(result, default=str)

    yield FunctionInfo.from_fn(
        _trigger_training,
        description=_trigger_training.__doc__
    )


# ── Tool 6: Deployment ────────────────────────────────────────
class DeploymentToolConfig(FunctionBaseConfig, name="deployment_tool"):
    """
    Autonomously deploys the fine-tuned model when decision = DEPLOY.
    Creates FastAPI service, Dockerfile, starts server, runs health check.
    Call this when decision_tool returns DEPLOY.
    """
    pass


@register_function(
    config_type=DeploymentToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def deployment_tool_function(
    config: DeploymentToolConfig,
    builder: Builder
):
    async def _deploy(task: str = "deploy model") -> str:
        """
        Deploys fine-tuned model as FastAPI service when quality passes.
        Creates serve/app.py, Dockerfile, starts service on port 8000,
        runs health check and returns deployment status.
        Call this ONLY when decision_tool returns DEPLOY decision.
        Do NOT call if decision is RETRAIN or ESCALATE.
        """
        result = run_deployment()
        return json.dumps(result, default=str)

    yield FunctionInfo.from_fn(
        _deploy,
        description=_deploy.__doc__
    )