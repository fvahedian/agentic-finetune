# Agentic Fine-tuning Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fvahedian/agentic-finetune/blob/main/training/colab_training.ipynb)

A NeMo ReAct orchestrator that autonomously runs the full LLM fine-tuning lifecycle — diagnosing failure modes, preparing data, triggering retraining, evaluating improvement, and deploying or retraining based on multi-metric results.

---

## What Makes This Agentic

The NeMo ReAct orchestrator makes real decisions at each step:

- **Diagnoses** base model failure patterns before training
- **Evaluates** 5 metrics: faithfulness, relevance, refusal rate, consistency, conciseness
- **Decides** DEPLOY / RETRAIN / ADJUST / ESCALATE based on multi-metric tradeoffs
- **Triggers retraining autonomously** — adjusts DPO beta, epochs, filters dataset
- **Deploys automatically** when quality passes threshold

A pipeline runs fixed steps. This agent reasons about what to do next.

---

## Architecture

```
NeMo ReAct Orchestrator
        ↓
┌─────────────────────────────────────────────────┐
│ diagnosis_tool     → identifies failure pattern  │
│ training_monitor   → checks HuggingFace Hub      │
│ evaluation_tool    → base vs fine-tuned metrics  │
│ decision_tool      → DEPLOY/RETRAIN/ESCALATE     │
│ training_trigger   → adjusts params + submits    │
│ deployment_tool    → FastAPI service + Docker     │
└─────────────────────────────────────────────────┘
```

---

## Model

- **Base:** Qwen2.5-7B-Instruct
- **Method:** QLoRA (rank=16, 40.4M trainable params, 0.92%)
- **SFT:** SQuAD v1.1 (500 examples) — faithful answering
- **DPO:** HaluEval QA (1000 pairs) — prefer faithful over hallucinated
- **SFT loss:** 1.99 → 0.51

HuggingFace adapters:
- SFT: [ftimavn/qwen25-rag-faithful-sft-lora](https://huggingface.co/ftimavn/qwen25-rag-faithful-sft-lora)
- DPO: [ftimavn/qwen25-rag-faithful-dpo-lora](https://huggingface.co/ftimavn/qwen25-rag-faithful-dpo-lora)

---

## Agentic Reasoning Trace

```
Input: "Run the full agentic fine-tuning pipeline"

Thought: Check training status first
Action:  training_monitor_tool → no model found

Thought: Diagnose base model before training
Action:  diagnosis_tool → mild_hallucination (20%), use DPO

Thought: Evaluate base vs fine-tuned
Action:  evaluation_tool → NO_IMPROVEMENT, faithfulness +0.0

Thought: Make deployment decision
Action:  decision_tool → RETRAIN (faithfulness below threshold)

Thought: Trigger autonomous retraining
Action:  training_trigger_tool
         → root cause: faithfulness_not_improved
         → adjusted: DPO beta 0.1 to 0.2, epochs 1 to 2
         → submitted job to RunPod (simulated)
         → ETA: 2 hours

Final: "Model retraining submitted. Re-evaluate in 2 hours."
```

---

## Quick Start

```bash
conda create -n agentic_ft python=3.11 -y
conda activate agentic_ft

git clone https://github.com/fvahedian/agentic-finetune.git
cd agentic-finetune

pip install nvidia-nat-core nvidia-nat-langchain \
            langchain-nvidia-ai-endpoints langchain-classic \
            langchain-text-splitters openai datasets pandas \
            python-dotenv huggingface_hub fastapi uvicorn \
            groq litellm

pip install -e . --no-deps

cp .env.example .env
# Add your NVIDIA_API_KEY, GROQ_API_KEY, HF_TOKEN
```

Run the full autonomous pipeline:

```bash
GROQ_API_KEY=your-key \
nat run \
  --config_file agent/configs/config.yml \
  --input "Run the full agentic fine-tuning pipeline"
```

Run individual tools:

```bash
python tools/diagnosis_tool.py
python tools/evaluation_tool.py
python tools/decision_tool.py
python tools/training_trigger_tool.py
```

---

## Training on Colab

Open the training notebook on Colab A100:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fvahedian/agentic-finetune/blob/main/training/colab_training.ipynb)

Runs QLoRA SFT + DPO training on Qwen2.5-7B. Saves LoRA adapters to HuggingFace Hub.

---

## Datasets

- SFT: [ftimavn/qwen3-rag-faithful-sft](https://huggingface.co/datasets/ftimavn/qwen3-rag-faithful-sft)
- DPO: [ftimavn/qwen3-rag-faithful-dpo](https://huggingface.co/datasets/ftimavn/qwen3-rag-faithful-dpo)

---

## Requirements

- Python 3.11
- NVIDIA API key (build.nvidia.com)
- Groq API key (console.groq.com)
- HuggingFace token (huggingface.co/settings/tokens)
- GPU for training (Google Colab Pro recommended)

---

## Author

Fatemeh Vahedian — Senior ML Scientist, Search and Discovery