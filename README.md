---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# 🏥 Medical Triage OpenEnv

## 📄 Overview & Motivation
The **Medical Triage Environment** is a high-fidelity, multi-turn clinical simulation designed to evaluate the decision-making capabilities of LLM-based agents in an Emergency Room (ER) setting. 

Agents acting as **Triage AI** must navigate complex, life-or-death scenarios where information is often incomplete or intentionally misleading. The environment tests:
- **Clinical Reasoning**: Identifying life-threatening emergencies over stable conditions.
- **Resource Management**: Allocating scarce ventilators and ICU beds effectively.
- **Interrogation Strategy**: Knowing when to ask follow-up questions vs. when to take immediate action.

---

## 🚀 Key Features
- **OpenEnv Compliant**: Follows strict `reset`/`step`/`state` patterns for automated evaluation.
- **Dynamic Task Difficulty**: Scales from single-patient triage to complex crisis resource management.
- **Diverse Patient Pool**: Includes 25+ unique medical cases across all task levels.
- **OpenAI SDK Support**: Built to work natively with any OpenAI-compatible inference provider (Hugging Face, Groq, OpenAI).

---

## 🕹️ Action Space
Agents interact using a structured JSON schema. 

| action_type | Description | Values |
| :--- | :--- | :--- |
| `ask_followup` | Reveal hidden diagnostic info (CXR, ECG, Lab results). | `N/A` |
| `assign_priority` | Set a clinical priority level for a patient. | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `allocate_resource` | (Hard Task) Assign a restricted medical resource. | `icu_bed`, `ventilator`, `general_ward` |

```json
{
  "action_type": "allocate_resource",
  "patient_id": "H1",
  "resource": "ventilator",
  "notes": "Patient in severe respiratory distress with 82% O2 sat."
}
```

---

## 📊 Task Breakdown

### 🟢 Task 1: Easy — Single Patient Triage
- **Focus**: Baseline clinical interpretation.
- **Pool**: 10 Patients (`E1` - `E10`).
- **Challenge**: Mapping vitals and complaints to one of four priority levels.
- **Scoring**: Linear reward based on priority proximity (+1.0 for perfect match).

### 🟡 Task 2: Medium — Multi-Patient Prioritization
- **Focus**: Concurrent patient management and ranking.
- **Pool**: 7 Patients (`M1` - `M7`).
- **Challenge**: Agents must prioritize a full waiting room. Correctly identifying "CRITICAL" sepsis vs. a "LOW" priority laceration.
- **Scoring**: Weighted matching against a 7-patient ground truth list.

### 🔴 Task 3: Hard — ER Crisis Resource Allocation
- **Focus**: Crisis Management under severe constraints.
- **Pool**: 8 Patients (`H1` - `H8`).
- **Challenge**: Misleading symptoms (e.g., anxiety mimicking a heart attack) and strictly limited resources (only 1 ventilator, 3 ICU beds).
- **Scoring**: Custom partial-credit grader. Critical failure to allocate the ventilator to the respiratory distress patient results in a heavy penalty.

---

## ⚙️ Setup & Deployment

### 1. Environment Configuration
Create a `.env` file in the root directory:
```env
HF_TOKEN="your_huggingface_token"
API_BASE_URL="https://router.huggingface.co/v1"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
```

### 2. Local Setup
```bash
pip install -r requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

### 3. Running Inference Baseline
```bash
python inference.py
```

---

## 🐳 Docker Support
The environment is containerized for seamless deployment to **Hugging Face Spaces**.

```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

---

## 📝 Evaluation Logging Format
The environment produces structured logs with specific tags for automated parsing and evaluation:

- **`[START]`**: Emitted when a task begins. Contains metadata like task name, model, and timestamp.
- **`[STEP]`**: Emitted after every environment step. Tracks turn number, agent action, reward, and completion status.
- **`[END]`**: Emitted when a task finishes. Reports the final total reward, turns used, and the normalized task score (0.0 to 1.0).

---

## 📈 Scoring & Metrics
The results of an evaluation run are summarized by a **Total Average Mean** score:
$$\text{Mean Score} = \frac{\text{Easy Score} + \text{Medium Score} + \text{Hard Score}}{3}$$

- **Accuracy**: Measured by the precision of priority/resource assignments.
- **Efficiency**: Penalties apply if an agent takes too many turns to reach a decision.
- **Robustness**: JSON parsing logic ensures the environment doesn't crash on slightly malformed model outputs.

---

## 🛠️ Technical Notes
- **Pydantic V2**: This project uses Pydantic V2. All models use `.model_dump()` for serialization to ensure future compatibility.
- **Deterministic Grading**: Graders are designed to be deterministic—given the same action sequence, the score will always be identical.
- **Multi-Strategy JSON Parser**: The baseline inference script uses a 4-level fallback strategy (Markdown stripping, Brace matching, Regex, and Header checks) to maximize successfully parsed LLM responses.

---

> [!IMPORTANT]
> **Medical Logic Disclaimer**: This environment is for AI research and evaluation purposes only and does not constitute medical advice.
