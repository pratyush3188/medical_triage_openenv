import os
import re
import json
import time
import math
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv
from environment.env import MedicalTriageEnv

load_dotenv()  # Loads variables from a .env file if it exists

# Ensure environment variables are loaded
api_key = os.environ.get("HF_TOKEN", "")
base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

client = OpenAI(
    api_key=api_key if api_key else "dummy_key",
    base_url=base_url if base_url else None
)

MAX_TURNS = {
    "easy": 5,
    "medium": 10,
    "hard": 15
}

EPS_SCORE = 1e-4  # ensures 4dp rounding won't hit 0.0/1.0

def to_strict_unit_interval(x: float, eps: float = EPS_SCORE) -> float:
    """
    Force score into (0, 1) strictly (never 0.0 or 1.0),
    and keep it safe even after rounding to 4 decimals.
    """
    try:
        v = float(x)
    except Exception:
        v = 0.5

    if not math.isfinite(v):
        v = 0.5

    v = max(eps, min(1.0 - eps, v))
    v = round(v, 4)
    v = max(eps, min(1.0 - eps, v))
    return v

def extract_json_from_response(text: str):
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Strip markdown code blocks
    try:
        cleaned = re.sub(r'```(?:json|JSON)?\s*', '', text)
        cleaned = cleaned.replace('```', '').strip()
        return json.loads(cleaned)
    except Exception:
        pass
    
    # Strategy 3: Find matching braces (handles text before/after JSON)
    try:
        start = text.find('{')
        if start != -1:
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i+1])
    except Exception:
        pass
    
    # Strategy 4: Regex fallback
    try:
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    
    return None

def prompt_agent(client, model, obs_dict):
    patient_id = obs_dict.get("patient_id", "")
    task_level = obs_dict.get("task_level", "easy")

    system_prompt = """You are an ER triage AI. Output ONLY raw JSON. Nothing else.

RULES:
- Start your response with { 
- End your response with }
- No explanations, no markdown, no backticks, no code blocks
- Pure JSON only

For easy/medium tasks use:
{"action_type": "assign_priority", "patient_id": "PATIENT_ID", "priority": "CRITICAL", "notes": "reason"}

For hard task use:
{"action_type": "allocate_resource", "patient_id": "PATIENT_ID", "resource": "icu_bed", "notes": "reason"}

Priority values: LOW | MEDIUM | HIGH | CRITICAL
Resource values: icu_bed | ventilator | general_ward

START YOUR RESPONSE WITH { NOW."""

    user_prompt = (
        f"Current Observation:\n{json.dumps(obs_dict, indent=2)}\n\n"
        f"What is your next triage action? Respond with JSON only."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        content = response.choices[0].message.content or ""
        parsed = extract_json_from_response(content)

        if parsed and "action_type" in parsed:
            return parsed

        if parsed:
            parsed["action_type"] = parsed.get("action_type", "assign_priority")
            parsed["patient_id"] = parsed.get("patient_id", patient_id)
            return parsed

    except Exception as e:
        print(f"\n[DEBUG] LLM Call or Parse Exception: {e}")
        pass

    fallback_priorities = {
        "easy": "HIGH",
        "medium": "MEDIUM",
        "hard": "CRITICAL"
    }
    return {
        "action_type": "assign_priority",
        "patient_id": patient_id,
        "priority": fallback_priorities.get(task_level, "HIGH"),
        "notes": "fallback decision due to LLM parse error"
    }

def run_task(env: MedicalTriageEnv, task_name: str, model: str):
    obs = env.reset(task_name=task_name)
    total_reward = 0.0
    turns = 0
    done = False
    final_score = 0.0
    max_turns = MAX_TURNS.get(task_name, 10)

    print(f'\n\n---------------------------------------------------------')
    print(f'[START] {{"task": "{task_name}", "model": "{model}", "timestamp": "{datetime.now(timezone.utc).isoformat()}"}}\n')

    while not done and turns < max_turns:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
        action = prompt_agent(client, model, obs_dict)
        time.sleep(1)  # Sleep added to prevent rate limits or disconnections

        if not isinstance(action, dict) or "action_type" not in action:
            action = {
                "action_type": "assign_priority",
                "patient_id": obs_dict.get("patient_id", ""),
                "priority": "HIGH",
                "notes": "safety fallback"
            }

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            reward = -0.5
            done = True
            info = {"score": 0.0}

        total_reward += reward
        turns += 1

        if done and "score" in info:
            final_score = to_strict_unit_interval(info["score"])
        elif done:
            final_score = to_strict_unit_interval(total_reward)

        print(f'[STEP] {{"task": "{task_name}", "turn": {turns}, "action": {json.dumps(action)}, "reward": {round(reward, 4)}, "done": {str(done).lower()}}}\n')

    if not done and turns >= max_turns:
        print(f'[STEP] {{"task": "{task_name}", "turn": {turns}, "action": {{}}, "reward": -0.5, "done": true}}\n')
        total_reward -= 0.5
        final_score = to_strict_unit_interval(total_reward / max(turns, 1))

    final_score = to_strict_unit_interval(final_score)

    print(f'[END] {{"task": "{task_name}", "total_reward": {round(total_reward, 4)}, "turns": {turns}, "score": {final_score}}}')
    print(f"\n======== Task '{task_name.upper()}' Completed. Average Score Output: {final_score:.2f} ========\n")
    return final_score

def main():
    env = MedicalTriageEnv()
    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        score = run_task(env, task, model_name)
        scores[task] = score
        time.sleep(1)

    print("\n\n###########################################")
    print("             BASELINE RESULTS              ")
    print("###########################################\n")
    for task in tasks:
        print(f"Task -> {task.upper()}:\tscore={scores[task]:.2f}")

    mean_score = sum(scores.values()) / len(scores) if scores else 0.0
    print("\n-------------------------------------------")
    print(f"TOTAL AVERAGE MEAN:\tscore={mean_score:.2f}")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    main()