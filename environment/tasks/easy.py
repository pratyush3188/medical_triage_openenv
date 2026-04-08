import random
from typing import Dict, Any, List
from ..models import Observation, Action, Vitals, AvailableResources

EPS_SCORE = 1e-4  # keep safe after 4dp rounding: never 0.0000/1.0000

def _strict_unit_interval(x: float, eps: float = EPS_SCORE) -> float:
    x = float(x)
    x = max(eps, min(1.0 - eps, x))
    x = round(x, 4)
    return max(eps, min(1.0 - eps, x))

PATIENTS = [
    {
        "patient_id": "E1",
        "age": 67,
        "chief_complaint": "crushing chest pain",
        "visible_vitals": {"heart_rate": 110, "blood_pressure": "90/60", "temperature": 37.1, "oxygen_saturation": 94},
        "additional_info": "ECG shows ST-elevation indicating a heart attack.",
        "true_priority": "CRITICAL"
    },
    {
        "patient_id": "E2",
        "age": 23,
        "chief_complaint": "mild headache",
        "visible_vitals": {"heart_rate": 72, "blood_pressure": "118/76", "temperature": 37.0, "oxygen_saturation": 99},
        "additional_info": "Patient has had adequate water and symptoms resolve with ibuprofen.",
        "true_priority": "LOW"
    },
    {
        "patient_id": "E3",
        "age": 45,
        "chief_complaint": "shortness of breath",
        "visible_vitals": {"heart_rate": 95, "blood_pressure": "130/85", "temperature": 37.8, "oxygen_saturation": 91},
        "additional_info": "History of asthma; wheezing heard on auscultation.",
        "true_priority": "HIGH"
    },
    {
        "patient_id": "E4",
        "age": 31,
        "chief_complaint": "sprained ankle",
        "visible_vitals": {"heart_rate": 80, "blood_pressure": "120/80", "temperature": 36.9, "oxygen_saturation": 98},
        "additional_info": "No bone deformities visible. Pain when bearing weight.",
        "true_priority": "LOW"
    },
    {
        "patient_id": "E5",
        "age": 58,
        "chief_complaint": "sudden severe headache 'worst of life'",
        "visible_vitals": {"heart_rate": 88, "blood_pressure": "180/110", "temperature": 37.2, "oxygen_saturation": 97},
        "additional_info": "Potential subarachnoid hemorrhage. Patient is photophobic.",
        "true_priority": "CRITICAL"
    },
    {
        "patient_id": "E6",
        "age": 8,
        "chief_complaint": "allergic reaction to peanuts",
        "visible_vitals": {"heart_rate": 125, "blood_pressure": "100/65", "temperature": 37.0, "oxygen_saturation": 95},
        "additional_info": "Mild facial swelling but airway is clear. Epipen not yet administered.",
        "true_priority": "HIGH"
    },
    {
        "patient_id": "E7",
        "age": 82,
        "chief_complaint": "fall from standing height",
        "visible_vitals": {"heart_rate": 85, "blood_pressure": "140/85", "temperature": 36.8, "oxygen_saturation": 98},
        "additional_info": "Small skin tear on arm. No head trauma, fully oriented.",
        "true_priority": "LOW"
    },
    {
        "patient_id": "E8",
        "age": 5,
        "chief_complaint": "high fever and barky cough",
        "visible_vitals": {"heart_rate": 130, "blood_pressure": "105/70", "temperature": 39.2, "oxygen_saturation": 96},
        "additional_info": "Possible croup. Respiration slightly noisy but no distress.",
        "true_priority": "MEDIUM"
    },
    {
        "patient_id": "E9",
        "age": 29,
        "chief_complaint": "paper cut",
        "visible_vitals": {"heart_rate": 72, "blood_pressure": "120/80", "temperature": 36.6, "oxygen_saturation": 100},
        "additional_info": "Superficial. No bleeding currently. Just seeking a band-aid.",
        "true_priority": "LOW"
    },
    {
        "patient_id": "E10",
        "age": 64,
        "chief_complaint": "sudden facial drooping",
        "visible_vitals": {"heart_rate": 78, "blood_pressure": "175/95", "temperature": 36.9, "oxygen_saturation": 97},
        "additional_info": "Strong suspicion of acute stroke. Symptom onset 30 mins ago.",
        "true_priority": "CRITICAL"
    }
]

PRIORITY_LEVELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

class EasyTask:
    def __init__(self):
        self.patient = random.choice(PATIENTS)
        self.turn = 0
        self.revealed_info = None
        
    def reset(self):
        self.patient = random.choice(PATIENTS)
        self.turn = 0
        self.revealed_info = None
        return self.get_obs(msg="A new patient has arrived at the triage desk. Assess priority.")

    def step(self, action: Action):
        self.turn += 1
        reward = 0.0
        done = False
        msg = ""

        if action.action_type == "ask_followup":
            self.revealed_info = self.patient["additional_info"]
            reward = 0.2
            msg = f"Additional information revealed: {self.revealed_info}"
        elif action.action_type == "assign_priority":
            true_p = PRIORITY_LEVELS[self.patient["true_priority"]]
            pred_p = PRIORITY_LEVELS.get(action.priority, -1)
            if pred_p == -1:
                reward = -0.5
                msg = "Invalid priority assigned."
            else:
                diff = abs(true_p - pred_p)
                if diff == 0:
                    reward = 1.0
                    msg = "Correct priority assigned."
                elif diff == 1:
                    reward = 0.5
                    msg = "Priority off by one level."
                else:
                    reward = -0.5
                    msg = "Wrong priority assigned."
            done = True
        else:
            reward = -0.1
            msg = "Invalid action for Easy task. Only ask_followup or assign_priority allowed."
            
        if self.turn >= 5 and not done:
            reward += -0.5
            done = True
            msg = "Turn limit reached. Forced termination."

        obs = self.get_obs(msg)
        # Map reward in [-0.5, 1.0] to score in [0, 1], then force into (0, 1).
        raw_score = (reward + 0.5) / 1.5
        score = _strict_unit_interval(raw_score)
        return obs, reward, done, {"true_priority": self.patient["true_priority"], "score": score}

    def get_obs(self, msg=""):
        return Observation(
            patient_id=self.patient["patient_id"],
            age=self.patient["age"],
            chief_complaint=self.patient["chief_complaint"],
            visible_vitals=Vitals(**self.patient["visible_vitals"]),
            additional_info=self.revealed_info,
            available_resources=AvailableResources(),
            turn_number=self.turn,
            task_level="easy",
            message=msg
        )
