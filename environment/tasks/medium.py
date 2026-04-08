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
        "patient_id": "M1",
        "age": 72,
        "chief_complaint": "confusion + fever",
        "visible_vitals": {"heart_rate": 105, "blood_pressure": "88/58", "temperature": 39.8, "oxygen_saturation": 93},
        "additional_info": "Signs of sepsis and potential septic shock.",
        "true_rank": 1
    },
    {
        "patient_id": "M2",
        "age": 45,
        "chief_complaint": "chest tightness",
        "visible_vitals": {"heart_rate": 92, "blood_pressure": "145/90", "temperature": 37.1, "oxygen_saturation": 96},
        "additional_info": "ECG shows mild non-specific changes, pain is intermittent.",
        "true_rank": 2
    },
    {
        "patient_id": "M3",
        "age": 28,
        "chief_complaint": "abdominal pain",
        "visible_vitals": {"heart_rate": 88, "blood_pressure": "122/78", "temperature": 37.9, "oxygen_saturation": 98},
        "additional_info": "Right lower quadrant pain guarding, suspect appendicitis.",
        "true_rank": 3
    },
    {
        "patient_id": "M4",
        "age": 55,
        "chief_complaint": "dizziness",
        "visible_vitals": {"heart_rate": 78, "blood_pressure": "135/85", "temperature": 37.0, "oxygen_saturation": 97},
        "additional_info": "History of vertigo, normal neuro exam.",
        "true_rank": 4
    },
    {
        "patient_id": "M5",
        "age": 19,
        "chief_complaint": "cut on hand",
        "visible_vitals": {"heart_rate": 75, "blood_pressure": "118/76", "temperature": 36.8, "oxygen_saturation": 99},
        "additional_info": "Superficial laceration, no tendon involvement.",
        "true_rank": 5
    },
    {
        "patient_id": "M6",
        "age": 60,
        "chief_complaint": "vomiting blood",
        "visible_vitals": {"heart_rate": 115, "blood_pressure": "95/65", "temperature": 37.0, "oxygen_saturation": 98},
        "additional_info": "Upper GI bleed, suspected bleeding varices. Dizzy upon standing.",
        "true_rank": 2
    },
    {
        "patient_id": "M7",
        "age": 35,
        "chief_complaint": "migraine aura",
        "visible_vitals": {"heart_rate": 70, "blood_pressure": "130/80", "temperature": 37.0, "oxygen_saturation": 100},
        "additional_info": "Known migraine profile. Just needs dark room and meds.",
        "true_rank": 7
    }
]

PRIORITY_LEVELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

GROUND_TRUTH_MEDIUM = {
    "M1": "CRITICAL",
    "M2": "HIGH",
    "M3": "MEDIUM",
    "M4": "MEDIUM",
    "M5": "LOW",
    "M6": "CRITICAL",
    "M7": "LOW"
}

class MediumTask:
    def __init__(self):
        self.patients = PATIENTS.copy()
        self.current_patient_index = 0
        self.turn = 0
        self.revealed_infos = {p["patient_id"]: None for p in self.patients}
        self.assignments = {}
        self.message = "7 patients are waiting. Assign a priority to each to rank them effectively."

    def reset(self):
        self.current_patient_index = 0
        self.turn = 0
        self.revealed_infos = {p["patient_id"]: None for p in self.patients}
        self.assignments = {}
        self.message = "7 patients are waiting. All are presented simultaneously. You will assess and set priorities."
        return self.get_obs()

    def grade(self) -> float:
        total = 0.0
        max_possible = len(GROUND_TRUTH_MEDIUM)
        
        for patient_id, true_priority in GROUND_TRUTH_MEDIUM.items():
            assigned = self.assignments.get(patient_id, "LOW")
            true_level = PRIORITY_LEVELS.get(true_priority, 0)
            assigned_level = PRIORITY_LEVELS.get(
                assigned.upper() if assigned else "LOW", 0
            )
            diff = abs(true_level - assigned_level)
            
            if diff == 0:
                total += 1.0      # Perfect match
            elif diff == 1:
                total += 0.5      # One level off
            elif diff == 2:
                total += 0.25     # Two levels off
            else:
                total += 0.0      # Too far off
        
        final_score = total / max_possible
        return _strict_unit_interval(final_score)

    def step(self, action: Action):
        self.turn += 1
        reward = 0.0
        done = False
        final_score = 0.0

        if action.patient_id not in [p["patient_id"] for p in self.patients]:
            self.message = f"Invalid patient ID: {action.patient_id}"
            reward = -0.1
        else:
            if action.action_type == "ask_followup":
                patient = next(p for p in self.patients if p["patient_id"] == action.patient_id)
                self.revealed_infos[action.patient_id] = patient["additional_info"]
                reward = 0.2
                self.message = f"Follow-up for {action.patient_id}: {patient['additional_info']}"
            elif action.action_type == "assign_priority":
                if action.priority not in PRIORITY_LEVELS:
                    reward = -0.5
                    self.message = "Invalid priority level."
                else:
                    self.assignments[action.patient_id] = action.priority
                    reward = 0.5 
                    self.message = f"Priority {action.priority} assigned to {action.patient_id}."
                    
                self.current_patient_index += 1
            else:
                reward = -0.1
                self.message = "Invalid action. Use ask_followup or assign_priority."

        if self.current_patient_index >= len(self.patients):
            done = True
            final_score = self.grade()
            reward += final_score * 2
            self.message = f"All patients triaged. Ranking score: {final_score:.2f}"
            
        elif self.turn >= 25: 
            done = True
            reward -= 2.0
            self.message = "Turn limit reached. Did not assign priority to all patients."

        if not done:
            return self.get_obs(), reward, done, {}
        else:
            return self.get_final_obs(), reward, done, {"score": final_score}

    def get_final_obs(self):
        idx = min(self.current_patient_index, len(self.patients) - 1)
        patient = self.patients[idx]
        overview_msg = "ER Status:\n"
        for p in self.patients:
            stat = self.assignments.get(p["patient_id"], "Pending")
            overview_msg += f"- {p['patient_id']} ({p['age']}y {p['chief_complaint']}): {stat}\n"
        overview_msg += f"\n>>> {self.message}"
        return Observation(
            patient_id=patient["patient_id"],
            age=patient["age"],
            chief_complaint=patient["chief_complaint"],
            visible_vitals=Vitals(**patient["visible_vitals"]),
            additional_info=self.revealed_infos[patient["patient_id"]],
            available_resources=AvailableResources(),
            turn_number=self.turn,
            task_level="medium",
            message=overview_msg
        )

    def get_obs(self):
        patient = self.patients[self.current_patient_index]
        overview_msg = "ER Status:\n"
        for p in self.patients:
            stat = self.assignments.get(p["patient_id"], "Pending")
            overview_msg += f"- {p['patient_id']} ({p['age']}y {p['chief_complaint']}): {stat}\n"
        
        overview_msg += f"\n>>> {self.message}"

        return Observation(
            patient_id=patient["patient_id"],
            age=patient["age"],
            chief_complaint=patient["chief_complaint"],
            visible_vitals=Vitals(**patient["visible_vitals"]),
            additional_info=self.revealed_infos[patient["patient_id"]],
            available_resources=AvailableResources(),
            turn_number=self.turn,
            task_level="medium",
            message=overview_msg
        )
