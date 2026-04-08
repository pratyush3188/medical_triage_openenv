from typing import Dict, Any, List
from ..models import Observation, Action, Vitals, AvailableResources

PATIENTS = [
    {
        "patient_id": "H1",
        "age": 68,
        "chief_complaint": "severe respiratory distress",
        "visible_vitals": {"heart_rate": 115, "blood_pressure": "140/90", "temperature": 37.5, "oxygen_saturation": 82},
        "additional_info": "Bilateral infiltrates on CXR. Requires intubation immediately.",
        "requires": "ventilator"
    },
    {
        "patient_id": "H2",
        "age": 71,
        "chief_complaint": "septic shock, collapsed",
        "visible_vitals": {"heart_rate": 130, "blood_pressure": "70/45", "temperature": 35.8, "oxygen_saturation": 90},
        "additional_info": "Lactate > 4 mmol/L. Vasopressors initiated.",
        "requires": "icu_bed"
    },
    {
        "patient_id": "H3",
        "age": 55,
        "chief_complaint": "high fever, feeling weak",
        "visible_vitals": {"heart_rate": 88, "blood_pressure": "115/75", "temperature": 40.2, "oxygen_saturation": 98},
        "additional_info": "Flu positive. Vitals stable despite high fever. Can monitor in general ward.",
        "requires": "general_ward"
    },
    {
        "patient_id": "H4",
        "age": 49,
        "chief_complaint": "chest pain",
        "visible_vitals": {"heart_rate": 82, "blood_pressure": "125/80", "temperature": 36.9, "oxygen_saturation": 99},
        "additional_info": "ECG normal, Troponin negative. Likely musculoskeletal or GERD.",
        "requires": "general_ward"
    },
    {
        "patient_id": "H5",
        "age": 62,
        "chief_complaint": "post-op complication, bleeding",
        "visible_vitals": {"heart_rate": 110, "blood_pressure": "95/60", "temperature": 37.1, "oxygen_saturation": 96},
        "additional_info": "Continuous oozing from surgical site. Needs close hemodynamic monitoring.",
        "requires": "icu_bed"
    },
    {
        "patient_id": "H6",
        "age": 33,
        "chief_complaint": "palpitations, panic",
        "visible_vitals": {"heart_rate": 105, "blood_pressure": "135/85", "temperature": 37.0, "oxygen_saturation": 99},
        "additional_info": "Anxiety attack mimicking cardiac event. Heart rhythm normal sinus.",
        "requires": "general_ward"
    },
    {
        "patient_id": "H7",
        "age": 42,
        "chief_complaint": "crushing chest pain, sweating",
        "visible_vitals": {"heart_rate": 110, "blood_pressure": "190/100", "temperature": 37.0, "oxygen_saturation": 96},
        "additional_info": "Active STEMI confirmed on ECG. High risk of decompensation.",
        "requires": "icu_bed"
    },
    {
        "patient_id": "H8",
        "age": 28,
        "chief_complaint": "asthma exacerbation",
        "visible_vitals": {"heart_rate": 115, "blood_pressure": "130/80", "temperature": 37.1, "oxygen_saturation": 88},
        "additional_info": "Responds well to nebulizer. Airway not fully compromised.",
        "requires": "general_ward"
    }
]

class HardTask:
    def __init__(self):
        self.patients = PATIENTS.copy()
        self.current_patient_index = 0
        self.turn = 0
        self.revealed_infos = {p["patient_id"]: None for p in self.patients}
        self.allocations = {}
        self.processed_patients = []
        self.message = "Crisis event! 8 patients, severe resource constraints."
        self.resources = AvailableResources(icu_beds=3, ventilators=1, general_ward_beds=5)

    def reset(self):
        self.current_patient_index = 0
        self.turn = 0
        self.revealed_infos = {p["patient_id"]: None for p in self.patients}
        self.allocations = {}
        self.processed_patients = []
        self.resources = AvailableResources(icu_beds=3, ventilators=1, general_ward_beds=5)
        self.message = "Crisis event! Allocate resources wisely. Ask follow-ups to differentiate true emergencies from misleading presentations."
        return self.get_obs()

    def _priority_to_resource(self, priority: str) -> str:
        mapping = {
            "CRITICAL": "icu_bed",
            "HIGH": "icu_bed",
            "MEDIUM": "general_ward",
            "LOW": "general_ward"
        }
        return mapping.get(priority.upper() if priority else "MEDIUM", "general_ward")

    def grade(self) -> float:
        GROUND_TRUTH = {
            "H1": "ventilator",
            "H2": "icu_bed",
            "H3": "general_ward",
            "H4": "general_ward",
            "H5": "icu_bed",
            "H6": "general_ward",
            "H7": "icu_bed",
            "H8": "general_ward"
        }
        
        if not self.allocations:
            return 0.0
        
        total = 0.0
        max_score = len(GROUND_TRUTH)
        
        for patient_id, correct_resource in GROUND_TRUTH.items():
            assigned = self.allocations.get(patient_id, None)
            if assigned is None:
                total += 0.0
            elif assigned == correct_resource:
                total += 1.0
            else:
                # Partial credit for close allocations
                # icu_bed vs ventilator = partial (both critical resources)
                # general_ward vs icu_bed = wrong
                if (correct_resource in ["icu_bed", "ventilator"] and 
                    assigned in ["icu_bed", "ventilator"]):
                    total += 0.5  # partial credit
                else:
                    total += 0.0
        
        final_score = round(total / max_score, 4)
        return max(0.0, min(1.0, final_score))

    def step(self, action: Action):
        self.turn += 1
        reward = 0.0
        done = False
        final_eval = 0.0

        if action.patient_id not in [p["patient_id"] for p in self.patients]:
            self.message = f"Invalid patient ID: {action.patient_id}"
            reward = -0.1
        else:
            if action.action_type == "ask_followup":
                patient = next(p for p in self.patients if p["patient_id"] == action.patient_id)
                self.revealed_infos[action.patient_id] = patient["additional_info"]
                reward = 0.2
                self.message = f"Follow-up for {action.patient_id}: {patient['additional_info']}"
            elif action.action_type in ["allocate_resource", "assign_priority"]:
                rsc = action.resource if action.action_type == "allocate_resource" else self._priority_to_resource(action.priority)
                if rsc == "icu_bed" and self.resources.icu_beds > 0:
                    self.resources.icu_beds -= 1
                    self.allocations[action.patient_id] = rsc
                    reward = 0.5 
                    self.message = f"Allocated icu_bed to {action.patient_id}."
                elif rsc == "ventilator" and self.resources.ventilators > 0:
                    self.resources.ventilators -= 1
                    self.allocations[action.patient_id] = rsc
                    reward = 0.5
                    self.message = f"Allocated ventilator to {action.patient_id}."
                elif rsc == "general_ward" and self.resources.general_ward_beds > 0:
                    self.resources.general_ward_beds -= 1
                    self.allocations[action.patient_id] = rsc
                    reward = 0.2
                    self.message = f"Allocated general_ward to {action.patient_id}."
                else:
                    self.allocations[action.patient_id] = rsc
                    self.message = f"Failed to allocate {rsc} to {action.patient_id}: Resource exhausted or invalid."
                    reward = -0.5
                
                if action.patient_id not in self.processed_patients:
                    self.processed_patients.append(action.patient_id)
                self.current_patient_index += 1
            else:
                reward = -0.1
                self.message = "Invalid action. Use ask_followup or allocate_resource."

        if self.current_patient_index >= len(self.patients):
            done = True
            final_eval = self.grade()
            reward += final_eval * 2.0
            self.message = f"All patients allocated. Grade: {final_eval:.2f}"
            
        elif self.turn >= 30: 
            done = True
            final_eval = self.grade()
            reward -= 2.0
            self.message = "Turn limit reached. Did not allocate resources to all patients."

        if not done:
            return self.get_obs(), reward, done, {}
        else:
            return self.get_final_obs(), reward, done, {"score": final_eval}
    
    def get_final_obs(self):
        idx = min(self.current_patient_index, len(self.patients) - 1)
        patient = self.patients[idx]
        overview_msg = f"Available - ICU: {self.resources.icu_beds}, Vent: {self.resources.ventilators}, Ward: {self.resources.general_ward_beds}\n"
        for p in self.patients:
            stat = self.allocations.get(p["patient_id"], "Pending")
            overview_msg += f"- {p['patient_id']} ({p['age']}y {p['chief_complaint']}): {stat}\n"
        overview_msg += f"\n>>> {self.message}"
        resources_dict = self.resources.model_dump() if hasattr(self.resources, "model_dump") else self.resources.dict()
        return Observation(
            patient_id=patient["patient_id"],
            age=patient["age"],
            chief_complaint=patient["chief_complaint"],
            visible_vitals=Vitals(**patient["visible_vitals"]),
            additional_info=self.revealed_infos[patient["patient_id"]],
            available_resources=AvailableResources(**resources_dict),
            turn_number=self.turn,
            task_level="hard",
            message=overview_msg
        )

    def get_obs(self):
        patient = self.patients[self.current_patient_index]
        overview_msg = f"Available - ICU: {self.resources.icu_beds}, Vent: {self.resources.ventilators}, Ward: {self.resources.general_ward_beds}\n"
        for p in self.patients:
            stat = self.allocations.get(p["patient_id"], "Pending")
            overview_msg += f"- {p['patient_id']} ({p['age']}y {p['chief_complaint']}): {stat}\n"
        overview_msg += f"\n>>> {self.message}"
        resources_dict = self.resources.model_dump() if hasattr(self.resources, "model_dump") else self.resources.dict()
        return Observation(
            patient_id=patient["patient_id"],
            age=patient["age"],
            chief_complaint=patient["chief_complaint"],
            visible_vitals=Vitals(**patient["visible_vitals"]),
            additional_info=self.revealed_infos[patient["patient_id"]],
            available_resources=AvailableResources(**resources_dict),
            turn_number=self.turn,
            task_level="hard",
            message=overview_msg
        )
