from pydantic import BaseModel, Field
from typing import Optional, Dict

class Vitals(BaseModel):
    heart_rate: int
    blood_pressure: str
    temperature: float
    oxygen_saturation: int

class AvailableResources(BaseModel):
    icu_beds: int = 0
    ventilators: int = 0
    general_ward_beds: int = 0

class Observation(BaseModel):
    patient_id: str
    age: int
    chief_complaint: str
    visible_vitals: Vitals
    additional_info: Optional[str] = None
    available_resources: AvailableResources
    turn_number: int
    task_level: str
    message: str

class Action(BaseModel):
    action_type: str = Field(description="'ask_followup', 'assign_priority', or 'allocate_resource'")
    priority: Optional[str] = Field(None, description="'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'")
    resource: Optional[str] = Field(None, description="'icu_bed', 'ventilator', or 'general_ward'")
    patient_id: str
    notes: Optional[str] = None
