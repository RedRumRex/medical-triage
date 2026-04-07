"""
Medical Triage Prioritization — Typed Pydantic Models
Implements full OpenEnv spec: Observation, Action, Reward
Dataset: Augmented Disease & Symptoms (246,945 records, 773 diseases, 377 symptoms)
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ESILevel(int, Enum):
    IMMEDIATE   = 1
    EMERGENT    = 2
    URGENT      = 3
    LESS_URGENT = 4
    NON_URGENT  = 5


ESI_LABEL = {
    1: "Immediate (ESI-1)",
    2: "Emergent (ESI-2)",
    3: "Urgent (ESI-3)",
    4: "Less Urgent (ESI-4)",
    5: "Non-Urgent (ESI-5)",
}

DISEASE_ESI: dict[str, int] = {
    "cardiac arrest":              1,
    "heart attack":                1,
    "stroke":                      1,
    "thoracic aortic aneurysm":    1,
    "abdominal aortic aneurysm":   1,
    "sepsis":                      1,
    "pulmonary embolism":          1,
    "ectopic pregnancy":           1,
    "meningitis":                  1,
    "appendicitis":                2,
    "acute pancreatitis":          2,
    "pneumonia":                   2,
    "anemia":                      2,
    "deep vein thrombosis":        2,
    "asthma":                      3,
    "acute bronchitis":            3,
    "urinary tract infection":     3,
    "migraine":                    3,
    "gout":                        4,
    "infectious gastroenteritis":  4,
    "tension headache":            4,
    "panic disorder":              5,
    "anxiety":                     5,
}


class Vitals(BaseModel):
    heart_rate:         Optional[int]   = Field(None, description="bpm")
    blood_pressure_sys: Optional[int]   = Field(None, description="mmHg systolic")
    blood_pressure_dia: Optional[int]   = Field(None, description="mmHg diastolic")
    respiratory_rate:   Optional[int]   = Field(None, description="breaths/min")
    oxygen_saturation:  Optional[float] = Field(None, description="SpO2 %")
    temperature:        Optional[float] = Field(None, description="Celsius")
    gcs:                Optional[int]   = Field(None, description="Glasgow Coma Scale 3-15")


class Patient(BaseModel):
    patient_id:      str
    age:             int
    sex:             str
    chief_complaint: str
    symptoms:        list[str]
    vitals:          Vitals
    history:         str
    true_disease:    Optional[str] = Field(None, exclude=True)
    true_esi:        Optional[int] = Field(None, exclude=True)


class TriageObservation(BaseModel):
    task_id:          str
    step:             int
    patients:         list[Patient]
    context:          str
    action_required:  str
    previous_actions: list[dict[str, Any]] = Field(default_factory=list)


class PatientRanking(BaseModel):
    patient_id: str
    esi_level:  int = Field(..., ge=1, le=5)
    rationale:  str


class TriageAction(BaseModel):
    rankings:         list[PatientRanking]
    additional_notes: Optional[str] = None


class TriageReward(BaseModel):
    total:             float = Field(..., ge=0.0, le=1.0)
    esi_accuracy:      float = Field(..., ge=0.0, le=1.0)
    rank_order:        float = Field(..., ge=0.0, le=1.0)
    critical_catch:    float = Field(..., ge=0.0, le=1.0)
    rationale_quality: float = Field(..., ge=0.0, le=1.0)
    penalties:         float = Field(default=0.0)
    breakdown:         dict[str, Any] = Field(default_factory=dict)


class TriageState(BaseModel):
    task_id:        str
    step:           int
    done:           bool
    patients:       list[Patient]
    agent_rankings: Optional[list[PatientRanking]]
    last_reward:    Optional[TriageReward]
    total_reward:   float
