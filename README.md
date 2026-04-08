---
title: Medical Triage Prioritization
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
tags: [openenv, healthcare, triage, ranking, esi, medical, real-world]
---

# 🏥 Medical Triage Prioritization — OpenEnv Environment

A real-world emergency department triage environment where agents assign
**Emergency Severity Index (ESI) levels** to patients and rank them by urgency.
Grounded in the **Augmented Disease & Symptoms dataset** (246,945 records,
773 diseases, 377 symptoms).

---

## Motivation

Medical triage is one of the highest-stakes real-world ranking tasks:
wrong prioritisation directly causes preventable deaths. This environment
evaluates whether agents can reason clinically under pressure — correctly
identifying life-threatening conditions even when vitals appear deceptively
stable (e.g., an abdominal aortic aneurysm with normal BP, or ectopic
pregnancy with seemingly mild pain).

Unlike synthetic benchmarks, every patient case is constructed from
empirically-derived symptom prevalence rates from real clinical data.

---

## Dataset

| Attribute | Value |
|---|---|
| Source | Augmented Disease & Symptoms Dataset |
| Records | 246,945 |
| Unique diseases | 773 |
| Symptom features | 377 |
| Usage | Symptom frequency distributions used to construct authentic patient profiles |

---

## ESI Levels (Ground Truth)

| Level | Name | Description |
|---|---|---|
| ESI-1 | Immediate | Life-threatening; requires immediate intervention |
| ESI-2 | Emergent | High risk; must not wait |
| ESI-3 | Urgent | Stable but needs ≥2 resources |
| ESI-4 | Less Urgent | Stable; needs 1 resource |
| ESI-5 | Non-Urgent | Stable; no resources needed |

---

## Environment Interface

```python
from medical_triage import MedicalTriageEnv, TriageAction, PatientRanking

env = MedicalTriageEnv(task_id="easy")  # "easy" | "medium" | "hard"

# Reset → initial observation
obs = env.reset()

# Build action
action = TriageAction(rankings=[
    PatientRanking(
        patient_id="E01",
        esi_level=1,
        rationale="62yo M with crushing chest pain, BP 88/60, HR 112 — STEMI."
    ),
    ...
])

# Step → (observation, reward, done, info)
obs, reward, done, info = env.step(action)

# State (for logging)
state = env.state()

print(reward.total)          # 0.0 – 1.0
print(reward.esi_accuracy)   # ESI exact-match rate
print(reward.critical_catch) # Did ESI-1/2 get top slots?
```

---

## Observation Space

```
TriageObservation
├── task_id: str
├── step: int
├── context: str               # scenario narrative
├── action_required: str       # instruction
├── previous_actions: list
└── patients: list[Patient]
    ├── patient_id: str
    ├── age: int
    ├── sex: "M" | "F"
    ├── chief_complaint: str
    ├── symptoms: list[str]    # from 377-feature dataset
    ├── history: str
    └── vitals: Vitals
        ├── heart_rate: int | None        (bpm)
        ├── blood_pressure_sys: int | None (mmHg)
        ├── blood_pressure_dia: int | None (mmHg)
        ├── respiratory_rate: int | None  (breaths/min)
        ├── oxygen_saturation: float | None (SpO2 %)
        ├── temperature: float | None     (°C)
        └── gcs: int | None              (Glasgow Coma Scale 3-15)
```

Note: `true_esi` and `true_disease` are excluded from serialisation (grader-only).

## Action Space

```
TriageAction
├── rankings: list[PatientRanking]   # ordered most→least urgent
│   ├── patient_id: str
│   ├── esi_level: int               # 1–5
│   └── rationale: str              # clinical reasoning
└── additional_notes: str | None
```

---

## Reward Function

The reward provides **partial progress signal** across four components:

| Component | Weight | Description |
|---|---|---|
| `esi_accuracy` | 30% | Exact + partial credit for ±1 ESI error |
| `rank_order` | 25% | Normalised Kendall-τ correlation with ground truth |
| `critical_catch` | 25% | Fraction of ESI-1/2 patients in top-n slots |
| `rationale_quality` | 20% | Heuristic: vitals + clinical keyword usage |
| **Penalties** | up to −0.5 | Dangerous downgrade (ESI-1 → ESI-4/5 = −0.25 each) |

**Final reward = clip(weighted_sum − penalties, 0.0, 1.0)**

---

## Tasks

### Easy (3 patients — textbook presentations)
- Heart attack (ESI-1): crushing chest pain, diaphoresis, hypotension
- Urinary tract infection (ESI-3): dysuria, frequency, stable vitals
- Panic disorder (ESI-5): palpitations, anxiety, normal vitals

**Expected agent score: 0.70 – 1.00**

### Medium (8 patients — overlapping severity)
- Stroke (ESI-1), Meningitis (ESI-1), Pulmonary embolism (ESI-1)
- Appendicitis (ESI-2), Pneumonia (ESI-2)
- Asthma (ESI-3), Migraine (ESI-3)
- Gastroenteritis (ESI-4)

**Expected agent score: 0.50 – 0.85**

### Hard (15 patients — misleading vitals, time-critical)
- AAA with "normal" BP (ESI-1 hidden in plain sight)
- Ectopic pregnancy with mild-seeming pain (ESI-1)
- Sepsis with confusion (ESI-1)
- Cardiac arrest with bradycardia (ESI-1)
- Meningitis "thunderclap" (ESI-1)
- Thoracic aortic aneurysm (ESI-1)
- Plus pancreatitis, anemia, PE, pneumonia, asthma, UTI, migraine, gastroenteritis, panic

**Expected agent score: 0.30 – 0.75**

---

## Baseline Scores (gpt-4o, seed=42)

| Task | Total | ESI Acc | Rank Order | Crit Catch | Penalties |
|---|---|---|---|---|---|
| Easy | 0.848 | 1.000 | 1.000 | 1.000 | 0.000 |
| Medium | 0.712 | 0.688 | 0.786 | 0.875 | 0.000 |
| Hard | 0.584 | 0.567 | 0.619 | 0.667 | 0.100 |
| **Average** | **0.715** | | | | |

---

## Setup & Usage

### Local

```bash
git clone <repo>
cd medical-triage-env
pip install -r requirements.txt
PYTHONPATH=src python baseline.py --model gpt-4o
```

### Docker

```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=$OPENAI_API_KEY medical-triage-env
```

### Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Project Structure

```
medical-triage-env/
├── src/medical_triage/
│   ├── __init__.py        # Public API
│   ├── models.py          # Pydantic typed models (Observation/Action/Reward)
│   ├── env.py             # MedicalTriageEnv: reset()/step()/state()
│   ├── grader.py          # Deterministic reward function
│   └── patients.py        # 26 patient cases from dataset
├── tests/
│   └── test_env.py        # 20 unit tests
├── app.py                 # Gradio HF Spaces demo
├── baseline.py            # OpenAI baseline inference script
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## Clinical Grounding

Patient symptom profiles are derived from the Augmented Disease & Symptoms
dataset. For example, the **heart attack** patient presents with:
- `burning chest pain` (65% prevalence), `fainting` (54%), `arm pain` (51%)
- Vitals: HR 112, BP 88/60 mmHg, SpO₂ 93% — authentic shock presentation

The **hard** task's AAA patient (H01) is intentionally misleading:
- HR 74 and BP 148/92 appear stable → but sudden-onset back+abdominal pain
  with pulsatile mass history = ESI-1. Agents that rely only on vitals fail here.

---

## License

Apache 2.0
