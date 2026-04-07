#!/usr/bin/env python3
"""
inference.py — Competition-compliant inference script for Medical Triage Prioritization.

Required environment variables:
  API_BASE_URL  : The API endpoint for the LLM (OpenAI-compatible)
  MODEL_NAME    : The model identifier to use for inference
  HF_TOKEN      : Your Hugging Face / API key

Structured stdout logs follow [START], [STEP], [END] format strictly.
Runtime target: < 20 minutes | Machine: vcpu=2, memory=8gb
"""
from __future__ import annotations
import os
import sys
import json
import textwrap
import time

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openai import OpenAI
from src.medical_triage import MedicalTriageEnv, TriageAction, PatientRanking

# ── Environment variables (mandatory per competition rules) ───────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME   = os.environ.get("MODEL_NAME",   "").strip()
HF_TOKEN     = os.environ.get("HF_TOKEN",     "").strip()

if not API_BASE_URL:
    print(json.dumps({"error": "API_BASE_URL environment variable is not set"}))
    sys.exit(1)
if not MODEL_NAME:
    print(json.dumps({"error": "MODEL_NAME environment variable is not set"}))
    sys.exit(1)
if not HF_TOKEN:
    print(json.dumps({"error": "HF_TOKEN environment variable is not set"}))
    sys.exit(1)

# ── OpenAI Client (using competition-required variables) ──────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency department triage nurse with 15+ years of experience.
Your role is to assess patients and assign ESI (Emergency Severity Index) levels:
  ESI-1: Immediate — requires immediate life-saving intervention
  ESI-2: Emergent — high-risk situation; should not wait
  ESI-3: Urgent — stable but needs ≥2 resources
  ESI-4: Less Urgent — stable, needs 1 resource
  ESI-5: Non-Urgent — stable, no resources needed

You will receive a list of patients. For each patient:
1. Assign an ESI level (1-5)
2. Provide clinical rationale referencing specific symptoms and vitals
3. Rank all patients from most urgent (ESI-1) to least urgent (ESI-5)

IMPORTANT: Prioritise life-threatening conditions even when vitals appear deceptively stable.
Watch for: aortic aneurysm (back pain + BP differential), ectopic pregnancy (missed period +
pelvic pain + haemodynamic instability), meningitis (fever + neck stiffness + photophobia),
sepsis (infection + altered mental status + tachycardia + hypotension).

Respond ONLY with valid JSON matching this schema:
{
  "rankings": [
    {
      "patient_id": "...",
      "esi_level": <int 1-5>,
      "rationale": "<clinical reasoning mentioning vitals and key symptoms>"
    }
  ],
  "additional_notes": "..."
}
Most urgent patient FIRST. Include ALL patients. No markdown fences.
""").strip()


def build_patient_prompt(obs) -> str:
    lines = [
        f"TRIAGE SCENARIO: {obs.context}\n",
        f"ACTION REQUIRED: {obs.action_required}\n",
        f"PATIENTS ({len(obs.patients)} total):\n",
    ]
    for p in obs.patients:
        v = p.vitals
        lines.append(f"Patient {p.patient_id}: {p.age}yo {p.sex}")
        lines.append(f"  Chief complaint: {p.chief_complaint}")
        lines.append(f"  Symptoms: {', '.join(p.symptoms)}")
        vitals_parts = []
        if v.heart_rate:         vitals_parts.append(f"HR {v.heart_rate}")
        if v.blood_pressure_sys: vitals_parts.append(f"BP {v.blood_pressure_sys}/{v.blood_pressure_dia}")
        if v.respiratory_rate:   vitals_parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation:  vitals_parts.append(f"SpO2 {v.oxygen_saturation}%")
        if v.temperature:        vitals_parts.append(f"Temp {v.temperature}°C")
        if v.gcs:                vitals_parts.append(f"GCS {v.gcs}")
        lines.append(f"  Vitals: {', '.join(vitals_parts)}")
        lines.append(f"  History: {p.history}\n")
    return "\n".join(lines)


def run_task(task_id: str) -> dict:
    env  = MedicalTriageEnv(task_id=task_id)
    obs  = env.reset()

    # ── [STEP] log ────────────────────────────────────────────────────────
    print(json.dumps({
        "type":    "[STEP]",
        "task_id": task_id,
        "step":    0,
        "status":  "calling_llm",
        "model":   MODEL_NAME,
        "n_patients": len(obs.patients),
    }), flush=True)

    prompt   = build_patient_prompt(obs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model disobeys
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # ── Parse response ────────────────────────────────────────────────────
    try:
        data   = json.loads(raw)
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in data["rankings"]],
            additional_notes=data.get("additional_notes"),
        )
    except Exception as e:
        print(json.dumps({
            "type":    "[STEP]",
            "task_id": task_id,
            "step":    0,
            "status":  "parse_error",
            "error":   str(e),
        }), flush=True)
        data   = {"rankings": [
            {"patient_id": p.patient_id, "esi_level": 3,
             "rationale": "Parse error — fallback assignment"}
            for p in obs.patients
        ]}
        action = TriageAction(rankings=[PatientRanking(**r) for r in data["rankings"]])

    _, reward, done, info = env.step(action)

    # ── [STEP] result log ─────────────────────────────────────────────────
    print(json.dumps({
        "type":             "[STEP]",
        "task_id":          task_id,
        "step":             1,
        "status":           "scored",
        "total":            round(reward.total, 4),
        "esi_accuracy":     round(reward.esi_accuracy, 4),
        "rank_order":       round(reward.rank_order, 4),
        "critical_catch":   round(reward.critical_catch, 4),
        "rationale_quality":round(reward.rationale_quality, 4),
        "penalties":        round(reward.penalties, 4),
    }), flush=True)

    return {
        "task_id":          task_id,
        "total":            reward.total,
        "esi_accuracy":     reward.esi_accuracy,
        "rank_order":       reward.rank_order,
        "critical_catch":   reward.critical_catch,
        "rationale_quality":reward.rationale_quality,
        "penalties":        reward.penalties,
        "reward":           reward.total,   # alias for grader compatibility
    }


def main():
    start_time = time.time()
    tasks      = ["easy", "medium", "hard"]

    # ── [START] log ───────────────────────────────────────────────────────
    print(json.dumps({
        "type":      "[START]",
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "tasks":     tasks,
        "timestamp": start_time,
    }), flush=True)

    results = []
    for task_id in tasks:
        result = run_task(task_id)
        results.append(result)

    avg_score = sum(r["total"] for r in results) / len(results)

    # ── [END] log ─────────────────────────────────────────────────────────
    print(json.dumps({
        "type":          "[END]",
        "tasks_completed": len(results),
        "results":       [
            {
                "task_id":           r["task_id"],
                "reward":            round(r["total"], 4),
                "esi_accuracy":      round(r["esi_accuracy"], 4),
                "rank_order":        round(r["rank_order"], 4),
                "critical_catch":    round(r["critical_catch"], 4),
                "rationale_quality": round(r["rationale_quality"], 4),
                "penalties":         round(r["penalties"], 4),
            }
            for r in results
        ],
        "average_reward":  round(avg_score, 4),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }), flush=True)


if __name__ == "__main__":
    main()