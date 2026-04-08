"""
Inference Script — Medical Triage Prioritization
===================================
MANDATORY environment variables:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.medical_triage import MedicalTriageEnv, TriageAction, PatientRanking

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")  or "gpt-4o"
BENCHMARK    = "medical-triage"
TASKS        = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

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


# ── Logging helpers (matching sample script exactly) ─────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Patient prompt builder ────────────────────────────────────────────────────
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


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_triage_action(client: OpenAI, obs) -> tuple[TriageAction, str]:
    """Call LLM and return a TriageAction + action string for logging."""
    user_prompt = build_patient_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model disobeys
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data   = json.loads(raw)
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in data["rankings"]],
            additional_notes=data.get("additional_notes"),
        )
        action_str = f"triage({len(data['rankings'])}patients)"
        return action, action_str

    except Exception as exc:
        # Fallback: assign ESI-3 to all patients
        obs_patients = obs.patients
        action = TriageAction(rankings=[
            PatientRanking(patient_id=p.patient_id, esi_level=3,
                           rationale="Fallback — parse error")
            for p in obs_patients
        ])
        return action, f"fallback_triage({len(obs_patients)}patients)"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASKS:
        rewards:     List[float] = []
        steps_taken: int         = 0
        score:       float       = 0.0
        success:     bool        = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        env = MedicalTriageEnv(task_id=task_id)

        try:
            obs  = env.reset()
            step = 1

            # Single-step environment — one triage ranking per episode
            action, action_str = get_triage_action(client, obs)

            try:
                _, reward_obj, done, info = env.step(action)
                reward    = round(reward_obj.total, 2)
                done_flag = done
                error     = None
            except Exception as e:
                reward    = 0.0
                done_flag = False
                error     = str(e).replace("\n", " ")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward,
                     done=done_flag, error=error)

            score   = min(max(reward, 0.0), 1.0)  # clamp to [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            log_step(step=1, action="null", reward=0.0, done=False, error=error_msg)
            rewards     = [0.0]
            steps_taken = 1

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
