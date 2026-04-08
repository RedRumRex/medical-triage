#!/usr/bin/env python3
"""
Baseline inference script — uses OpenAI API client against MedicalTriageEnv.
Reads: OPENAI_API_KEY from environment.
Produces reproducible baseline scores across all 3 tasks.

Usage:
  export OPENAI_API_KEY=sk-...
  python baseline.py [--model gpt-4o] [--seed 42]
"""
from __future__ import annotations
import os, sys, json, argparse, textwrap

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.medical_triage import MedicalTriageEnv, TriageAction, PatientRanking


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
    lines = [f"TRIAGE SCENARIO: {obs.context}\n",
             f"ACTION REQUIRED: {obs.action_required}\n",
             f"PATIENTS ({len(obs.patients)} total):\n"]
    for p in obs.patients:
        v = p.vitals
        lines.append(f"Patient {p.patient_id}: {p.age}yo {p.sex}")
        lines.append(f"  Chief complaint: {p.chief_complaint}")
        lines.append(f"  Symptoms: {', '.join(p.symptoms)}")
        vitals_parts = []
        if v.heart_rate:     vitals_parts.append(f"HR {v.heart_rate}")
        if v.blood_pressure_sys: vitals_parts.append(f"BP {v.blood_pressure_sys}/{v.blood_pressure_dia}")
        if v.respiratory_rate:   vitals_parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation:  vitals_parts.append(f"SpO2 {v.oxygen_saturation}%")
        if v.temperature:        vitals_parts.append(f"Temp {v.temperature}°C")
        if v.gcs:                vitals_parts.append(f"GCS {v.gcs}")
        lines.append(f"  Vitals: {', '.join(vitals_parts)}")
        lines.append(f"  History: {p.history}\n")
    return "\n".join(lines)


def run_task(client: OpenAI, model: str, task_id: str, seed: int) -> dict:
    env = MedicalTriageEnv(task_id=task_id)
    obs = env.reset()
    prompt = build_patient_prompt(obs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        seed=seed,
    )
    raw = response.choices[0].message.content.strip()

    # Parse response
    try:
        data = json.loads(raw)
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in data["rankings"]],
            additional_notes=data.get("additional_notes"),
        )
    except Exception as e:
        print(f"  [PARSE ERROR] {e}\n  Raw response: {raw[:300]}")
        # Fallback: random ESI-3 for all
        action = TriageAction(rankings=[
            PatientRanking(patient_id=p.patient_id, esi_level=3,
                           rationale="Parse error — fallback assignment")
            for p in obs.patients
        ])

    _, reward, done, info = env.step(action)
    print(env.render())
    return {
        "task_id":         task_id,
        "total":           reward.total,
        "esi_accuracy":    reward.esi_accuracy,
        "rank_order":      reward.rank_order,
        "critical_catch":  reward.critical_catch,
        "rationale":       reward.rationale_quality,
        "penalties":       reward.penalties,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("API_BASE_URL")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = []

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}")
        print(f"  Running task: {task_id.upper()}")
        print(f"{'='*60}")
        result = run_task(client, args.model, task_id, args.seed)
        results.append(result)
        print(f"\n  Score: {result['total']:.3f}")

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Total':>7} {'ESI':>7} {'Rank':>7} {'Crit':>7} {'Pen':>7}")
    print(f"  {'-'*47}")
    for r in results:
        print(f"  {r['task_id']:<10} {r['total']:>7.3f} {r['esi_accuracy']:>7.3f} "
              f"{r['rank_order']:>7.3f} {r['critical_catch']:>7.3f} {r['penalties']:>7.3f}")
    avg = sum(r["total"] for r in results) / len(results)
    print(f"  {'AVERAGE':<10} {avg:>7.3f}")
    print(f"\n  Model: {args.model}  |  Seed: {args.seed}")


if __name__ == "__main__":
    main()
