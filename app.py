"""
Gradio demo for Medical Triage Env.
OpenAI API key is read from the OPENAI_API_KEY environment variable.
"""
#temp
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
import sys, os, json, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.medical_triage import MedicalTriageEnv, TriageAction, PatientRanking
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
def root():
    return {"message": "API is running"}

from typing import Literal

class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"
    
class StepRequest(BaseModel):
    task_id: str
    rankings: list
_envs: dict[str, MedicalTriageEnv] = {}

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
        if v.heart_rate:         vitals_parts.append(f"HR {v.heart_rate}")
        if v.blood_pressure_sys: vitals_parts.append(f"BP {v.blood_pressure_sys}/{v.blood_pressure_dia}")
        if v.respiratory_rate:   vitals_parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation:  vitals_parts.append(f"SpO2 {v.oxygen_saturation}%")
        if v.temperature:        vitals_parts.append(f"Temp {v.temperature}°C")
        if v.gcs:                vitals_parts.append(f"GCS {v.gcs}")
        lines.append(f"  Vitals: {', '.join(vitals_parts)}")
        lines.append(f"  History: {p.history}\n")
    return "\n".join(lines)


def format_results(task_id: str, reward, info: dict, data: dict) -> str:
    rw = reward
    submitted_levels = [r["esi_level"] for r in data.get("rankings", [])]
    lines = [f"## Results for {task_id.upper()} Task\n"]

    if len(set(submitted_levels)) == 1:
        lines.append(
            f"> ⚠️ **All patients assigned ESI-{submitted_levels[0]}** — "
            f"Rank Order and Critical Catch scores are not meaningful.\n"
        )

    empty_rationale_ids = [
        r["patient_id"] for r in data.get("rankings", [])
        if not r.get("rationale", "").strip()
    ]
    if empty_rationale_ids:
        lines.append(
            f"> ⚠️ Empty rationale for: {', '.join(empty_rationale_ids)} — "
            f"Rationale Quality will be 0.000 for these patients.\n"
        )

    lines += [
        f"### 🏆 Total Score: **{rw.total:.3f} / 1.000**\n",
        "| Component | Score |",
        "|---|---|",
        f"| ESI Accuracy | {rw.esi_accuracy:.3f} |",
        f"| Rank Order (Kendall-τ) | {rw.rank_order:.3f} |",
        f"| Critical Catch | {rw.critical_catch:.3f} |",
        f"| Rationale Quality | {rw.rationale_quality:.3f} |",
        f"| Penalties | -{rw.penalties:.3f} |",
        "\n### Per-patient breakdown",
    ]

    bd = rw.breakdown.get("esi", {})
    for pid, info_d in bd.items():
        true_esi = info_d.get("true", "?")
        pred_esi = info_d.get("pred")
        pred_str = str(pred_esi) if pred_esi is not None else "❓ not submitted"
        icon     = "✅" if info_d.get("exact") else ("⚠️" if info_d.get("near") else "❌")
        lines.append(f"- **{pid}**: Assigned ESI-{pred_str} | True ESI-{true_esi} {icon}")

    if info.get("missing_patients"):
        lines.append(f"\n> ⚠️ No ranking received for: {info['missing_patients']}")

    if rw.breakdown.get("penalty_notes"):
        lines.append("\n### ⚠️ Safety Penalties")
        for note in rw.breakdown["penalty_notes"]:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _fresh_env(task_id: str) -> MedicalTriageEnv:
    env = MedicalTriageEnv(task_id=task_id)
    env.reset()
    _envs[task_id] = env
    return env


def load_task(task_id: str) -> tuple[str, str]:
    env = _fresh_env(task_id)
    obs = env.state()
    # re-fetch clean obs
    env2 = MedicalTriageEnv(task_id=task_id)
    obs  = env2.reset()
    _envs[task_id] = env2

    lines = [f"## {task_id.upper()} Task — {len(obs.patients)} Patients\n",
             f"> {obs.context}\n"]
    for p in obs.patients:
        v = p.vitals
        lines.append(f"### Patient {p.patient_id}: {p.age}yo {p.sex}")
        lines.append(f"**Chief Complaint:** {p.chief_complaint}")
        lines.append(f"**Symptoms:** {', '.join(p.symptoms)}")
        vitals_parts = []
        if v.heart_rate:         vitals_parts.append(f"HR {v.heart_rate} bpm")
        if v.blood_pressure_sys: vitals_parts.append(f"BP {v.blood_pressure_sys}/{v.blood_pressure_dia} mmHg")
        if v.respiratory_rate:   vitals_parts.append(f"RR {v.respiratory_rate}/min")
        if v.oxygen_saturation:  vitals_parts.append(f"SpO₂ {v.oxygen_saturation}%")
        if v.temperature:        vitals_parts.append(f"Temp {v.temperature}°C")
        if v.gcs:                vitals_parts.append(f"GCS {v.gcs}/15")
        lines.append(f"**Vitals:** {' | '.join(vitals_parts)}")
        lines.append(f"**History:** {p.history}\n")
    lines.append("---")
    lines.append("**Instructions:** Fill in `esi_level` (1–5) and `rationale`, or click **Run AI Baseline**.")
    lines.append(obs.action_required)

    template = json.dumps({
        "rankings": [
            {"patient_id": p.patient_id, "esi_level": None, "rationale": ""}
            for p in obs.patients
        ]
    }, indent=2)

    return "\n".join(lines), template


def submit_ranking(task_id: str, ranking_json: str) -> str:
    env = _envs.get(task_id)
    if env is None:
        return "⚠️ Load a task first by clicking **Load Task**."
    try:
        data = json.loads(ranking_json)
    except Exception as e:
        return f"❌ JSON parse error: {e}"

    null_ids = [r["patient_id"] for r in data.get("rankings", []) if r.get("esi_level") is None]
    if null_ids:
        return f"❌ Please assign an ESI level (1–5) for: {', '.join(null_ids)}"

    invalid = [
        r["patient_id"] for r in data.get("rankings", [])
        if not isinstance(r.get("esi_level"), int) or not (1 <= r["esi_level"] <= 5)
    ]
    if invalid:
        return f"❌ ESI level must be an integer 1–5. Invalid: {', '.join(invalid)}"

    env_state = env.state()
    expected_ids  = {p.patient_id for p in env_state.patients}
    submitted_ids = {r["patient_id"] for r in data.get("rankings", [])}
    missing = expected_ids - submitted_ids
    if missing:
        return f"❌ Missing rankings for: {', '.join(sorted(missing))}"

    try:
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in data["rankings"]],
            additional_notes=data.get("additional_notes"),
        )
    except Exception as e:
        return f"❌ Action construction error: {e}"

    try:
        _, reward, done, info = env.step(action)
    except RuntimeError:
        env = _fresh_env(task_id)
        _, reward, done, info = env.step(action)

    result = format_results(task_id, reward, info, data)
    _fresh_env(task_id)  # reload so user can resubmit
    return result


def run_ai_baseline(task_id: str, model: str, seed: int) -> tuple[str, str]:
    """Read API key from OPENAI_API_KEY env var — no UI input needed."""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI package not installed. Run: `pip install openai`", ""

    api_key = os.environ.get("", "").strip()
    if not api_key:
        return (
            "❌ `OPENAI_API_KEY` environment variable is not set.\n\n"
            "Set it before launching:\n```\n"
            "# Windows PowerShell\n$env:OPENAI_API_KEY='sk-...'\n\n"
            "# macOS / Linux\nexport OPENAI_API_KEY='sk-...'\n```",
            ""
        )

    env = _envs.get(task_id)
    if env is None:
        return "⚠️ Load a task first by clicking **Load Task**.", ""

    try:
        client  = OpenAI(api_key=api_key)
        fresh   = MedicalTriageEnv(task_id=task_id)
        obs     = fresh.reset()
        _envs[task_id] = fresh

        prompt   = build_patient_prompt(obs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            seed=int(seed),
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model disobeys
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data   = json.loads(raw)
            action = TriageAction(
                rankings=[PatientRanking(**r) for r in data["rankings"]],
                additional_notes=data.get("additional_notes"),
            )
        except Exception:
            data = {"rankings": [
                {"patient_id": p.patient_id, "esi_level": 3,
                 "rationale": "Parse error — fallback assignment"}
                for p in obs.patients
            ]}
            action = TriageAction(rankings=[PatientRanking(**r) for r in data["rankings"]])

        _, reward, done, info = fresh.step(action)
        _fresh_env(task_id)

        return format_results(task_id, reward, info, data), json.dumps(data, indent=2)

    except Exception as e:
        return f"❌ API error: {e}", ""

@app.post("/reset")
def reset_env(req: ResetRequest = ResetRequest()):
    try:
        task_id = req.task_id if req else "easy"

        env = MedicalTriageEnv(task_id=task_id)
        obs = env.reset()
        _envs[task_id] = env

        return {
            "patients": [p.patient_id for p in obs.patients],
            "context": obs.context,
            "action_required": obs.action_required
        }

    except Exception as e:
        return {"error": str(e)}
@app.post("/step")
def step_env(req: StepRequest):
    env = _envs.get(req.task_id)

    if not env:
        return {"error": "Environment not initialized. Call /reset first."}

    action = TriageAction(
        rankings=[PatientRanking(**r) for r in req.rankings]
    )

    _, reward, done, info = env.step(action)

    return {
        "reward": reward.total,
        "done": done,
        "info": info
    }
# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLE_EASY = json.dumps({
    "rankings": [
        {"patient_id": "E01", "esi_level": 1, "rationale": "62yo M, crushing chest pain, diaphoresis, BP 88/60, HR 112, SpO2 93%. Classic STEMI — immediate ESI-1."},
        {"patient_id": "E02", "esi_level": 3, "rationale": "34yo F, UTI symptoms, vitals stable. Needs urine culture + antibiotics — 2 resources = ESI-3."},
        {"patient_id": "E03", "esi_level": 5, "rationale": "27yo M, panic symptoms, HR 96, BP 128/82, SpO2 99%, GCS 15. Haemodynamically stable, no resources needed = ESI-5."},
    ]
}, indent=2)

with gr.Blocks(title="Medical Triage Prioritization — OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏥 Medical Triage Prioritization\n"
        "**OpenEnv Environment** | Dataset: 246,945 cases · 773 diseases · 377 symptoms\n\n"
        "**How to use:** Select difficulty → Load Task → Fill ESI levels manually **or** Run AI Baseline → Submit"
    )

    with gr.Row():
        task_selector = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Difficulty")
        load_btn      = gr.Button("🔄 Load Task", variant="primary")

    scenario_display = gr.Markdown("*Select a difficulty and click **Load Task** to begin.*")

    with gr.Accordion("✍️ Manual Ranking", open=True):
        gr.Markdown(
            "Set `esi_level` (1–5) per patient. Order **most → least urgent**. "
            "Write clinical `rationale` referencing vitals and symptoms for a better score."
        )
        ranking_input = gr.Code(value=EXAMPLE_EASY, language="json", label="Your Ranking JSON")
        submit_btn    = gr.Button("✅ Submit Ranking", variant="secondary")

    with gr.Accordion("🤖 AI Baseline", open=False):
        # Read OPENAI_API_KEY from env — show current status
        key_status = "✅ `OPENAI_API_KEY` detected in environment." \
            if os.environ.get("OPENAI_API_KEY") else \
            "⚠️ `OPENAI_API_KEY` not set. Set it before launching the app."
        gr.Markdown(
            f"{key_status}\n\n"
            "Uses the same system prompt and logic as `baseline.py`. "
            "The AI fills in the JSON editor — you can review and edit before submitting."
        )
        with gr.Row():
            model_input = gr.Textbox(value="gpt-4o", label="Model", scale=2)
            seed_input  = gr.Number(value=42, label="Seed", precision=0, scale=1)
        ai_btn = gr.Button("🚀 Run AI Baseline", variant="primary")

    result_display = gr.Markdown()

    load_btn.click(load_task,        inputs=task_selector,                         outputs=[scenario_display, ranking_input])
    submit_btn.click(submit_ranking, inputs=[task_selector, ranking_input],         outputs=result_display)
    ai_btn.click(run_ai_baseline,    inputs=[task_selector, model_input, seed_input], outputs=[result_display, ranking_input])

import uvicorn

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)