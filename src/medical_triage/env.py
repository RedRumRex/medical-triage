"""
MedicalTriageEnv — OpenEnv-compliant environment.
Implements: reset(), step(), state()
Three tasks: easy (3 patients), medium (8 patients), hard (15 patients)
"""
from __future__ import annotations
import json
import copy
from typing import Any

from .models import (
    Patient, TriageObservation, TriageAction, TriageReward,
    TriageState, PatientRanking,
)
from .grader import grade
from .patients import EASY_PATIENTS, MEDIUM_PATIENTS, HARD_PATIENTS

TASK_REGISTRY: dict[str, list[Patient]] = {
    "easy":   EASY_PATIENTS,
    "medium": MEDIUM_PATIENTS,
    "hard":   HARD_PATIENTS,
}

TASK_CONTEXT = {
    "easy": (
        "Community ED, suburban hospital. Moderate volume. "
        "3 patients waiting. Assign ESI 1-5 and rank by urgency."
    ),
    "medium": (
        "Urban Level-II trauma centre. Moderate surge. "
        "8 patients arrived simultaneously. Several have overlapping presentations. "
        "Careful prioritisation required."
    ),
    "hard": (
        "Level-I trauma centre. Mass-casualty drill scenario. "
        "15 patients across a spectrum of acuity — several with MISLEADING VITALS "
        "or atypical presentations. Time-critical decisions required. "
        "Some patients may look stable but have immediately life-threatening conditions."
    ),
}


class MedicalTriageEnv:
    """OpenEnv-compliant Medical Triage environment."""

    VALID_TASKS = list(TASK_REGISTRY.keys())

    def __init__(self, task_id: str = "easy"):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id = task_id
        self._patients: list[Patient] = []
        self._step = 0
        self._done = False
        self._last_reward: TriageReward | None = None
        self._total_reward = 0.0
        self._agent_rankings: list[PatientRanking] | None = None

    # ── OpenEnv interface ────────────────────────────────────────────────
    def reset(self) -> TriageObservation:
        """Reset environment; return initial observation."""
        self._patients = copy.deepcopy(TASK_REGISTRY[self.task_id])
        self._step = 0
        self._done = False
        self._last_reward = None
        self._total_reward = 0.0
        self._agent_rankings = None
        return self._build_observation()

    def step(self, action: TriageAction) -> tuple[TriageObservation, TriageReward, bool, dict[str, Any]]:
        """
        Execute one triage step.
        Returns (observation, reward, done, info).
        The task is single-step: one triage ranking per episode.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Validate all patients are ranked
        submitted_ids = {r.patient_id for r in action.rankings}
        expected_ids  = {p.patient_id for p in self._patients}
        missing = expected_ids - submitted_ids
        extra   = submitted_ids - expected_ids

        info: dict[str, Any] = {
            "missing_patients": list(missing),
            "extra_patients":   list(extra),
            "task_id": self.task_id,
        }

        if missing:
            # Partial ranking — penalise but still grade
            info["warning"] = f"Missing rankings for: {missing}"

        reward = grade(self._patients, action.rankings)
        self._last_reward   = reward
        self._total_reward  += reward.total
        self._agent_rankings = action.rankings
        self._step          += 1
        self._done           = True

        obs = self._build_observation(action=action)
        return obs, reward, self._done, info

    def state(self) -> TriageState:
        """Return full current state (for logging / debugging)."""
        return TriageState(
            task_id=self.task_id,
            step=self._step,
            done=self._done,
            patients=self._patients,
            agent_rankings=self._agent_rankings,
            last_reward=self._last_reward,
            total_reward=round(self._total_reward, 4),
        )

    # ── Helpers ──────────────────────────────────────────────────────────
    def _build_observation(self, action: TriageAction | None = None) -> TriageObservation:
        prev = []
        if action:
            prev = [r.model_dump() for r in action.rankings]
        return TriageObservation(
            task_id=self.task_id,
            step=self._step,
            patients=self._patients,
            context=TASK_CONTEXT[self.task_id],
            action_required=(
                "Assign each patient an ESI level (1=Immediate, 5=Non-Urgent) "
                "and return them ranked from most to least urgent. "
                "Provide clinical rationale for each ranking decision."
            ),
            previous_actions=prev,
        )

    def render(self) -> str:
        """Human-readable summary of current state."""
        lines = [f"\n{'='*60}", f"  Medical Triage — {self.task_id.upper()} task", f"{'='*60}"]
        for p in self._patients:
            lines.append(
                f"  [{p.patient_id}] {p.age}yo {p.sex} | {p.chief_complaint[:60]}"
            )
            if p.vitals.heart_rate:
                lines.append(
                    f"       HR {p.vitals.heart_rate} | BP {p.vitals.blood_pressure_sys}/"
                    f"{p.vitals.blood_pressure_dia} | SpO2 {p.vitals.oxygen_saturation}% "
                    f"| GCS {p.vitals.gcs}"
                )
        if self._agent_rankings:
            lines.append(f"\n  Agent Rankings:")
            for r in self._agent_rankings:
                p = next((p for p in self._patients if p.patient_id == r.patient_id), None)
                true_esi = p.true_esi if p else "?"
                lines.append(f"    {r.patient_id} → ESI-{r.esi_level} (true ESI-{true_esi})")
        if self._last_reward:
            rw = self._last_reward
            lines.append(f"\n  Reward: {rw.total:.3f}")
            lines.append(f"    ESI accuracy={rw.esi_accuracy:.3f}  rank_order={rw.rank_order:.3f}")
            lines.append(f"    critical_catch={rw.critical_catch:.3f}  rationale={rw.rationale_quality:.3f}")
            if rw.penalties > 0:
                lines.append(f"    ⚠ penalties={rw.penalties:.3f}")
        return "\n".join(lines)
