"""
Unit tests for MedicalTriageEnv — deterministic, reproducible.
Tests: reset(), step(), state(), reward components, penalty logic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from src.medical_triage import (
    MedicalTriageEnv, TriageAction, PatientRanking,
    TriageObservation, TriageReward, TriageState,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def perfect_action(env: MedicalTriageEnv) -> TriageAction:
    """Build a perfect triage action using ground-truth ESI."""
    obs = env.reset()
    patients = env.state().patients
    ranked = sorted(patients, key=lambda p: (p.true_esi or 9, p.patient_id))
    return TriageAction(rankings=[
        PatientRanking(
            patient_id=p.patient_id,
            esi_level=p.true_esi,
            rationale=(
                f"Patient has critical vitals: HR {p.vitals.heart_rate}, "
                f"BP {p.vitals.blood_pressure_sys}/{p.vitals.blood_pressure_dia}, "
                f"SpO2 {p.vitals.oxygen_saturation}%, GCS {p.vitals.gcs}. "
                f"Symptoms include {', '.join(p.symptoms[:3])}. "
                "Assigned based on ESI guidelines."
            )
        )
        for p in ranked
    ])


def worst_action(env: MedicalTriageEnv) -> TriageAction:
    """Worst possible action: reverse order, all ESI-5."""
    obs = env.reset()
    patients = list(reversed(env.state().patients))
    return TriageAction(rankings=[
        PatientRanking(patient_id=p.patient_id, esi_level=5, rationale="fine")
        for p in patients
    ])


# ── Reset tests ───────────────────────────────────────────────────────────────

def test_reset_returns_observation():
    for task in ["easy", "medium", "hard"]:
        env = MedicalTriageEnv(task_id=task)
        obs = env.reset()
        assert isinstance(obs, TriageObservation)
        assert obs.task_id == task
        assert obs.step == 0
        assert len(obs.patients) > 0


def test_reset_patient_counts():
    assert len(MedicalTriageEnv("easy").reset().patients)   == 3
    assert len(MedicalTriageEnv("medium").reset().patients) == 8
    assert len(MedicalTriageEnv("hard").reset().patients)   == 15


def test_reset_clears_state():
    env = MedicalTriageEnv("easy")
    action = perfect_action(env)
    env.step(action)
    # Reset should clear done flag
    env.reset()
    state = env.state()
    assert state.done is False
    assert state.total_reward == 0.0


def test_invalid_task_raises():
    with pytest.raises(ValueError):
        MedicalTriageEnv("nightmare")


# ── Step tests ────────────────────────────────────────────────────────────────

def test_step_returns_correct_types():
    env = MedicalTriageEnv("easy")
    action = perfect_action(env)
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, TriageObservation)
    assert isinstance(reward, TriageReward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_done_after_one_step():
    env = MedicalTriageEnv("easy")
    perfect_action(env)
    action = TriageAction(rankings=[
        PatientRanking(patient_id=p.patient_id, esi_level=p.true_esi,
                       rationale="test") for p in env.state().patients
    ])
    _, _, done, _ = env.step(action)
    assert done is True


def test_step_after_done_raises():
    env = MedicalTriageEnv("easy")
    action = perfect_action(env)
    env.step(action)
    with pytest.raises(RuntimeError):
        env.step(action)


# ── Reward tests ──────────────────────────────────────────────────────────────

def test_perfect_score_easy():
    env = MedicalTriageEnv("easy")
    action = perfect_action(env)
    _, reward, _, _ = env.step(action)
    assert reward.total >= 0.75, f"Perfect action should score ≥0.75, got {reward.total}"
    assert reward.critical_catch == 1.0
    assert reward.esi_accuracy >= 0.9


def test_perfect_score_medium():
    env = MedicalTriageEnv("medium")
    action = perfect_action(env)
    _, reward, _, _ = env.step(action)
    assert reward.total >= 0.70


def test_perfect_score_hard():
    env = MedicalTriageEnv("hard")
    action = perfect_action(env)
    _, reward, _, _ = env.step(action)
    assert reward.total >= 0.65


def test_worst_score_lower_than_perfect():
    for task in ["easy", "medium", "hard"]:
        env = MedicalTriageEnv(task)
        best = perfect_action(env)
        _, best_r, _, _ = env.step(best)

        env.reset()
        worst = worst_action(env)
        _, worst_r, _, _ = env.step(worst)

        assert best_r.total > worst_r.total, (
            f"{task}: perfect {best_r.total:.3f} should beat worst {worst_r.total:.3f}"
        )


def test_reward_in_range():
    for task in ["easy", "medium", "hard"]:
        env = MedicalTriageEnv(task)
        action = worst_action(env)
        _, reward, _, _ = env.step(action)
        assert 0.0 <= reward.total <= 1.0
        assert 0.0 <= reward.esi_accuracy <= 1.0
        assert 0.0 <= reward.rank_order <= 1.0
        assert 0.0 <= reward.critical_catch <= 1.0
        assert 0.0 <= reward.rationale_quality <= 1.0


def test_penalty_for_dangerous_downgrade():
    """ESI-1 patient assigned ESI-5 must incur penalty."""
    env = MedicalTriageEnv("easy")
    env.reset()
    patients = env.state().patients
    # E01 is ESI-1 (heart attack) — assign it ESI-5
    rankings = [
        PatientRanking(patient_id="E01", esi_level=5, rationale="fine"),
        PatientRanking(patient_id="E02", esi_level=3, rationale="uti stable"),
        PatientRanking(patient_id="E03", esi_level=5, rationale="panic"),
    ]
    _, reward, _, _ = env.step(TriageAction(rankings=rankings))
    assert reward.penalties > 0.0, "Dangerous downgrade should incur penalty"


def test_critical_catch_all_esi1_first():
    """If ESI-1 patients come first, critical_catch should be 1.0."""
    env = MedicalTriageEnv("easy")
    env.reset()
    # E01 is ESI-1 — put it first
    rankings = [
        PatientRanking(patient_id="E01", esi_level=1, rationale="immediate BP 88/60, HR 112, SpO2 93%"),
        PatientRanking(patient_id="E02", esi_level=3, rationale="uti"),
        PatientRanking(patient_id="E03", esi_level=5, rationale="panic"),
    ]
    _, reward, _, _ = env.step(TriageAction(rankings=rankings))
    assert reward.critical_catch == 1.0


# ── State tests ───────────────────────────────────────────────────────────────

def test_state_structure():
    env = MedicalTriageEnv("medium")
    env.reset()
    state = env.state()
    assert isinstance(state, TriageState)
    assert state.task_id == "medium"
    assert len(state.patients) == 8
    assert state.done is False


def test_state_after_step():
    env = MedicalTriageEnv("easy")
    action = perfect_action(env)
    env.step(action)
    state = env.state()
    assert state.done is True
    assert state.last_reward is not None
    assert state.agent_rankings is not None


# ── Observation structure tests ───────────────────────────────────────────────

def test_patients_have_required_fields():
    for task in ["easy", "medium", "hard"]:
        env = MedicalTriageEnv(task)
        obs = env.reset()
        for p in obs.patients:
            assert p.patient_id
            assert isinstance(p.age, int)
            assert p.sex in ("M", "F")
            assert p.chief_complaint
            assert len(p.symptoms) > 0
            assert p.vitals is not None
            assert p.history


def test_true_esi_hidden_from_observation():
    """true_esi and true_disease must not be serialised in observation."""
    env = MedicalTriageEnv("easy")
    obs = env.reset()
    obs_dict = obs.model_dump()
    for p_dict in obs_dict["patients"]:
        assert "true_esi" not in p_dict
        assert "true_disease" not in p_dict


def test_determinism():
    """Same task always produces same patients and reward for same action."""
    env1 = MedicalTriageEnv("hard")
    env2 = MedicalTriageEnv("hard")
    obs1 = env1.reset()
    obs2 = env2.reset()
    ids1 = [p.patient_id for p in obs1.patients]
    ids2 = [p.patient_id for p in obs2.patients]
    assert ids1 == ids2

    action1 = perfect_action(env1)
    action2 = perfect_action(env2)
    _, r1, _, _ = env1.step(action1)
    _, r2, _, _ = env2.step(action2)
    assert r1.total == r2.total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
