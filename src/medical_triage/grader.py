"""
Reward / grader logic — deterministic, reproducible.
Implements a meaningful reward function with partial-progress signals:
  - ESI accuracy (exact match)
  - Rank-order quality (Kendall-τ)
  - Critical catch (ESI-1/2 patients get top slots)
  - Rationale quality (heuristic keyword check)
  - Penalties for dangerous downgrade of ESI-1 patients
"""
from __future__ import annotations
import math
from .models import Patient, PatientRanking, TriageReward, DISEASE_ESI

CRITICAL_KEYWORDS = [
    "life", "immediate", "emergent", "critical", "unstable", "hypotension",
    "hypoxia", "tachycardia", "bradycardia", "altered", "consciousness",
    "sepsis", "stroke", "infarct", "arrest", "aneurysm", "embolism",
    "ectopic", "meningitis", "confusion", "gcs", "oxygen", "spo2",
    "blood pressure", "respiratory", "systolic", "hemodynamic",
]


def _kendall_tau(pred_order: list[str], true_order: list[str]) -> float:
    """Normalised Kendall-τ ∈ [0, 1] (1 = perfect concordance)."""
    n = len(pred_order)
    if n <= 1:
        return 1.0
    rank_map = {pid: i for i, pid in enumerate(true_order)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi = pred_order[i]
            pj = pred_order[j]
            if pi not in rank_map or pj not in rank_map:
                continue
            sign_pred = -1  # i before j means i is more urgent (lower index = higher urgency)
            sign_true = -1 if rank_map[pi] < rank_map[pj] else 1
            if sign_pred == sign_true:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    tau = (concordant - discordant) / total
    return (tau + 1.0) / 2.0  # normalise to [0, 1]


def grade(patients: list[Patient], rankings: list[PatientRanking]) -> TriageReward:
    true_map = {p.patient_id: p for p in patients}
    rank_map = {r.patient_id: r for r in rankings}

    # Ground-truth ordering: sort by true_esi ascending (lower = more urgent)
    true_ordered = sorted(patients, key=lambda p: (p.true_esi or 9, p.patient_id))
    true_order_ids = [p.patient_id for p in true_ordered]

    # Agent ordering: as submitted
    agent_order_ids = [r.patient_id for r in rankings]

    # ── ESI accuracy ──────────────────────────────────────────────────────
    exact_matches = 0
    near_matches  = 0  # off by 1 ESI level
    total = len(patients)
    esi_breakdown: dict[str, dict] = {}

    for p in patients:
        r = rank_map.get(p.patient_id)
        if r is None:
            esi_breakdown[p.patient_id] = {"true": p.true_esi, "pred": None, "match": False}
            continue
        diff = abs((r.esi_level or 0) - (p.true_esi or 0))
        exact = diff == 0
        near  = diff <= 1
        if exact:
            exact_matches += 1
        if near:
            near_matches += 1
        esi_breakdown[p.patient_id] = {
            "true": p.true_esi,
            "pred": r.esi_level,
            "exact": exact,
            "near": near,
        }

    esi_accuracy = (exact_matches + 0.5 * (near_matches - exact_matches)) / max(total, 1)

    # ── Rank order ────────────────────────────────────────────────────────
    rank_order = _kendall_tau(agent_order_ids, true_order_ids)

    # ── Critical catch ────────────────────────────────────────────────────
    critical_patients = [p for p in patients if p.true_esi in (1, 2)]
    n_crit = len(critical_patients)
    if n_crit == 0:
        critical_catch = 1.0
    else:
        crit_ids = {p.patient_id for p in critical_patients}
        # Top-n slots in agent ordering
        top_n = agent_order_ids[:n_crit]
        caught = len(crit_ids & set(top_n))
        critical_catch = caught / n_crit

    # ── Rationale quality ─────────────────────────────────────────────────
    rationale_scores = []
    for r in rankings:
        text = r.rationale.lower()
        hits = sum(1 for kw in CRITICAL_KEYWORDS if kw in text)
        # Reward having symptoms mentioned + clinical reasoning
        has_vitals_ref = any(v in text for v in ["bp", "heart rate", "spo2", "gcs", "temp", "rr"])
        score = min(hits / 3.0, 1.0) * 0.7 + (0.3 if has_vitals_ref else 0.0)
        rationale_scores.append(score)
    rationale_quality = sum(rationale_scores) / max(len(rationale_scores), 1)

    # ── Penalties ─────────────────────────────────────────────────────────
    penalties = 0.0
    penalty_notes = []
    for p in patients:
        r = rank_map.get(p.patient_id)
        if r is None:
            continue
        if p.true_esi == 1 and r.esi_level >= 4:
            penalties += 0.25
            penalty_notes.append(f"{p.patient_id}: ESI-1 wrongly assigned ESI-{r.esi_level}")
        elif p.true_esi == 1 and r.esi_level == 3:
            penalties += 0.10
            penalty_notes.append(f"{p.patient_id}: ESI-1 assigned ESI-3")
        elif p.true_esi == 2 and r.esi_level >= 5:
            penalties += 0.10
            penalty_notes.append(f"{p.patient_id}: ESI-2 assigned ESI-5")

    penalties = min(penalties, 0.5)  # cap

    # ── Composite reward ──────────────────────────────────────────────────
    raw = (
        0.30 * esi_accuracy +
        0.25 * rank_order +
        0.25 * critical_catch +
        0.20 * rationale_quality
    )
    total_reward = max(0.0, min(1.0, raw - penalties))

    return TriageReward(
        total=round(total_reward, 4),
        esi_accuracy=round(esi_accuracy, 4),
        rank_order=round(rank_order, 4),
        critical_catch=round(critical_catch, 4),
        rationale_quality=round(rationale_quality, 4),
        penalties=round(penalties, 4),
        breakdown={
            "esi": esi_breakdown,
            "penalty_notes": penalty_notes,
            "n_patients": total,
            "n_critical": n_crit,
            "exact_esi_matches": exact_matches,
        },
    )
