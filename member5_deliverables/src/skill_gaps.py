"""Skill gap: Priority(s) = importance * (1 - mastery)."""

def compute_skill_gaps(job_skills, learner_estimates):
    gaps = []
    for s in job_skills:
        d = s["importance"]
        m = learner_estimates.get(s["esco_id"], 0.0)  # unknown skill -> 0
        gaps.append({
            "esco_id": s["esco_id"], "label": s["label"],
            "importance": d, "mastery": m,
            "priority": round(d * (1 - m), 3),
        })
    return sorted(gaps, key=lambda g: g["priority"], reverse=True)
