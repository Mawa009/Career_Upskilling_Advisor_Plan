"""Course ranking: gap-based score, prerequisites, difficulty, diversity."""
import pandas as pd
from skill_gaps import compute_skill_gaps

MIN_HOURS = 5  # stops 1-2 hour courses from winning just by being short


def score_courses(gaps, courses_df, min_hours=MIN_HOURS):
    priority = {g["esco_id"]: g["priority"] for g in gaps}
    rows = []
    for _, c in courses_df.iterrows():
        total = sum(priority.get(s, 0.0) for s in c["esco_skill_ids"].split("|"))
        rows.append({
            "course_id": c["course_id"], "title": c["title"],
            "hours": c["estimated_hours"], "gap_closed": round(total, 3),
            "score": round(total / max(c["estimated_hours"], min_hours), 4),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def filter_by_prerequisites(scores, courses_df, completed=None):
    completed = completed or set()
    prereq = dict(zip(courses_df["course_id"], courses_df["prerequisites"]))
    def unlocked(cid):
        return all(p in completed for p in prereq[cid].split("|") if p)
    return scores[scores["course_id"].apply(unlocked)]


def filter_by_difficulty(scores, courses_df, learner_estimates, min_mastery=0.5):
    catalog = courses_df.set_index("course_id")
    def ready(cid):
        if catalog.loc[cid, "difficulty"] != "advanced":
            return True
        skills = catalog.loc[cid, "esco_skill_ids"].split("|")
        return sum(learner_estimates.get(s, 0.0) for s in skills) / len(skills) >= min_mastery
    return scores[scores["course_id"].apply(ready)]


def _pick_diverse(job_skills, learner_estimates, catalogue, completed, top_n):
    remaining = [dict(g) for g in compute_skill_gaps(job_skills, learner_estimates)]
    skills_of = catalogue.set_index("course_id")["esco_skill_ids"]
    chosen = []
    for _ in range(top_n):
        scores = score_courses(remaining, catalogue)
        avail = filter_by_prerequisites(scores, catalogue, completed)
        avail = filter_by_difficulty(avail, catalogue, learner_estimates)
        avail = avail[~avail["course_id"].isin(chosen) & (avail["score"] > 0)]
        if avail.empty:
            break
        best = avail.iloc[0]["course_id"]
        chosen.append(best)
        covered = set(skills_of[best].split("|"))
        for g in remaining:
            if g["esco_id"] in covered:
                g["priority"] = 0.0
    return chosen


def recommend_courses(job_skills, learner_estimates, course_catalogue,
                      completed=None, top_n=5):
    """Team contract: returns [{course_id, score, reason}]."""
    ids = _pick_diverse(job_skills, learner_estimates, course_catalogue, completed, top_n)
    gaps_by_id = {g["esco_id"]: g for g in compute_skill_gaps(job_skills, learner_estimates)}
    cat = course_catalogue.set_index("course_id")
    results, covered = [], set()
    for cid in ids:
        skills = cat.loc[cid, "esco_skill_ids"].split("|")
        new = [s for s in skills
               if s in gaps_by_id and gaps_by_id[s]["priority"] > 0 and s not in covered]
        covered.update(skills)
        hours = cat.loc[cid, "estimated_hours"]
        score = sum(gaps_by_id[s]["priority"] for s in new) / max(hours, MIN_HOURS)
        parts = ", ".join(f"{gaps_by_id[s]['label']} (priority {gaps_by_id[s]['priority']})"
                          for s in new)
        results.append({"course_id": cid, "score": round(float(score), 4),
                        "reason": f"Closes new gap: {parts} in {int(hours)} hours"})
    return results
