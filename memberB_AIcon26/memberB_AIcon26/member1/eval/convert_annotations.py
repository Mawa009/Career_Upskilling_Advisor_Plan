"""Convert (jobs.csv + offset annotations.csv) -> gold JSONL for eval.evaluate.

    python -m eval.convert_annotations data/jobs_linkedin_50.csv data/annotations_draft_50.csv data/gold.jsonl

Validates that every annotation's character offsets really select its phrase in the job text
(blank offsets are allowed for expanded phrases and skip that check),
and reports problems instead of silently scoring against misaligned labels.
Re-run it after the human review pass; only rows you keep in the CSV reach the gold file.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict


def convert(jobs_csv: str, ann_csv: str, out_jsonl: str, text_col: str = "job_description") -> dict:
    jobs = {r["job_id"]: r for r in csv.DictReader(open(jobs_csv, encoding="utf-8-sig", newline=""))}
    by, problems, statuses = defaultdict(list), [], set()
    for a in csv.DictReader(open(ann_csv, encoding="utf-8-sig", newline="")):
        job = jobs.get(a["job_id"])
        if job is None:
            problems.append(f"{a['job_id']}: annotation for unknown job")
            continue
        text = job[text_col]
        if str(a.get("start_char", "")).strip() or str(a.get("end_char", "")).strip():
            s, e = int(a["start_char"]), int(a["end_char"])
            if text[s:e] != a["skill_phrase"]:
                problems.append(f"{a['job_id']}: offsets {s}:{e} select {text[s:e]!r}, not {a['skill_phrase']!r}")
                continue
        # blank offsets = an expanded phrase that is not a contiguous substring ("collect data" from
        # "Collect, clean, and organize data"); accepted as-is, so leave start_char/end_char empty for those.
        statuses.add(a.get("annotation_status", ""))
        if a["skill_phrase"] not in by[a["job_id"]]:
            by[a["job_id"]].append(a["skill_phrase"])
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for job_id, r in jobs.items():
            f.write(json.dumps({"id": job_id, "category": r.get("job_category", ""), "text": r[text_col],
                                "skills": by.get(job_id, [])}, ensure_ascii=False) + "\n")
    return {"ads": len(jobs), "annotations": sum(map(len, by.values())), "problems": problems, "statuses": sorted(statuses)}


if __name__ == "__main__":
    info = convert(*sys.argv[1:4])
    print(info)
    sys.exit(1 if info["problems"] else 0)
