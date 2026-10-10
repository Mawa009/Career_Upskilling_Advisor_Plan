"""Verify Member 4 quiz skills against the frozen ESCO 1.2.1 vocabulary."""
import csv
from pathlib import Path


def load_esco_skills(path):
    """Return ESCO URI → preferred label, deduplicated by concept URI."""
    lookup = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uri = row["conceptUri"].strip()
            label = row["preferredLabel"].strip()
            if uri in lookup and lookup[uri] != label:
                raise ValueError(f"Conflicting labels for {uri}")
            lookup[uri] = label
    return lookup


def validate_question_esco(questions, skills_file):
    """Fail fast for invalid URIs, noncanonical labels or placeholder IDs."""
    known = load_esco_skills(skills_file)
    for q in questions:
        uri = q["esco_id"]
        if uri not in known:
            raise ValueError(f"Unknown ESCO skill in {q['question_id']}: {uri}")
        if q.get("esco_label") != known[uri]:
            raise ValueError(f"ESCO label mismatch in {q['question_id']}: {q.get('esco_label')!r} != {known[uri]!r}")
    return True
