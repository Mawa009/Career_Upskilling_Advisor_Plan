
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"D:\AIcon26")
ESCO_FILE = BASE_DIR / "skills_en.csv"
INPUT_FILE = BASE_DIR / "member1" / "mock" / "extracted_skills.json"

OUTPUT_CSV = BASE_DIR / "member1_esco_linking_final_results.csv"
OUTPUT_JSON = BASE_DIR / "job_skills.json"
REVIEW_JSON = BASE_DIR / "esco_review_required.json"
CACHE_FILE = BASE_DIR / "esco_label_embeddings_cache.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Conservative semantic matching thresholds.
# Similarity is NOT a probability.
MIN_SCORE = 0.84
MIN_MARGIN = 0.05
MIN_CANDIDATE_SCORE = 0.20

# This specific alias in the supplied results is misleading:
# TensorFlow is a framework, not a synonym for computer vision.
BLOCKED_ALIAS_PAIRS = {
    ("tensorflow", "computer vision"),
}

# Used only if Member 1 provides no usable importance metadata.
DEFAULT_IMPORTANCE = 0.7


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value):
    return clean(value).casefold()


def parse_labels(value):
    return [
        clean(part)
        for part in re.split(r"[\r\n]+", value or "")
        if clean(part)
    ]


def load_esco():
    concepts = []
    seen_uris = set()

    with ESCO_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"conceptUri", "preferredLabel"}
        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "ESCO CSV must contain conceptUri and preferredLabel columns."
            )

        for row in reader:
            if clean(row.get("status")).lower() != "released":
                continue

            uri = clean(row.get("conceptUri"))
            label = clean(row.get("preferredLabel"))

            if not uri or not label or uri in seen_uris:
                continue

            seen_uris.add(uri)

            aliases = []
            for column in ("altLabels", "hiddenLabels"):
                aliases.extend(parse_labels(row.get(column, "")))

            concepts.append({
                "uri": uri,
                "label": label,
                "aliases": list(dict.fromkeys(aliases)),
                "description": clean(row.get("description")),
            })

    if not concepts:
        raise ValueError(
            "No released ESCO concepts loaded. Check the CSV and status column."
        )

    return concepts


def get_importance(metadata):
    raw = metadata.get("importance_hint")

    if raw is not None and raw != "":
        try:
            score = float(raw)
            if 0 <= score <= 1:
                return round(score, 3), "importance_hint"
        except (TypeError, ValueError):
            pass

    requirement = normalize(metadata.get("requirement"))

    scores = {
        "required": 1.0,
        "essential": 1.0,
        "must-have": 1.0,
        "must have": 1.0,
        "mandatory": 1.0,
        "responsibility": 0.8,
        "preferred": 0.5,
        "nice-to-have": 0.5,
        "nice to have": 0.5,
        "company": 0.3,
        "unknown": 0.7,
    }

    if requirement in scores:
        return scores[requirement], "requirement_category"

    return DEFAULT_IMPORTANCE, "provisional_default"


def load_skills():
    with INPUT_FILE.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("skills", data.get("extracted_skills", []))

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of skills.")

    results = []
    seen = set()

    for item in data:
        if isinstance(item, str):
            skill = clean(item)
            metadata = {}
        elif isinstance(item, dict):
            skill = clean(
                item.get("text")
                or item.get("extracted_skill")
                or item.get("skill")
                or item.get("preferredLabel")
            )
            metadata = item
        else:
            continue

        key = normalize(skill)

        if not key or key in seen:
            continue

        seen.add(key)
        importance, importance_source = get_importance(metadata)

        results.append({
            "text": skill,
            "importance": importance,
            "importance_source": importance_source,
            "requirement": clean(metadata.get("requirement")),
            "skill_confidence": metadata.get("confidence", ""),
        })

    return results


# ============================================================
# EMBEDDINGS AND CACHE
# ============================================================

def get_embeddings(model, concepts):
    labels = [concept["label"] for concept in concepts]

    fingerprint = hashlib.sha256(
        (MODEL_NAME + "\n" + "\n".join(labels)).encode("utf-8")
    ).hexdigest()

    if CACHE_FILE.exists():
        try:
            with np.load(CACHE_FILE, allow_pickle=False) as cached:
                if (
                    str(cached["fingerprint"].item()) == fingerprint
                    and cached["vectors"].shape[0] == len(labels)
                ):
                    print("Using cached ESCO embeddings.")
                    return cached["vectors"].astype(
                        np.float32, copy=False
                    )
        except (OSError, ValueError, KeyError):
            print("Cache invalid; rebuilding embeddings.")

    print("Encoding ESCO labels. This may take several minutes.")

    vectors = model.encode(
        labels,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez_compressed(
        CACHE_FILE,
        fingerprint=np.array(fingerprint),
        vectors=vectors,
    )

    return vectors


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def exact_candidates(skill, concepts):
    key = normalize(skill)
    label_matches = []
    alias_matches = []

    for index, concept in enumerate(concepts):
        label_key = normalize(concept["label"])

        if key == label_key:
            label_matches.append((index, "EXACT_LABEL"))

        elif any(key == normalize(alias) for alias in concept["aliases"]):
            alias_matches.append((index, "EXACT_ALIAS"))

    # A preferred-label match takes precedence over alias matches.
    return label_matches if label_matches else alias_matches


def semantic_candidates(
    skill, query_vector, concept_vectors, concepts
):
    scores = concept_vectors @ query_vector
    ranked = np.argsort(scores)[::-1]

    candidates = []
    seen_labels = set()

    for index in ranked:
        index = int(index)
        score = float(scores[index])

        if score < MIN_CANDIDATE_SCORE:
            break

        concept = concepts[index]
        key = normalize(concept["label"])

        if key in seen_labels:
            continue

        seen_labels.add(key)

        candidates.append({
            "index": index,
            "label": concept["label"],
            "uri": concept["uri"],
            "score": score,
            "description": concept["description"],
            "match_method": "SEMANTIC",
        })

        if len(candidates) >= TOP_K:
            break

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():
    if not ESCO_FILE.exists():
        raise FileNotFoundError(f"ESCO file not found: {ESCO_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("Loading ESCO...")
    concepts = load_esco()
    print(f"Loaded {len(concepts)} ESCO concepts.")

    skills = load_skills()
    print(f"Loaded {len(skills)} input skills.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    concept_vectors = get_embeddings(model, concepts)

    query_vectors = model.encode(
        [item["text"] for item in skills],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    csv_rows = []
    accepted = []
    review_items = []

    for item, query_vector in zip(skills, query_vectors):
        skill = item["text"]

        exact = exact_candidates(skill, concepts)
        semantic = semantic_candidates(
            skill, query_vector, concept_vectors, concepts
        )

        candidates = []

        # Exact preferred labels and aliases are considered first.
        for index, method in exact:
            concept = concepts[index]
            candidates.append({
                "index": index,
                "label": concept["label"],
                "uri": concept["uri"],
                "score": "",
                "description": concept["description"],
                "match_method": method,
            })

        exact_uris = {c["uri"] for c in candidates}

        for candidate in semantic:
            if candidate["uri"] not in exact_uris:
                candidates.append(candidate)

        # Determine one decision for the skill, not for every candidate.
        selected = None
        status = "REVIEW_REQUIRED"

        if exact:
            # An explicit known-bad alias must not be accepted.
            safe_exact = []

            for index, method in exact:
                concept = concepts[index]
                blocked = (
                    method == "EXACT_ALIAS"
                    and (
                        normalize(skill),
                        normalize(concept["label"]),
                    ) in BLOCKED_ALIAS_PAIRS
                )

                if not blocked:
                    safe_exact.append((index, method))

            if safe_exact:
                # Multiple concepts matching the same alias are ambiguous.
                if len(safe_exact) == 1:
                    index, method = safe_exact[0]
                    selected = next(
                        c for c in candidates
                        if c["uri"] == concepts[index]["uri"]
                    )
                    status = method
                else:
                    status = "AMBIGUOUS_EXACT_ALIAS"

        # Semantic-only matching uses a high threshold and a margin.
        elif semantic:
            top = semantic[0]
            second_score = (
                semantic[1]["score"] if len(semantic) > 1 else -1.0
            )
            margin = top["score"] - second_score

            if (
                top["score"] >= MIN_SCORE
                and margin >= MIN_MARGIN
            ):
                selected = top
                status = "LIKELY_MATCH_REQUIRES_CONFIRMATION"

        # Candidate rows retain the overall skill decision.
        for rank, candidate in enumerate(candidates[:TOP_K], start=1):
            is_selected = (
                selected is not None
                and candidate["uri"] == selected["uri"]
            )

            csv_rows.append({
                "input_skill": skill,
                "rank": rank,
                "esco_label": candidate["label"],
                "esco_uri": candidate["uri"],
                "similarity_score": (
                    round(candidate["score"], 4)
                    if candidate["score"] != "" else ""
                ),
                "match_method": candidate["match_method"],
                "match_status": status,
                "candidate_selected": is_selected,
                "importance_score": item["importance"],
                "importance_source": item["importance_source"],
                "requirement": item["requirement"],
                "skill_confidence": item["skill_confidence"],
                "description": candidate["description"],
            })

        if selected is not None:
            record = {
                "input_skill": skill,
                "esco_id": selected["uri"],
                "label": selected["label"],
                "importance": item["importance"],
                "importance_source": item["importance_source"],
                "match_method": selected["match_method"],
                "match_status": status,
                "similarity_score": (
                    round(selected["score"], 4)
                    if selected["score"] != "" else None
                ),
            }

            # Exact preferred-label and alias matches can be handed off.
            # Semantic matches remain provisional and need confirmation.
            if selected["match_method"] in (
                "EXACT_LABEL", "EXACT_ALIAS"
            ):
                accepted.append({
                    "esco_id": selected["uri"],
                    "label": selected["label"],
                    "importance": item["importance"],
                })

            else:
                review_items.append(record)

        else:
            review_items.append({
                "input_skill": skill,
                "status": status,
                "candidates": [
                    {
                        "label": c["label"],
                        "esco_id": c["uri"],
                        "similarity_score": (
                            round(c["score"], 4)
                            if c["score"] != "" else None
                        ),
                    }
                    for c in candidates[:TOP_K]
                ],
            })

        print(f"{skill}: {status}")

    # Candidate audit CSV.
    fieldnames = [
        "input_skill", "rank", "esco_label", "esco_uri",
        "similarity_score", "match_method", "match_status",
        "candidate_selected", "importance_score",
        "importance_source", "requirement", "skill_confidence",
        "description",
    ]

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Team contract: [{esco_id, label, importance}, ...]
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(accepted, file, indent=2, ensure_ascii=False)

    with REVIEW_JSON.open("w", encoding="utf-8") as file:
        json.dump(review_items, file, indent=2, ensure_ascii=False)

    print("\nFinished.")
    print(f"Input skills: {len(skills)}")
    print(f"Accepted exact mappings: {len(accepted)}")
    print(f"Items needing review: {len(review_items)}")
    print(f"Candidate CSV: {OUTPUT_CSV}")
    print(f"Team handoff JSON: {OUTPUT_JSON}")
    print(f"Review file: {REVIEW_JSON}")
    print(
        "\nNote: semantic matches are deliberately excluded from "
        "job_skills.json until a human confirms them."
    )


if __name__ == "__main__":
    main()
