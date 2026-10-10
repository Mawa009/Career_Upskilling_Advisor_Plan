
import csv
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


BASE_DIR = Path(r"D:\AIcon26")
ESCO_FILE = BASE_DIR / "skills_en.csv"
INPUT_FILE = BASE_DIR / "member1" / "mock" / "extracted_skills.json"
OUTPUT_FILE = BASE_DIR / "member1_esco_linking_transformer_results.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Provisional thresholds: validate them against reviewed examples.
MIN_SCORE = 0.35
MIN_CANDIDATE_SCORE = 0.20
REVIEW_GAP = 0.03


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_esco():
    concepts = []

    with ESCO_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if clean(row.get("status")).lower() != "released":
                continue

            uri = clean(row.get("conceptUri"))
            label = clean(row.get("preferredLabel"))

            if not uri or not label:
                continue

            aliases = []
            for column in ("altLabels", "hiddenLabels"):
                aliases.extend(
                    clean(alias)
                    for alias in (row.get(column) or "").splitlines()
                    if clean(alias)
                )

            concepts.append({
                "uri": uri,
                "label": label,
                "aliases": list(dict.fromkeys(aliases)),
                "description": clean(row.get("description"))
            })

    return concepts


def load_skills():
    with INPUT_FILE.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    # Accept Member 1's simple list or detailed dictionaries.
    skills = []
    for item in data:
        if isinstance(item, str):
            skill = clean(item)
        elif isinstance(item, dict):
            skill = clean(
                item.get("text")
                or item.get("extracted_skill")
                or item.get("skill")
            )
        else:
            continue

        if skill:
            skills.append(skill)

    return list(dict.fromkeys(skills))


def main():
    print("Loading ESCO concepts...")
    concepts = load_esco()
    print(f"Loaded {len(concepts)} concepts.")

    print("Loading Member 1's skill phrases...")
    skills = load_skills()
    print(f"Loaded {len(skills)} unique skills.")

    print(f"Loading pretrained model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Represent each ESCO concept using its label, aliases,
    # and description. The model may truncate long descriptions.
    concept_texts = []

    for concept in concepts:
        aliases = concept["aliases"][:10]
        description = concept["description"][:350]

        text = (
            f"Skill: {concept['label']}. "
            f"Alternative labels: {'; '.join(aliases)}. "
            f"Description: {description}"
        )
        concept_texts.append(text)

    print("Encoding ESCO concepts. This may take a few minutes...")
    concept_vectors = model.encode(
        concept_texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Encoding input skills...")
    query_vectors = model.encode(
        skills,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print("Calculating semantic similarities...")
    similarity_matrix = cos_sim(query_vectors, concept_vectors)

    csv_rows = []

    for skill_index, skill in enumerate(skills):
        scores = similarity_matrix[skill_index]
        top_indices = scores.topk(
            min(TOP_K, len(concepts))
        ).indices.tolist()

        candidates = []

        for index in top_indices:
            concept = concepts[index]
            score = float(scores[index].item())

            # Don't present very weak candidates as useful matches.
            if score < MIN_CANDIDATE_SCORE:
                continue

            candidates.append({
                "label": concept["label"],
                "uri": concept["uri"],
                "score": score,
                "description": concept["description"]
            })

        best_score = candidates[0]["score"] if candidates else 0.0
        second_score = (
            candidates[1]["score"] if len(candidates) > 1 else 0.0
        )

        if not candidates or best_score < MIN_SCORE:
            status = "LOW_CONFIDENCE_REVIEW"
        elif len(candidates) > 1 and best_score - second_score < REVIEW_GAP:
            status = "AMBIGUOUS_REVIEW"
        else:
            status = "LIKELY_MATCH"

        print(f"\nSkill: {skill}")
        print(f"Status: {status}")

        if not candidates:
            print("  No candidate passed the minimum score filter.")

        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"  {rank}. {candidate['label']} "
                f"| similarity={candidate['score']:.4f}"
            )

            csv_rows.append({
                "input_skill": skill,
                "rank": rank,
                "esco_label": candidate["label"],
                "esco_uri": candidate["uri"],
                "similarity_score": round(candidate["score"], 4),
                "match_status": status,
                "description": candidate["description"]
            })

    fieldnames = [
        "input_skill",
        "rank",
        "esco_label",
        "esco_uri",
        "similarity_score",
        "match_status",
        "description"
    ]

    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nProcessed {len(skills)} skills.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
