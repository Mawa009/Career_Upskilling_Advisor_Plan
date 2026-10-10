
import csv
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ESCO_FILE = Path(r"D:\AIcon26\skills_en.csv")
OUTPUT_FILE = Path(r"D:\AIcon26\esco_linking_results.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
MIN_SCORE = 0.35
REVIEW_GAP = 0.03


def clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def main():
    print("Loading ESCO dataset...")

    concepts = {}
    with ESCO_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status", "").lower() != "released":
                continue

            uri = clean(row.get("conceptUri"))
            label = clean(row.get("preferredLabel"))

            if not uri or not label:
                continue

            concepts[uri] = {
                "uri": uri,
                "label": label,
                "description": clean(row.get("description")),
                "aliases": [
                    clean(alias)
                    for col in ("altLabels", "hiddenLabels")
                    for alias in (row.get(col) or "").splitlines()
                    if clean(alias)
                ]
            }

    items = list(concepts.values())
    print(f"Loaded {len(items)} ESCO concepts.")

    print("Loading pretrained ML model...")
    model = SentenceTransformer(MODEL_NAME)

    # Represent each ESCO concept by its official label,
    # aliases and description.
    concept_texts = []
    for item in items:
        parts = [item["label"]]
        parts.extend(item["aliases"])
        if item["description"]:
            parts.append(item["description"])

        concept_texts.append(
            item["label"] + ". " + ". ".join(parts[1:])
        )

    print("Encoding ESCO concepts. This may take a little time...")
    concept_vectors = model.encode(
        concept_texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    test_skills = [
        "write SQL queries",
        "build dashboards",
        "analyse data",
        "program in Python"
    ]

    print("\nEncoding input skills...")
    query_vectors = model.encode(
        test_skills,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarity_matrix = cos_sim(query_vectors, concept_vectors)

    output_rows = []

    for skill_index, skill in enumerate(test_skills):
        scores = similarity_matrix[skill_index]
        top_indices = scores.topk(min(TOP_K, len(items))).indices.tolist()

        candidates = [
            (items[index], float(scores[index].item()))
            for index in top_indices
        ]

        best_item, best_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0

        if best_score < MIN_SCORE:
            status = "LOW_CONFIDENCE_REVIEW"
        elif best_score - second_score < REVIEW_GAP:
            status = "AMBIGUOUS_REVIEW"
        else:
            status = "LIKELY_MATCH"

        print(f"\nInput skill: {skill}")
        print(f"Status: {status}")

        for rank, (item, score) in enumerate(candidates, start=1):
            print(
                f"  {rank}. {item['label']} | "
                f"score={score:.4f} | {item['uri']}"
            )

            output_rows.append({
                "input_skill": skill,
                "rank": rank,
                "esco_label": item["label"],
                "esco_uri": item["uri"],
                "similarity_score": round(score, 4),
                "match_status": status,
                "description": item["description"]
            })

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "input_skill", "rank", "esco_label", "esco_uri",
            "similarity_score", "match_status", "description"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
