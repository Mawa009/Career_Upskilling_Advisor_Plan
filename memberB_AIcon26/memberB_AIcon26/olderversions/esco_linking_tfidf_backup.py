import json
import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ESCO_FILE = r"D:\AIcon26\skills_en.csv"


class ESCOSkillLinker:
    def __init__(self, csv_path=ESCO_FILE):
        self.concepts = {}
        self.aliases = []
        self.alias_concept_uris = []

        self._load_dataset(csv_path)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.alias_vectors = self.vectorizer.fit_transform(
            self.aliases
        )

        print(f"Loaded {len(self.concepts)} ESCO concepts.")
        print(f"Indexed {len(self.aliases)} labels and aliases.")

    def _load_dataset(self, csv_path):
        with open(
            csv_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                if row.get("status", "").lower() != "released":
                    continue

                uri = row.get("conceptUri", "").strip()
                label = row.get("preferredLabel", "").strip()

                if not uri or not label:
                    continue

                self.concepts[uri] = {
                    "conceptUri": uri,
                    "preferredLabel": label,
                    "description": row.get("description", "").strip()
                }

                # Include the official name and alternative names.
                labels = [label]

                for column in ("altLabels", "hiddenLabels"):
                    value = row.get(column, "")

                    if value:
                        labels.extend(value.splitlines())

                for alias in labels:
                    alias = re.sub(r"\s+", " ", alias).strip()

                    if alias:
                        self.aliases.append(alias)
                        self.alias_concept_uris.append(uri)

    def map_to_esco(self, extracted_skills, job_text=None, top_k=3):
        results = []

        for skill in extracted_skills:
            skill = str(skill).strip()

            if not skill:
                continue

            query_vector = self.vectorizer.transform([skill])

            scores = cosine_similarity(
                query_vector,
                self.alias_vectors
            ).flatten()

            # Keep the best score for each ESCO concept.
            best_scores = {}

            for index, score in enumerate(scores):
                uri = self.alias_concept_uris[index]

                if uri not in best_scores or score > best_scores[uri]:
                    best_scores[uri] = float(score)

            ranked_uris = sorted(
                best_scores,
                key=best_scores.get,
                reverse=True
            )[:top_k]

            candidates = []

            for uri in ranked_uris:
                concept = self.concepts[uri]

                candidates.append({
                    "conceptUri": uri,
                    "preferredLabel": concept["preferredLabel"],
                    "similarity_score": round(best_scores[uri], 4),
                    "description": concept["description"]
                })

            
            best_score = (
                candidates[0]["similarity_score"]
                if candidates else 0.0
            )

            second_score = (
                candidates[1]["similarity_score"]
                if len(candidates) > 1 else 0.0
            )

            if best_score < 0.55:
                match_status = "LOW_CONFIDENCE_REVIEW"
            elif len(candidates) > 1 and (
                best_score - second_score < 0.05
            ):
                match_status = "AMBIGUOUS_REVIEW"
            else:
                match_status = "LIKELY_MATCH"

            
            best_score = (
                candidates[0]["similarity_score"]
                if candidates else 0.0
            )

            second_score = (
                candidates[1]["similarity_score"]
                if len(candidates) > 1 else 0.0
            )

            if best_score < 0.55:
                match_status = "LOW_CONFIDENCE_REVIEW"
            elif len(candidates) > 1 and (
                best_score - second_score < 0.05
            ):
                match_status = "AMBIGUOUS_REVIEW"
            else:
                match_status = "LIKELY_MATCH"

            results.append({
                "extracted_skill": skill,
                "best_match": candidates[0] if candidates else None,
                "candidates": candidates,
                "match_status": match_status
            })



        return results




if __name__ == "__main__":
    import csv
    import json

    linker = ESCOSkillLinker()

    # Load skill phrases from Member 1.
    input_file = r"D:\AIcon26\member1\mock\extracted_skills.json"
    output_file = r"D:\AIcon26\member1_esco_linking_results.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        test_skills = json.load(f)

    print(f"Loaded {len(test_skills)} skills from Member 1.")

    # Run Member 2's ESCO matcher.
    results = linker.map_to_esco(test_skills, top_k=3)

    csv_rows = []

    for result in results:
        skill = result["extracted_skill"]
        status = result["match_status"]

        print(f"\nExtracted skill: {skill}")
        print(f"Match status: {status}")

        for candidate in result["candidates"]:
            label = candidate["preferredLabel"]
            uri = candidate["conceptUri"]
            score = candidate["similarity_score"]

            print(f"  ESCO match: {label}")
            print(f"  Similarity: {score}")

            csv_rows.append({
                "extracted_skill": skill,
                "match_status": status,
                "esco_label": label,
                "esco_uri": uri,
                "similarity_score": score,
                "description": candidate["description"]
            })

    # Save results to a new CSV.
    fieldnames = [
        "extracted_skill",
        "match_status",
        "esco_label",
        "esco_uri",
        "similarity_score",
        "description"
    ]

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nProcessed {len(results)} input skills.")
    print(f"Results saved to: {output_file}")
