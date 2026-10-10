
"""Final held-out test evaluation for Member 3.

Compare first-order Markov and second-order career models
on the same unseen JobHop v2 test histories.
"""

import json
from pathlib import Path

import pandas as pd

from .career_paths import CareerTransitionModel
from .second_order import SecondOrderCareerModel
from .preprocessing import records_to_trajectories


BASE_DIR = Path(__file__).resolve().parents[1]

FIRST_ORDER_PATH = BASE_DIR / "models" / "transition_counts.json"
SECOND_ORDER_PATH = BASE_DIR / "models" / "second_order_counts.json"
TEST_PATH = BASE_DIR / "data" / "JobHop_v2_test.parquet"


def score_predictions(predictions, actual, k=5):
    """Calculate hit and reciprocal rank."""
    predicted_ids = [
        item["occupation_id"]
        for item in predictions[:k]
    ]

    if actual not in predicted_ids:
        return 0, 0.0

    rank = predicted_ids.index(actual) + 1
    return 1, 1.0 / rank


def main():

    print("Loading saved first-order model...", flush=True)
    baseline = CareerTransitionModel.load(FIRST_ORDER_PATH)

    print("Loading saved second-order model...", flush=True)
    new_model = SecondOrderCareerModel.load(SECOND_ORDER_PATH)

    print("Loading final test histories...", flush=True)
    test_df = pd.read_parquet(TEST_PATH)

    trajectories = list(
        records_to_trajectories(test_df.to_dict("records"))
    )

    print("Evaluating both models...", flush=True)

    results = {
        "baseline": {
            "hits": 0,
            "rr": 0.0,
            "covered": 0
        },
        "second_order": {
            "hits": 0,
            "rr": 0.0,
            "covered": 0
        }
    }

    n = 0
    contexts_used = 0

    for seq in trajectories:

        # Both models are evaluated on the same cases.
        # At least three occupations are required.
        if len(seq) < 3:
            continue

        previous, current, actual = seq[-3:]

        if current == actual:
            continue

        n += 1

        context_counts = new_model.context_counts.get(
            (previous, current), {}
        )

        context_support = sum(context_counts.values())

        if context_support >= new_model.min_context_support:
            contexts_used += 1

        predictions = {
            "baseline": baseline.predict_next_occupations(
                current, top_k=5
            ),
            "second_order": new_model.predict_next_occupations(
                current, previous, top_k=5
            )
        }

        for name, recs in predictions.items():

            if recs:
                results[name]["covered"] += 1

            hit, rr = score_predictions(
                recs, actual, k=5
            )

            results[name]["hits"] += hit
            results[name]["rr"] += rr

    output = {
        "dataset": "JobHop v2 Test",
        "evaluated_cases": n,
        "context_available_rate": (
            contexts_used / n if n else 0
        )
    }

    for name, result in results.items():

        output[name] = {
            "recall@5": (
                result["hits"] / n if n else 0
            ),
            "mrr": (
                result["rr"] / n if n else 0
            ),
            "coverage": (
                result["covered"] / n if n else 0
            )
        }

    print("\nFINAL TEST RESULTS")
    print("=" * 50)
    print(json.dumps(output, indent=2))

    results_path = BASE_DIR / "models" / "final_test_results.json"
    results_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    print("\nResults saved to:", results_path)


if __name__ == "__main__":
    main()
