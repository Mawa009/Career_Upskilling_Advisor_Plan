
"""Compare first-order and context-aware career prediction."""

import json
import pandas as pd

from .second_order import SecondOrderCareerModel
from .career_paths import CareerTransitionModel
from .preprocessing import records_to_trajectories


def score_predictions(predictions, actual, k=5):
    ids = [
        item["occupation_id"]
        for item in predictions[:k]
    ]
    if actual not in ids:
        return 0, 0.0

    return 1, 1.0 / (ids.index(actual) + 1)


def main():
    print("Loading training histories...", flush=True)
    train = pd.read_parquet("data/JobHop_v2_train.parquet")

    train_sequences = list(
        records_to_trajectories(train.to_dict("records"))
    )

    print("Training second-order model...", flush=True)
    new_model = SecondOrderCareerModel().fit(train_sequences)

    baseline = CareerTransitionModel.load(
        "models/transition_counts.json"
    )

    print("Loading validation histories...", flush=True)
    val = pd.read_parquet("data/JobHop_v2_val.parquet")
    sequences = list(
        records_to_trajectories(val.to_dict("records"))
    )

    results = {
        "baseline": {"hits": 0, "rr": 0},
        "second_order": {"hits": 0, "rr": 0},
    }

    n = 0
    contexts_used = 0

    for seq in sequences:
        if len(seq) < 3:
            continue

        previous, current, actual = seq[-3:]

        if current == actual:
            continue

        n += 1

        context_support = sum(
            new_model.context_counts.get(
                (previous, current), {}
            ).values()
        )

        if context_support >= new_model.min_context_support:
            contexts_used += 1

        predictions = {
            "baseline": baseline.predict_next_occupations(current),
            "second_order": new_model.predict_next_occupations(
                current, previous
            )
        }

        for name, recs in predictions.items():
            hit, rr = score_predictions(recs, actual)
            results[name]["hits"] += hit
            results[name]["rr"] += rr

    output = {
        "evaluated_cases": n,
        "context_available_rate": contexts_used / n if n else 0,
    }

    for name, result in results.items():
        output[name] = {
            "recall@5": result["hits"] / n if n else 0,
            "mrr": result["rr"] / n if n else 0,
        }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
