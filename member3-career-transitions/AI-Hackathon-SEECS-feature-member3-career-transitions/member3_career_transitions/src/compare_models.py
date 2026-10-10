
"""Compare global popularity with first-order career transitions."""

from collections import Counter
import argparse
import json

import pandas as pd

from .career_paths import CareerTransitionModel
from .preprocessing import records_to_trajectories
from .evaluation import evaluate


def global_popularity(model, k=5):
    """
    Derive globally popular destination occupations
    from the fitted transition counts.
    """
    counts = Counter()

    for destinations in model.transitions.values():
        counts.update(destinations)

    return [
        occupation
        for occupation, _ in counts.most_common(k)
    ]


def evaluate_popularity(trajectories, top_jobs, k=5):
    n = hits = rr_sum = 0

    for seq in trajectories:
        if len(seq) < 2:
            continue

        origin, actual = seq[-2], seq[-1]

        if origin == actual:
            continue

        n += 1

        if actual in top_jobs[:k]:
            hits += 1
            rr_sum += 1 / (top_jobs.index(actual) + 1)

    return {
        "n_test": n,
        f"recall@{k}": hits / n if n else 0,
        "mrr": rr_sum / n if n else 0,
    }


def evaluate_direct_coverage(model, trajectories):
    n = direct = 0

    for seq in trajectories:
        if len(seq) < 2:
            continue

        origin, actual = seq[-2], seq[-1]

        if origin == actual:
            continue

        n += 1

        if origin in model.transitions:
            direct += 1

    return {
        "evaluated": n,
        "seen_origin_rate": direct / n if n else 0,
        "unseen_origin_rate": (n - direct) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/transition_counts.json"
    )
    parser.add_argument(
        "--val",
        default="data/JobHop_v2_val.parquet"
    )

    args = parser.parse_args()

    model = CareerTransitionModel.load(args.model)

    frame = pd.read_parquet(args.val)
    trajectories = list(
        records_to_trajectories(frame.to_dict("records"))
    )

    popular_jobs = global_popularity(model)

    results = {
        "popularity": evaluate_popularity(
            trajectories, popular_jobs
        ),
        "markov": evaluate(model, trajectories),
        "coverage_diagnostics": evaluate_direct_coverage(
            model, trajectories
        ),
        "top_popular_occupation_codes": popular_jobs,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
