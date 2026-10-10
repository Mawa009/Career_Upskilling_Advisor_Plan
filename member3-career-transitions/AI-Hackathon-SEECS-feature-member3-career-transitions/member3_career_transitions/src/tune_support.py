
"""Tune minimum support using the validation split."""

import json
import pandas as pd

from .career_paths import CareerTransitionModel
from .preprocessing import records_to_trajectories
from .evaluation import evaluate


def main():
    model = CareerTransitionModel.load(
        "models/transition_counts.json"
    )

    validation = pd.read_parquet(
        "data/JobHop_v2_val.parquet"
    )

    trajectories = list(
        records_to_trajectories(
            validation.to_dict("records")
        )
    )

    results = []

    for threshold in [1, 3, 5, 10, 20]:
        model.min_support = threshold

        metrics = evaluate(model, trajectories)

        results.append({
            "min_support": threshold,
            **metrics
        })

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
