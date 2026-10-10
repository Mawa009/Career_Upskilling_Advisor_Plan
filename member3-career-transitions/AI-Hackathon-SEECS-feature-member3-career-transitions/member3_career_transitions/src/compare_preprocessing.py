
"""Fair validation comparison of original vs strict preprocessing."""

import json
import pandas as pd

from .career_paths import CareerTransitionModel
from .preprocessing import records_to_trajectories
from .overlap_experiment import strict_trajectories
from .evaluation import evaluate


def main():
    print("Loading training records...")
    train = pd.read_parquet(
        "data/JobHop_v2_train.parquet"
    ).drop_duplicates()

    print("Preparing strict trajectories...")
    strict_train = list(
        strict_trajectories(train.to_dict("records"))
    )

    print("Training strict model...")
    strict_model = CareerTransitionModel(
        min_support=3
    ).fit(strict_train)

    strict_model.save(
        "models/strict_transition_counts.json"
    )

    print("Loading original model...")
    original_model = CareerTransitionModel.load(
        "models/transition_counts.json"
    )

    print("Preparing shared validation cases...")
    val = pd.read_parquet(
        "data/JobHop_v2_val.parquet"
    )
    val_trajectories = list(
        records_to_trajectories(val.to_dict("records"))
    )

    print("Evaluating both models...")
    results = {
        "original": evaluate(
            original_model, val_trajectories
        ),
        "strict": evaluate(
            strict_model, val_trajectories
        ),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
