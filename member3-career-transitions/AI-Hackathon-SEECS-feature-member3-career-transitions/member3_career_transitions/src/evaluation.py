"""Leave-last-transition-out evaluation on independent held-out people."""
from __future__ import annotations
from .career_paths import CareerTransitionModel
from .preprocessing import records_to_trajectories


def evaluate(model: CareerTransitionModel, trajectories, k=5):
    n = hits = rr_sum = covered = 0
    for seq in trajectories:
        if len(seq) < 2:
            continue
        origin, actual = seq[-2], seq[-1]
        if origin == actual:
            continue
        n += 1
        recs = model.predict_next_occupations(origin, top_k=k)
        if recs:
            covered += 1
        matches = [r['occupation_id'] for r in recs]
        if actual in matches:
            hits += 1
            rr_sum += 1 / (matches.index(actual) + 1)
    return {'n_test': n, f'recall@{k}': hits / n if n else 0,
            'mrr': rr_sum / n if n else 0,
            'coverage': covered / n if n else 0}


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/transition_counts.json')
    parser.add_argument('--test', default='data/JobHop_v2_test.parquet')
    args = parser.parse_args()
    model = CareerTransitionModel.load(args.model)
    frame = pd.read_parquet(args.test)
    print(evaluate(model, records_to_trajectories(frame.to_dict('records'))))


if __name__ == '__main__':
    main()
