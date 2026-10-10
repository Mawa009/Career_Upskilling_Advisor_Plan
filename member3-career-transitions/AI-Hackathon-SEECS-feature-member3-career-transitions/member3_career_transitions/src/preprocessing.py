"""Reconstruct conservative transitions from JobHop v2 rows.

Install pandas and pyarrow first. Downloads only the published v2 train split.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
from pathlib import Path
import re
from .career_paths import CareerTransitionModel

QUARTER = re.compile(r'^Q([1-4])\s+(\d{4})$')
URL = 'https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_v2_train.parquet'


def quarter_value(value):
    if not isinstance(value, str):
        return None
    match = QUARTER.fullmatch(value.strip())
    if not match:
        return None
    return int(match.group(2)) * 4 + int(match.group(1)) - 1


def records_to_trajectories(rows):
    """Yield one sequence per person; skip missing starts and concurrent starts.

    Equal-quarter starts are not arbitrarily ordered. Overlap is allowed when
    start dates differ, but it should be reported as a limitation.
    """
    grouped = defaultdict(list)
    for row in rows:
        person = row.get('person_id', row.get('resume_id'))
        code = row.get('esco_code', row.get('matched_code'))
        start = quarter_value(row.get('start_date'))
        if person is None or start is None or not code or str(code).lower() == 'unknown':
            continue
        grouped[person].append((start, str(code)))
    for group in grouped.values():
        group.sort(key=lambda item: (item[0], item[1]))
        # Ambiguous same-start occupations cannot be ordered.
        starts = defaultdict(set)
        for start, code in group:
            starts[start].add(code)
        yield [next(iter(starts[s])) for s in sorted(starts) if len(starts[s]) == 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/JobHop_v2_train.parquet'))
    parser.add_argument('--out', type=Path, default=Path('models/transition_counts.json'))
    parser.add_argument('--min-support', type=int, default=3)
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f'File not found: {args.data}\nDownload v2 train from {URL}')
    import pandas as pd
    frame = pd.read_parquet(args.data)
    required = {'start_date'}
    if not required.issubset(frame.columns) or not ({'person_id', 'resume_id'} & set(frame.columns)) or not ({'esco_code', 'matched_code'} & set(frame.columns)):
        raise ValueError(f'Unexpected schema: {list(frame.columns)}')
    trajectories = list(records_to_trajectories(frame.to_dict('records')))
    model = CareerTransitionModel(min_support=args.min_support).fit(trajectories)
    model.save(args.out)
    print(f'Persons={len(trajectories)}; origins={len(model.transitions)}; stored={args.out}')


if __name__ == '__main__':
    main()
