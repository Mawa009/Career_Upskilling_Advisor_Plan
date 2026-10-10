
"""Compare original and overlap-aware career trajectories."""

from collections import defaultdict

import pandas as pd

from .preprocessing import (
    quarter_value,
    records_to_trajectories,
)


def strict_trajectories(rows):
    grouped = defaultdict(list)

    for row in rows:
        person = row.get("resume_id")
        code = row.get("matched_code")
        start = quarter_value(row.get("start_date"))
        end = quarter_value(row.get("end_date"))

        if (
            person is None
            or start is None
            or not isinstance(code, str)
            or not code.strip()
            or code.strip().lower() in ("unknown", "nan", "none")
        ):
            continue

        grouped[person].append(
            (start, end, code.strip())
        )

    for jobs in grouped.values():
        jobs = sorted(set(jobs), key=lambda x: (x[0], x[2]))

        by_start = defaultdict(set)
        for start, end, code in jobs:
            by_start[start].add(code)

        ambiguous_starts = {
            start for start, codes in by_start.items()
            if len(codes) > 1
        }

        clean_jobs = [
            job for job in jobs
            if job[0] not in ambiguous_starts
        ]

        sequence = []
        previous_end = None

        for start, end, code in clean_jobs:
            if sequence and (
                previous_end is None or start < previous_end
            ):
                # Strict rule: do not infer a move while
                # the previous role may still be active.
                continue

            if not sequence or sequence[-1] != code:
                sequence.append(code)

            previous_end = end

        yield sequence


def count_transitions(trajectories):
    return sum(
        max(len(seq) - 1, 0)
        for seq in trajectories
    )


def main():
    df = pd.read_parquet(
        "data/JobHop_v2_train.parquet"
    ).drop_duplicates()

    records = df.to_dict("records")

    original = list(records_to_trajectories(records))
    strict = list(strict_trajectories(records))

    print("Original trajectories:", len(original))
    print("Strict trajectories:", len(strict))
    print(
        "Original candidate transitions:",
        count_transitions(original)
    )
    print(
        "Strict candidate transitions:",
        count_transitions(strict)
    )


if __name__ == "__main__":
    main()
