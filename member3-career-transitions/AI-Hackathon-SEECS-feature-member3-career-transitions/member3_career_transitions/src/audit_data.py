
"""Audit JobHop v2 career history quality."""

import pandas as pd

PATH = "data/JobHop_v2_train.parquet"


def quarter_number(series):
    parts = series.astype("string").str.extract(
        r"^Q([1-4])\s+(\d{4})$"
    )
    quarter = pd.to_numeric(parts[0], errors="coerce")
    year = pd.to_numeric(parts[1], errors="coerce")
    return year * 4 + quarter - 1


def main():
    df = pd.read_parquet(PATH)

    df["start_q"] = quarter_number(df["start_date"])
    df["end_q"] = quarter_number(df["end_date"])

    print("Total employment records:", len(df))
    print("Unique people:", df["resume_id"].nunique())
    print("Missing/invalid starts:", df["start_q"].isna().sum())
    print("Missing/invalid ends:", df["end_q"].isna().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df = df.dropna(subset=["start_q"])
    df = df.sort_values(
        ["resume_id", "start_q", "matched_code"],
        kind="stable"
    )

    previous_start = df.groupby("resume_id")["start_q"].shift()
    previous_end = df.groupby("resume_id")["end_q"].shift()

    same_start = df["start_q"].eq(previous_start)
    overlap = (
        previous_end.notna()
        & (df["start_q"] < previous_end)
        & ~same_start
    )

    print("Adjacent records with same start:", int(same_start.sum()))
    print("Potential adjacent overlaps:", int(overlap.sum()))


if __name__ == "__main__":
    main()
