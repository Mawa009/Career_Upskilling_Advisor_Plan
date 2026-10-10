"""Speed up hand-labeling: run the extractor over raw ads and write a draft gold file.

Input : a .jsonl with {"id","text"} per line, OR a .txt where ads are separated by a line of '====='.
Output: .jsonl in gold format, "skills" pre-filled with the model's guesses.
Then open the output, DELETE wrong skills and ADD missed ones. Correcting is ~3x faster than labeling from scratch.
Note: pre-labeling biases the gold toward the model; add every missed skill you can see, and label blind
(--blank) for a sample of ~10 ads to measure that bias.

    python -m eval.prelabel ads.txt eval/gold_draft.jsonl --esco skills_en.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from src.extraction import SkillExtractor
except ImportError:
    from extraction import SkillExtractor


def read_ads(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        for i, line in enumerate(l for l in raw.splitlines() if l.strip()):
            row = json.loads(line)
            yield row.get("id", f"ad{i+1:03d}"), row["text"]
    else:
        for i, chunk in enumerate(c.strip() for c in re.split(r"^\s*={3,}\s*$", raw, flags=re.M) if c.strip()):
            yield f"ad{i+1:03d}", chunk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ads")
    ap.add_argument("out")
    ap.add_argument("--esco", default=None)
    ap.add_argument("--blank", action="store_true", help="write empty skill lists (blind labeling)")
    args = ap.parse_args(argv)
    ads = list(read_ads(args.ads))
    ex = None if args.blank else SkillExtractor(args.esco)
    preds = [[] for _ in ads] if args.blank else ex.extract_skills_many([t for _, t in ads])
    with open(args.out, "w", encoding="utf-8") as fh:
        for (ad_id, text), skills in zip(ads, preds):
            fh.write(json.dumps({"id": ad_id, "text": text, "skills": skills}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ads)} ads to {args.out}")


if __name__ == "__main__":
    main()
