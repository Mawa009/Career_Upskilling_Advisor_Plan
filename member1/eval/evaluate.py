"""Experiment A harness for Member 1.

Gold format (JSONL, one ad per line):
    {"id": "ad01", "text": "...job ad...", "skills": ["write SQL queries", "Python", ...]}

Usage:
    python -m eval.evaluate --gold eval/sample_gold.jsonl --esco skills_en.csv
    python -m eval.evaluate --gold my_50_ads.jsonl --esco skills_en.csv --no-models   # dictionary-only baseline

Reports micro precision / recall / F1 under two matching rules:
  exact    - normalized strings are equal
  partial  - exact first, then greedy one-to-one match on token Jaccard >= 0.5
and an ablation over the extractor's sources, plus the most common false
positives / false negatives (that list is the fastest way to improve the system).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from src.extraction import ExtractorConfig, SkillExtractor, alias_to_canonical
except ImportError:  # running inside the Colab notebook, where the module sits next to us
    from extraction import ExtractorConfig, SkillExtractor, alias_to_canonical


def _base_norm(s: str) -> str:
    s = s.casefold().replace("’", "'")
    s = re.sub(r"[^\w+#./ -]+", " ", s)
    return " ".join(s.split()).strip(" .-/")


_ALIAS = {_base_norm(k): _base_norm(v) for k, v in alias_to_canonical().items()}


def norm(s: str) -> str:
    """Normalize for matching. Known tool aliases collapse ('React' == 'React.js', 'A/B tests' == 'A/B testing')."""
    n = _base_norm(s)
    return _ALIAS.get(n, n)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def match_counts(pred: Sequence[str], gold: Sequence[str], partial: bool, thr: float = 0.5) -> Tuple[int, List[str], List[str]]:
    """Return (true_positives, unmatched_pred, unmatched_gold) with one-to-one matching."""
    pred_n = list(dict.fromkeys(norm(p) for p in pred if norm(p)))
    gold_n = list(dict.fromkeys(norm(g) for g in gold if norm(g)))
    gold_left, pred_left = list(gold_n), []
    tp = 0
    for p in pred_n:
        if p in gold_left:
            gold_left.remove(p)
            tp += 1
        else:
            pred_left.append(p)
    if partial:
        still = []
        for p in pred_left:
            best, best_j = None, 0.0
            for g in gold_left:
                j = _jaccard(p, g)
                if j > best_j:
                    best, best_j = g, j
            if best is not None and best_j >= thr:
                gold_left.remove(best)
                tp += 1
            else:
                still.append(p)
        pred_left = still
    return tp, pred_left, gold_left


def prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score(preds: List[List[str]], golds: List[List[str]]):
    out, fps, fns = {}, Counter(), Counter()
    for mode in ("exact", "partial"):
        tp = n_pred = n_gold = 0
        for p, g in zip(preds, golds):
            t, p_left, g_left = match_counts(p, g, partial=(mode == "partial"))
            tp += t
            n_pred += len(set(norm(x) for x in p if norm(x)))
            n_gold += len(set(norm(x) for x in g if norm(x)))
            if mode == "partial":
                fps.update(p_left)
                fns.update(g_left)
        out[mode] = prf(tp, n_pred, n_gold)
    return out, fps, fns


def load_gold(path: str) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not isinstance(row.get("skills"), list):
                raise ValueError(f"Bad gold row (need 'text' str and 'skills' list): {line[:80]}")
            rows.append(row)
    return rows


ABLATIONS = {
    "skill model only": dict(use_skill_model=True, use_knowledge_model=False, use_esco=False, use_tech=False),
    "+ knowledge model": dict(use_skill_model=True, use_knowledge_model=True, use_esco=False, use_tech=False),
    "+ ESCO dictionary": dict(use_skill_model=True, use_knowledge_model=True, use_esco=True, use_tech=False),
    "+ tech lexicon (full)": dict(use_skill_model=True, use_knowledge_model=True, use_esco=True, use_tech=True),
    "dictionaries only": dict(use_skill_model=False, use_knowledge_model=False, use_esco=True, use_tech=True),
}


def run_ablation(extractor: SkillExtractor, rows: List[dict]):
    texts = [r["text"] for r in rows]
    golds = [r["skills"] for r in rows]
    table, details = [], {}
    for name, flags in ABLATIONS.items():
        ex = extractor.with_config(**flags)
        needs_skill = flags["use_skill_model"] and ex.skill_tagger is None
        knowledge_only_row = name == "+ knowledge model" and ex.knowledge_tagger is None
        if needs_skill or knowledge_only_row:   # model not loaded: row would be meaningless
            continue
        preds = ex.extract_skills_many(texts)
        res, fps, fns = score(preds, golds)
        table.append((name, res))
        details[name] = (preds, fps, fns)
    # Lower bound: remove every lexicon entry whose name/alias is itself a gold phrase. The lexicon was
    # partly written while looking at the dev ads, so the full row is optimistic and this row is pessimistic
    # (it also removes legitimate entries such as Python). The truth for unseen ads lies between them.
    gold_base = {_base_norm(g) for gs in golds for g in gs}
    exclude = {c for c, al in extractor._lexicon.items() if any(_base_norm(a) in gold_base for a in al + [c])}
    full = extractor.with_config(**ABLATIONS["+ tech lexicon (full)"])
    if full.skill_tagger is not None:
        ex = full.without_tech_terms(exclude)
        preds = ex.extract_skills_many(texts)
        res, fps, fns = score(preds, golds)
        name = f"lexicon minus gold terms ({len(exclude)} removed)"
        table.append((name, res))
        details[name] = (preds, fps, fns)
    return table, details


def by_category(extractor: SkillExtractor, rows: List[dict]) -> Dict[str, dict]:
    """Partial-match P/R/F1 per row['category'] for the full configuration."""
    out = {}
    for cat in sorted({r.get("category", "all") for r in rows}):
        sub = [r for r in rows if r.get("category", "all") == cat]
        preds = extractor.extract_skills_many([r["text"] for r in sub])
        out[cat] = score(preds, [r["skills"] for r in sub])[0]["partial"]
    return out


def format_table(table) -> str:
    head = f"{'configuration':<26}| {'P(exact)':>8} {'R(exact)':>8} {'F1(exact)':>9} | {'P(part)':>7} {'R(part)':>7} {'F1(part)':>8}"
    lines = [head, "-" * len(head)]
    for name, r in table:
        e, p = r["exact"], r["partial"]
        lines.append(f"{name:<26}| {e['precision']:>8.3f} {e['recall']:>8.3f} {e['f1']:>9.3f} | {p['precision']:>7.3f} {p['recall']:>7.3f} {p['f1']:>8.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--esco", default=None, help="path to ESCO skills_en.csv")
    ap.add_argument("--no-models", action="store_true", help="skip loading JobBERT (dictionary-only)")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--top", type=int, default=15, help="how many FP/FN strings to print")
    args = ap.parse_args(argv)

    rows = load_gold(args.gold)
    print(f"{len(rows)} ads, {sum(len(r['skills']) for r in rows)} gold skills")
    extractor = SkillExtractor(args.esco, load_models=not args.no_models)
    table, details = run_ablation(extractor, rows)
    print("\n" + format_table(table))
    print("\nNOTE: the 'full' row is optimistic if the tech lexicon was written while looking at these ads;"
          "\n      'lexicon minus gold terms' is the pessimistic bound. Unseen-ad performance lies between them."
          "\n      Rows without the tech lexicon (model / +knowledge / +ESCO) are unaffected by this.")
    full_name = "+ tech lexicon (full)" if "+ tech lexicon (full)" in details else table[-1][0]
    _, fps, fns = details[full_name]
    print(f"\nTop false positives ({full_name}):")
    for s, c in fps.most_common(args.top):
        print(f"  {c:>2}x  {s}")
    print(f"\nTop false negatives ({full_name}):")
    for s, c in fns.most_common(args.top):
        print(f"  {c:>2}x  {s}")
    if any("category" in r for r in rows) and not args.no_models:
        print("\nPartial-match P/R/F1 by category (full pipeline):")
        for cat, m in by_category(extractor, rows).items():
            print(f"  {cat:<24} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({n: r for n, r in table}, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
