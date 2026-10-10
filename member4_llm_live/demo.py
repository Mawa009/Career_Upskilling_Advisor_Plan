"""Run from project root: python demo.py"""
import json
from pathlib import Path
from src.learner_model import estimate_skill_knowledge

ROOT = Path(__file__).resolve().parent
events = json.loads((ROOT / "data/interactions.json").read_text())
for method in ("accuracy", "bkt"):
    result = estimate_skill_knowledge("L001", events, method=method)
    print(f"{method}: {json.dumps(result, indent=2)}")
    if method == "bkt":
        (ROOT / "outputs/learner_estimates.json").write_text(json.dumps(result, indent=2) + "\n")
print("Saved outputs/learner_estimates.json")
