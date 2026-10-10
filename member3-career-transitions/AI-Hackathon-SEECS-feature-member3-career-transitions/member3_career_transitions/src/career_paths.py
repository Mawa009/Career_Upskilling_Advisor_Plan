"""Career-transition predictor. Scores are empirical conditional probabilities.

No assumptions about job titles or applicant success probabilities.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path


class CareerTransitionModel:
    def __init__(self, min_support: int = 1):
        self.min_support = min_support
        self.transitions = defaultdict(Counter)
        self.popularity = Counter()

    def fit(self, trajectories):
        """Each trajectory is a time-ordered list of occupation codes."""
        self.transitions = defaultdict(Counter)
        self.popularity = Counter()
        for seq in trajectories:
            cleaned = []
            for raw in seq:
                code = str(raw).strip() if raw is not None else ''
                if not code or code.lower() in ('unknown', 'nan', 'none'):
                    continue
                if not cleaned or code != cleaned[-1]:
                    cleaned.append(code)
            for a, b in zip(cleaned, cleaned[1:]):
                self.transitions[a][b] += 1
                self.popularity[b] += 1
        return self

    def predict_next_occupations(self, occupation_id: str, top_k: int = 5):
        """Return top transitions. Empty list means no reliable source evidence."""
        origin = str(occupation_id).strip()
        counts = self.transitions.get(origin, Counter())
        support = sum(counts.values())
        if support < self.min_support:
            return []
        return [
            {'occupation_id': code, 'label': code, 'probability': n / support}
            for code, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        ]

    def save(self, filename):
        payload = {'min_support': self.min_support,
                   'transitions': {k: dict(v) for k, v in self.transitions.items()}}
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, filename):
        payload = json.loads(Path(filename).read_text(encoding='utf-8'))
        m = cls(min_support=payload.get('min_support', 1))
        m.transitions = defaultdict(Counter, {k: Counter(v) for k, v in payload['transitions'].items()})
        m.popularity = Counter()
        for successors in m.transitions.values():
            m.popularity.update(successors)
        return m


_default_model = None


def predict_next_occupations(occupation_id):
    """Contract for integration; uses MEMBER3_MODEL_PATH or models/transition_counts.json."""
    import os
    global _default_model
    if _default_model is None:
        model_path = Path(os.getenv('MEMBER3_MODEL_PATH',
                                  str(Path(__file__).resolve().parents[1] / 'models' / 'transition_counts.json')))
        if not model_path.is_file():
            raise FileNotFoundError(f'Member 3 model missing: {model_path}. Run python -m src.preprocessing')
        _default_model = CareerTransitionModel.load(model_path)
    return _default_model.predict_next_occupations(occupation_id)
