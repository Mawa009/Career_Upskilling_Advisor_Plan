
"""Second-order career transition model with first-order backoff."""

from collections import Counter, defaultdict
from .career_paths import CareerTransitionModel


class SecondOrderCareerModel:
    def __init__(self, min_context_support=5, smoothing=10):
        self.min_context_support = min_context_support
        self.smoothing = smoothing
        self.first_order = CareerTransitionModel(min_support=1)
        self.context_counts = defaultdict(Counter)

    def fit(self, trajectories):
        trajectories = list(trajectories)
        self.first_order.fit(trajectories)
        self.context_counts = defaultdict(Counter)

        for seq in trajectories:
            cleaned = []

            for raw in seq:
                if raw is None:
                    continue

                code = str(raw).strip()

                if code and code.lower() not in (
                    "unknown", "nan", "none"
                ):
                    if not cleaned or cleaned[-1] != code:
                        cleaned.append(code)

            for previous, current, future in zip(
                cleaned, cleaned[1:], cleaned[2:]
            ):
                self.context_counts[
                    (previous, current)
                ][future] += 1

        return self

    def predict_next_occupations(
        self, current, previous=None, top_k=5
    ):
        current = str(current).strip()

        base_counts = self.first_order.transitions.get(
            current, Counter()
        )
        base_total = sum(base_counts.values())

        if base_total == 0:
            return []

        base_probs = {
            job: count / base_total
            for job, count in base_counts.items()
        }

        context = Counter()

        if previous is not None:
            context = self.context_counts.get(
                (str(previous).strip(), current),
                Counter()
            )

        context_total = sum(context.values())

        if context_total < self.min_context_support:
            context_total = 0
            context = Counter()

        weight = (
            context_total / (context_total + self.smoothing)
            if context_total else 0
        )

        scores = {}

        for job in set(base_probs) | set(context):
            base_probability = base_probs.get(job, 0)

            context_probability = (
                context[job] / context_total
                if context_total else 0
            )

            scores[job] = (
                (1 - weight) * base_probability
                + weight * context_probability
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )[:top_k]

        return [
            {
                "occupation_id": job,
                "label": job,
                "probability": probability
            }
            for job, probability in ranked
        ]

    def save(self, filename):
        """Save first-order and second-order transition counts."""
        import json
        from pathlib import Path

        payload = {
            "min_context_support": self.min_context_support,
            "smoothing": self.smoothing,
            "first_order": {
                origin: dict(counts)
                for origin, counts in self.first_order.transitions.items()
            },
            "context_counts": [
                {
                    "previous": previous,
                    "current": current,
                    "destinations": dict(counts)
                }
                for (previous, current), counts
                in self.context_counts.items()
            ]
        }

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, filename):
        """Load a trained second-order model."""
        import json
        from pathlib import Path

        payload = json.loads(
            Path(filename).read_text(encoding="utf-8")
        )

        model = cls(
            min_context_support=payload["min_context_support"],
            smoothing=payload["smoothing"]
        )

        model.first_order.transitions = defaultdict(
            Counter,
            {
                origin: Counter(counts)
                for origin, counts
                in payload["first_order"].items()
            }
        )

        model.context_counts = defaultdict(Counter)

        for item in payload["context_counts"]:
            key = (item["previous"], item["current"])
            model.context_counts[key] = Counter(
                item["destinations"]
            )

        return model
