"""Member 4 learner model: transparent MVP estimates, not validated competence scores.

Team contract: estimate_skill_knowledge(learner_id, interaction_history) ->
{esco_id: mastery_float}. BKT parameters below are illustrative until fitted.
"""
from collections import defaultdict
from datetime import datetime
from typing import Any

DEFAULT_BKT = {"prior": 0.30, "guess": 0.20, "slip": 0.10, "learn": 0.0}


def _validate_probability(value: float, name: str) -> float:
    v = float(value)
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return v


def _group_history(learner_id: str, interaction_history: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for event in interaction_history:
        if event.get("learner_id") != learner_id:
            continue
        for field in ("question_id", "esco_id", "correct", "timestamp"):
            if field not in event:
                raise ValueError(f"Interaction missing {field}")
        if type(event["correct"]) not in (int, bool) or event["correct"] not in (0, 1):
            raise ValueError("correct must be 0 or 1")
        try:
            datetime.fromisoformat(str(event["timestamp"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO 8601") from exc
        if not isinstance(event["esco_id"], str) or not event["esco_id"]:
            raise ValueError("esco_id must be a nonempty string")
        grouped[event["esco_id"]].append(event)
    for skill in grouped:
        grouped[skill].sort(key=lambda e: e["timestamp"])
    return dict(grouped)


def smoothed_accuracy(answers: list[int], alpha: float = 1.0, beta: float = 1.0) -> float:
    """Beta-Bernoulli posterior mean. NOT a validated mastery probability."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not answers:
        raise ValueError("Unassessed skills should be omitted")
    return (sum(answers) + alpha) / (len(answers) + alpha + beta)


def bkt_update(prior: float, correct: int, guess: float = 0.2,
               slip: float = 0.1, learn: float = 0.0) -> float:
    """BKT posterior, then learning transition. For assessment-only, learn=0.

    Parameters are illustrative defaults; estimated values require training data.
    """
    prior = _validate_probability(prior, "prior")
    guess = _validate_probability(guess, "guess")
    slip = _validate_probability(slip, "slip")
    learn = _validate_probability(learn, "learn")
    if correct not in (0, 1):
        raise ValueError("correct must be 0 or 1")
    if correct:
        numerator = prior * (1 - slip)
        denominator = numerator + (1 - prior) * guess
    else:
        numerator = prior * slip
        denominator = numerator + (1 - prior) * (1 - guess)
    # If parameters make the observation impossible, no Bayesian update is defined.
    if denominator == 0:
        raise ValueError("Observed response has zero probability under these BKT parameters")
    posterior = numerator / denominator
    return posterior + (1 - posterior) * learn


def estimate_skill_knowledge(learner_id: str,
                             interaction_history: list[dict[str, Any]],
                             method: str = "bkt",
                             bkt_params: dict[str, float] | None = None) -> dict[str, float]:
    """Contract output for Member 5; only return observed skill IDs.

    method='bkt' uses illustrative fixed parameters unless supplied;
    method='accuracy' uses smoothed accuracy. Not job-readiness certified.
    """
    grouped = _group_history(learner_id, interaction_history)
    if method not in {"bkt", "accuracy"}:
        raise ValueError("method must be 'bkt' or 'accuracy'")
    params = {**DEFAULT_BKT, **(bkt_params or {})}
    if method == "bkt":
        for name, value in params.items():
            _validate_probability(value, name)
    output = {}
    for skill, events in grouped.items():
        answers = [int(e["correct"]) for e in events]
        if method == "accuracy":
            estimate = smoothed_accuracy(answers)
        else:
            estimate = params["prior"]
            for answer in answers:
                estimate = bkt_update(estimate, answer, params["guess"], params["slip"], params["learn"])
        output[skill] = round(estimate, 4)
    return output


def predict_correct_probability(prior_mastery: float, guess: float = 0.2,
                                slip: float = 0.1) -> float:
    """BKT observation probability: P(correct)=P(L)*(1-slip)+(1-P(L))*guess."""
    p = _validate_probability(prior_mastery, "prior_mastery")
    g = _validate_probability(guess, "guess")
    s = _validate_probability(slip, "slip")
    return p * (1 - s) + (1 - p) * g
