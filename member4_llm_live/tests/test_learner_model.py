import pytest
from src.learner_model import bkt_update, estimate_skill_knowledge, predict_correct_probability

HISTORY = [
    {"learner_id": "L1", "question_id": "q2", "esco_id": "ESCO_PENDING_SQL", "correct": 0, "timestamp": "2026-10-10T10:02:00Z"},
    {"learner_id": "L1", "question_id": "q1", "esco_id": "ESCO_PENDING_SQL", "correct": 1, "timestamp": "2026-10-10T10:01:00Z"},
    {"learner_id": "L2", "question_id": "q3", "esco_id": "ESCO_PENDING_PYTHON", "correct": 1, "timestamp": "2026-10-10T10:00:00Z"},
]

def test_posterior_correct_increases_mastery():
    assert bkt_update(0.3, 1) > 0.3

def test_posterior_wrong_decreases_mastery():
    assert bkt_update(0.3, 0) < 0.3

def test_filter_learners():
    assert list(estimate_skill_knowledge("L1", HISTORY)) == ["ESCO_PENDING_SQL"]

def test_chronological_order():
    expected = bkt_update(bkt_update(0.3, 1), 0)
    actual = estimate_skill_knowledge("L1", HISTORY)["ESCO_PENDING_SQL"]
    assert actual == round(expected, 4)

def test_accuracy_baseline():
    assert estimate_skill_knowledge("L1", HISTORY, method="accuracy") == {"ESCO_PENDING_SQL": 0.5}

def test_unassessed_omitted():
    assert estimate_skill_knowledge("unknown", HISTORY) == {}

def test_bad_answer_rejected():
    invalid = [{**HISTORY[0], "correct": 2}]
    with pytest.raises(ValueError):
        estimate_skill_knowledge("L1", invalid)

def test_observation_vs_mastery():
    assert predict_correct_probability(0.3) == pytest.approx(0.41)
