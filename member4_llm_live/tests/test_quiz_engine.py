from pathlib import Path
import pytest
from src.quiz_engine import load_questions, select_questions, grade_answer, make_interaction

QUESTIONS = load_questions(Path(__file__).resolve().parents[1] / 'data' / 'questions.json')


def test_question_bank_is_valid():
    assert len(QUESTIONS) == 60


def test_select_unique_balanced_questions():
    skills = sorted({q['esco_id'] for q in QUESTIONS})
    selected = select_questions(QUESTIONS, skills, 3, seed=7)
    assert len(selected) == 15
    assert len({q['question_id'] for q in selected}) == 15
    assert all(sum(q['esco_id'] == skill for q in selected) == 3 for skill in skills)


def test_correct_and_wrong_answers():
    q = QUESTIONS[0]
    assert grade_answer(q, q['correct_answer']) == 1
    assert grade_answer(q, (q['correct_answer'] + 1) % len(q['options'])) == 0


def test_history_matches_contract():
    q = QUESTIONS[0]
    event = make_interaction('test_learner', q, q['correct_answer'], timestamp='2026-10-10T10:00:00Z')
    assert event == {'learner_id': 'test_learner', 'question_id': q['question_id'],
                     'esco_id': q['esco_id'], 'correct': 1, 'timestamp': '2026-10-10T10:00:00Z'}


def test_bad_option_rejected():
    with pytest.raises(ValueError):
        grade_answer(QUESTIONS[0], -1)
