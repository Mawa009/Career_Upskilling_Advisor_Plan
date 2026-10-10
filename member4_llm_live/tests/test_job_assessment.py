import json
from pathlib import Path
import pytest
from src.esco_validation import load_esco_skills
from src.job_assessment import parse_job_skills, make_coverage_plan, assessable_ids
from src.quiz_engine import load_questions, select_questions

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = load_questions(ROOT / 'data/questions.json')
ESCO = load_esco_skills(ROOT / 'data/esco_skills_en.csv')
SQL = next(q['esco_id'] for q in QUESTIONS if q['skill_label'] == 'SQL')
PYTHON = next(q['esco_id'] for q in QUESTIONS if q['skill_label'] == 'Python')
VIZ = next(q['esco_id'] for q in QUESTIONS if q['esco_mapping_status'] != 'direct')


def test_parses_member2_contract():
    assert parse_job_skills(json.dumps([{'esco_id': SQL, 'importance': .7}]))[0]['importance'] == .7
    assert parse_job_skills({'job_skills': [{'esco_id': SQL, 'd': .4}]})[0]['importance'] == .4


def test_dedup_keeps_highest_importance():
    entries = parse_job_skills([{'esco_id': SQL, 'd': .3}, {'esco_id': SQL, 'd': .9}])
    assert len(entries) == 1 and entries[0]['importance'] == .9


@pytest.mark.parametrize('payload', ['not json', {}, [{'label':'SQL'}], [{'esco_id':SQL,'importance':2}], [{'esco_id':SQL,'importance':True}]])
def test_invalid_payload_rejected(payload):
    with pytest.raises(ValueError):
        parse_job_skills(payload)


def test_coverage_and_missing_are_not_zero():
    entries = parse_job_skills([{'esco_id': SQL}, {'esco_id': VIZ}, {'esco_id': 'http://data.europa.eu/esco/skill/not-real'}])
    plan = make_coverage_plan(entries, QUESTIONS, ESCO)
    assert [p['coverage'] for p in plan] == ['assessable','provisional_mapping','unknown_esco_id']
    assert assessable_ids(plan) == [SQL]


def test_valid_uncovered_esco_skill():
    unused = next(uri for uri in ESCO if uri not in {q['esco_id'] for q in QUESTIONS})
    plan = make_coverage_plan([{'esco_id': unused}], QUESTIONS, ESCO)
    assert plan[0]['coverage'] == 'no_question_bank'


def test_opt_in_provisional_and_question_selection():
    plan = make_coverage_plan([{'esco_id': SQL},{'esco_id': PYTHON},{'esco_id': VIZ}], QUESTIONS, ESCO, True)
    assert assessable_ids(plan) == [SQL,PYTHON,VIZ]
    selected = select_questions(QUESTIONS, assessable_ids(plan), 3, seed=2)
    assert len(selected) == 9
