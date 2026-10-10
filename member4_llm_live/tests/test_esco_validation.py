import json
from pathlib import Path
import pytest
from src.esco_validation import load_esco_skills, validate_question_esco
from src.quiz_engine import load_questions

BASE = Path(__file__).resolve().parents[1]

def test_skills_are_in_official_esco_121():
    questions = load_questions(BASE / 'data/questions.json')
    assert validate_question_esco(questions, BASE / 'data/esco_skills_en.csv')
    assert len(set(q['esco_id'] for q in questions)) == 5

def test_no_placeholders_and_unique_uris():
    q = load_questions(BASE / 'data/questions.json')
    assert all(x['esco_id'].startswith('http://data.europa.eu/esco/skill/') for x in q)
    assert all('ESCO_PENDING' not in x['esco_id'] for x in q)

def test_reject_unknown_skill_id():
    q = load_questions(BASE / 'data/questions.json')
    q[0]['esco_id'] = 'not-an-esco-skill'
    with pytest.raises(ValueError, match='Unknown ESCO'):
        validate_question_esco(q, BASE / 'data/esco_skills_en.csv')

def test_source_has_actual_labels():
    d = load_esco_skills(BASE / 'data/esco_skills_en.csv')
    assert d['http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742'] == 'SQL'
