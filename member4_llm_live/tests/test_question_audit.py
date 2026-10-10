import json
from pathlib import Path
from collections import Counter
from src.quiz_engine import load_questions, select_questions
from src.esco_validation import load_esco_skills
from src.question_audit import audit_question_bank
from src.job_assessment import make_coverage_plan
ROOT=Path(__file__).resolve().parents[1]

def test_bank_has_five_esco_skills_and_60_questions():
    q=load_questions(ROOT/'data/questions.json')
    assert len(q)==60
    assert len(set(x['esco_id'] for x in q))==5

def test_structural_audit_passes():
    q=load_questions(ROOT/'data/questions.json')
    report=audit_question_bank(q,load_esco_skills(ROOT/'data/esco_skills_en.csv'))
    assert report['passed'], report['failures']
    assert len(report['warnings'])>=12

def test_job_driven_new_skills_covered():
    q=load_questions(ROOT/'data/questions.json')
    lookup=load_esco_skills(ROOT/'data/esco_skills_en.csv')
    new_ids=[x['esco_id'] for x in q if x['question_id'] in ('STA_001','PM_001')]
    plan=make_coverage_plan([{'esco_id':uri,'label':lookup[uri],'importance':.8} for uri in new_ids],q,lookup)
    assert [x['coverage'] for x in plan]==['assessable','assessable']
    assert len(select_questions(q,new_ids,per_skill=12,seed=7))==24

def test_provisional_visualization_excluded():
    q=load_questions(ROOT/'data/questions.json')
    lookup=load_esco_skills(ROOT/'data/esco_skills_en.csv')
    uri=next(x['esco_id'] for x in q if x['question_id']=='VIS_001')
    plan=make_coverage_plan([{'esco_id':uri,'label':lookup[uri],'importance':.8}],q,lookup)
    assert plan[0]['coverage']=='provisional_mapping'
