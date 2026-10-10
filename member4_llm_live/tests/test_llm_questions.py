import pytest
from src.llm_questions import validate_generated, generate_skill_questions

GOOD = [{'prompt':'Which operation correctly checks whether two values are equal?',
         'options':['==','!=','>=','<='], 'correct_answer':0, 'difficulty':'easy',
         'topic':'Comparison operators','explanation':'Equality uses ==.'}]

def test_generated_valid():
    q=validate_generated(GOOD,'uri','Test',1)
    assert q[0]['esco_id']=='uri' and q[0]['source']=='llm_unverified'

def test_invalid_key_rejected():
    with pytest.raises(ValueError): validate_generated([{**GOOD[0], 'correct_answer':4}], 'uri','Test',1)

def test_duplicate_option_rejected():
    with pytest.raises(ValueError): validate_generated([{**GOOD[0], 'options':['a','a','c','d']}], 'uri','Test',1)

def test_missing_key_no_api_call(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(RuntimeError,match='GEMINI_API_KEY'): generate_skill_questions('uri','Test')
