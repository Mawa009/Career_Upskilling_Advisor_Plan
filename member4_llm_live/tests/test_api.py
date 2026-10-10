from fastapi.testclient import TestClient
from api import app, SKILLS, QUESTIONS

client = TestClient(app)


def sample_skills():
    q = next(x for x in QUESTIONS if x.get('esco_mapping_status') == 'direct')
    return [{'esco_id': q['esco_id'], 'label': q['skill_label'], 'importance': 0.8}]


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['esco_version'] == '1.2.1'


def test_coverage():
    r = client.post('/v1/skills/coverage', json={'job_skills': sample_skills()})
    assert r.status_code == 200
    assert r.json()['coverage'][0]['coverage'] == 'assessable'


def test_start_no_leak_and_submit():
    r = client.post('/v1/assessments/start', json={'job_skills': sample_skills(), 'questions_per_skill': 3})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert len(payload['questions']) == 3
    assert all('correct_answer' not in q and 'explanation' not in q for q in payload['questions'])
    response = {'assessment_id': payload['assessment_id'],
                'answers': [{'question_id': q['question_id'], 'selected_option': 0} for q in payload['questions']]}
    result = client.post('/v1/assessments/submit', json=response)
    assert result.status_code == 200, result.text
    estimate = result.json()['learner_estimates']
    assert set(estimate) == {sample_skills()[0]['esco_id']}
    assert 0 <= next(iter(estimate.values())) <= 1
    assert client.post('/v1/assessments/submit', json=response).status_code == 409


def test_unknown_skill_rejected():
    r = client.post('/v1/assessments/start', json={'job_skills': [{'esco_id': 'fake_id'}]})
    assert r.status_code == 422


def test_incomplete_answers_rejected():
    r = client.post('/v1/assessments/start', json={'job_skills': sample_skills()}).json()
    bad = client.post('/v1/assessments/submit', json={'assessment_id': r['assessment_id'], 'answers': []})
    assert bad.status_code == 422


def test_direct_history():
    from datetime import datetime, timezone
    s = sample_skills()[0]['esco_id']
    r = client.post('/v1/estimates/from-history', json={'learner_id': 'L1', 'interaction_history': [
        {'learner_id': 'L1', 'question_id': 'T1', 'esco_id': s, 'correct': 1, 'timestamp': datetime.now(timezone.utc).isoformat()}
    ]})
    assert r.status_code == 200, r.text
    assert s in r.json()['learner_estimates']
