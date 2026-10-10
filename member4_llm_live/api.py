"""Member 4 FastAPI service. Single-process, in-memory assessment sessions (hackathon MVP)."""
from __future__ import annotations
import os
import secrets
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.esco_validation import load_esco_skills, validate_question_esco
from src.job_assessment import parse_job_skills, make_coverage_plan
from src.quiz_engine import load_questions, select_questions, make_interaction
from src.learner_model import estimate_skill_knowledge
from src.universal_assessment import merge_estimates
from src.llm_questions import generate_skill_questions

ROOT = Path(__file__).resolve().parent
QUESTIONS = load_questions(ROOT / 'data/questions.json')
SKILLS = load_esco_skills(ROOT / 'data/esco_skills_en.csv')
validate_question_esco(QUESTIONS, ROOT / 'data/esco_skills_en.csv')
app = FastAPI(title='Member 4 ESCO Learner Assessment API', version='1.0.0',
              description='Hackathon MVP: curated/LLM quizzes and experimental BKT estimates. Not a certification.')
@app.middleware('http')
async def optional_api_token(request: Request, call_next):
    # Configure API_SERVICE_KEY before exposing beyond localhost.
    required = os.getenv('API_SERVICE_KEY')
    if required and request.url.path.startswith('/v1/'):
        supplied = request.headers.get('Authorization', '')
        if not secrets.compare_digest(supplied, f'Bearer {required}'):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={'detail': 'Invalid or missing API token'})
    return await call_next(request)

_sessions: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()
SESSION_TTL = 3600
MAX_SESSIONS = 500

class JobSkill(BaseModel):
    esco_id: str
    label: str | None = None
    importance: float = Field(default=1.0, ge=0, le=1)

class StartRequest(BaseModel):
    job_skills: list[JobSkill] = Field(min_length=1, max_length=15)
    questions_per_skill: int = Field(default=3, ge=1, le=12)
    generate_missing: bool = False
    max_ai_skills: int = Field(default=3, ge=0, le=5)
    include_provisional: bool = False

class Answer(BaseModel):
    question_id: str
    selected_option: int = Field(ge=0)

class SubmitRequest(BaseModel):
    assessment_id: str
    answers: list[Answer]
    self_ratings: dict[str, float] = Field(default_factory=dict)

class EstimateRequest(BaseModel):
    learner_id: str
    interaction_history: list[dict[str, Any]]
    method: str = 'bkt'


def _clean_expired():
    now = time.time()
    for sid in list(_sessions):
        if now - _sessions[sid]['created_at'] > SESSION_TTL:
            _sessions.pop(sid, None)


def _public_question(q):
    return {k: q.get(k) for k in ('question_id', 'esco_id', 'skill_label', 'difficulty', 'topic', 'prompt', 'options')}

@app.get('/health')
def health():
    return {'status': 'ok', 'esco_version': '1.2.1', 'question_bank_size': len(QUESTIONS),
            'unique_esco_skills': len(SKILLS), 'llm_key_configured': bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))}

@app.post('/v1/skills/coverage')
def coverage(req: StartRequest):
    parsed = parse_job_skills([x.model_dump() for x in req.job_skills])
    return {'coverage': make_coverage_plan(parsed, QUESTIONS, SKILLS, include_provisional=req.include_provisional)}

@app.post('/v1/assessments/start')
def start_assessment(req: StartRequest):
    parsed = parse_job_skills([x.model_dump() for x in req.job_skills])
    plan = make_coverage_plan(parsed, QUESTIONS, SKILLS, include_provisional=req.include_provisional)
    unknown = [x['esco_id'] for x in plan if x['coverage'] == 'unknown_esco_id']
    if unknown:
        raise HTTPException(422, detail={'error': 'unknown_esco_ids', 'ids': unknown})
    questions, generated_ids, errors = [], set(), []
    for item in plan:
        uri = item['esco_id']
        if item['coverage'].startswith('assessable'):
            n = min(item['question_count'], req.questions_per_skill)
            questions.extend(select_questions(QUESTIONS, [uri], per_skill=n))
    missing = [x for x in plan if x['coverage'] in ('no_question_bank', 'provisional_mapping')]
    if req.generate_missing:
        for item in missing[:req.max_ai_skills]:
            uri = item['esco_id']
            try:
                generated = generate_skill_questions(uri, item['canonical_label'] or item['label'],
                                                     count=req.questions_per_skill)
                questions.extend(generated)
                generated_ids.add(uri)
            except Exception as exc:
                # Do not disclose key, URL, or provider error body in public HTTP responses.
                errors.append({'esco_id': uri, 'reason': 'question_generation_unavailable',
                               'type': type(exc).__name__})
    if not questions and not missing:
        raise HTTPException(422, detail='No assessable skills requested')
    token = secrets.token_urlsafe(24)
    with _lock:
        _clean_expired()
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(503, detail='Assessment capacity exceeded; retry later')
        _sessions[token] = {'created_at': time.time(), 'questions': questions, 'coverage': plan,
                            'generated_ids': generated_ids, 'completed': False}
    return {'assessment_id': token, 'expires_in_seconds': SESSION_TTL,
            'questions': [_public_question(q) for q in questions],
            'coverage': plan, 'generation_errors': errors,
            'unassessed_esco_ids': [x['esco_id'] for x in plan if x['esco_id'] not in {q['esco_id'] for q in questions}],
            'notice': 'LLM questions and mastery estimates are unverified; self-ratings are explicitly labeled.'}

@app.post('/v1/assessments/submit')
def submit_assessment(req: SubmitRequest):
    with _lock:
        _clean_expired()
        session = _sessions.get(req.assessment_id)
        if session is None:
            raise HTTPException(404, detail='Assessment not found or expired')
        if session['completed']:
            raise HTTPException(409, detail='Assessment already submitted')
        questions = session['questions']
        keys = [q['question_id'] for q in questions]
        submitted = [a.question_id for a in req.answers]
        if len(set(submitted)) != len(submitted) or Counter(submitted) != Counter(keys):
            raise HTTPException(422, detail='Submit exactly one answer for every assessment question')
        answers = {a.question_id: a.selected_option for a in req.answers}
        if any(answers[q['question_id']] >= len(q['options']) for q in questions):
            raise HTTPException(422, detail='Answer option index out of range')
        allowed = {x['esco_id'] for x in session['coverage'] if x['esco_id'] not in {q['esco_id'] for q in questions}}
        if set(req.self_ratings) - allowed:
            raise HTTPException(422, detail='Self-ratings are only accepted for unquizzed requested ESCO skills')
        learner_id = 'assessment_' + req.assessment_id[:12]
        history = [make_interaction(learner_id, q, answers[q['question_id']]) for q in questions]
        bkt = estimate_skill_knowledge(learner_id, history)
        accuracy = estimate_skill_knowledge(learner_id, history, method='accuracy')
        try:
            estimates, sources = merge_estimates(bkt, req.self_ratings, session['coverage'])
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        for uri in session['generated_ids']:
            if uri in bkt:
                sources[uri] = {'method': 'llm_generated_quiz_experimental_bkt',
                                'estimate': bkt[uri], 'verified': False}
        session['completed'] = True
        return {'learner_estimates': estimates, 'estimate_sources': sources,
                'smoothed_accuracy': accuracy,
                'summary': {'answered_questions': len(questions), 'correct': sum(x['correct'] for x in history)},
                'notice': 'Experimental, uncalibrated BKT parameters; not certified mastery.'}

@app.post('/v1/estimates/from-history')
def estimate_from_history(req: EstimateRequest):
    if req.method not in ('bkt', 'accuracy'):
        raise HTTPException(422, detail='method must be bkt or accuracy')
    try:
        for item in req.interaction_history:
            uri = item.get('esco_id')
            if uri not in SKILLS:
                raise ValueError(f'Unknown ESCO ID: {uri}')
        values = estimate_skill_knowledge(req.learner_id, req.interaction_history, method=req.method)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return {'learner_estimates': values, 'method': req.method,
            'notice': 'Trusted internal endpoint; submit only verified graded interactions.'}
