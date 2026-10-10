"""Streamlit quiz UI for Member 4. Start with: streamlit run app.py"""
from pathlib import Path
import json
from uuid import uuid4

import streamlit as st

from src.learner_model import estimate_skill_knowledge
from src.quiz_engine import load_questions, select_questions, make_interaction
from src.esco_validation import validate_question_esco, load_esco_skills
from src.job_assessment import parse_job_skills, make_coverage_plan, assessable_ids
from src.universal_assessment import split_coverage, merge_estimates
from src.llm_questions import generate_skill_questions

BASE = Path(__file__).resolve().parent
QUESTIONS = load_questions(BASE / 'data' / 'questions.json')
validate_question_esco(QUESTIONS, BASE / 'data' / 'esco_skills_en.csv')
SKILLS = {q['esco_id']: q['skill_label'] for q in QUESTIONS}
ESCO_LOOKUP = load_esco_skills(BASE / 'data' / 'esco_skills_en.csv')
EXAMPLE_JOB_SKILLS = [
    {'esco_id': uri, 'label': label, 'importance': 0.8}
    for uri, label in list(SKILLS.items())[:2]
]

st.set_page_config(page_title='Career Skills Check | Member 4', page_icon='🎯', layout='centered')
st.title('🎯 Career Skills Check')
st.caption('Member 4 · Learner model proof of concept')
st.info('ESCO v1.2.1 integrated. Results are experimental estimates, not a certification of job readiness.')
st.caption('Curated quizzes where available; Gemini questions for missing skills if enabled; self-ratings only as fallback.')


def begin_quiz(skill_ids, per_skill, coverage=None):
    st.session_state.quiz = {
        'learner_id': f"demo_{uuid4().hex[:10]}",
        'questions': select_questions(QUESTIONS, skill_ids, per_skill),
        'history': [],
        'index': 0,
        'coverage': coverage or [],
        'self_ratings': {},
        'llm_skills': [],
    }


with st.sidebar:
    st.header('Assessment setup')
    mode = st.radio('Assessment mode', ['Job-driven (Member 2)', 'Manual skill selection'])
    per_skill = st.selectbox('Questions per skill', [3, 6, 9, 12], index=0)
    ai_enabled = st.checkbox('Generate missing quiz questions with AI (subject to free-tier limits)', value=False)
    if ai_enabled:
        st.caption('Uses GEMINI_API_KEY on the server and Gemini structured output. AI questions are unverified drafts.')
        max_ai_skills = st.slider('Max AI-generated skills per assessment', 1, 8, 3)
    if mode == 'Job-driven (Member 2)':
        st.caption('Paste Member 2 job_skills.json, or upload it. Accepted format: '
                   '[{"esco_id": "official URI", "label": "SQL", "importance": 0.8}].')
        uploaded = st.file_uploader('Upload job_skills.json', type=['json'])
        pasted = st.text_area('Or paste job skills JSON',
                              value=json.dumps(EXAMPLE_JOB_SKILLS, indent=2), height=180)
        provisional = st.checkbox('Include provisional skill mappings (experimental)', value=False)
        try:
            payload = uploaded.getvalue().decode('utf-8-sig') if uploaded is not None else pasted
            job_skills = parse_job_skills(payload)
            coverage = make_coverage_plan(job_skills, QUESTIONS, ESCO_LOOKUP, include_provisional=provisional)
            st.subheader('Skill coverage')
            for item in coverage:
                name = item['canonical_label'] or item['label']
                st.caption(f"{name}: {item['coverage']} ({item['question_count']} bank questions)")
            chosen = assessable_ids(coverage)
            unsupported = [x for x in coverage if x['coverage'] not in ('assessable', 'assessable_provisional')]
            if unsupported:
                st.info(f'{len(unsupported)} requested skill(s) have no direct quiz coverage. '
                        'Valid uncovered skills will use clearly marked self-assessment.')
            if any(x['coverage'] == 'assessable_provisional' for x in coverage):
                st.warning('Provisional mappings do not establish proficiency in the official ESCO skill.')
            error = None
        except (ValueError, UnicodeDecodeError) as exc:
            chosen, coverage, error = [], [], str(exc)
            st.error(error)
    else:
        available = {uri: label for uri, label in SKILLS.items()
                     if any(q['esco_id'] == uri and q.get('esco_mapping_status') == 'direct' for q in QUESTIONS)}
        chosen_labels = st.multiselect('Select skills to assess', list(available.values()),
                                       default=list(available.values()))
        chosen = [uri for uri, label in available.items() if label in chosen_labels]
        coverage, error = [], None
        st.caption('Manual mode offers directly mapped skills. Provisional visualization is excluded.')
    start = st.button('Start / restart assessment', type='primary', use_container_width=True,
                      disabled=bool(error) or not (chosen or (coverage and any(x['coverage'] in ('no_question_bank', 'provisional_mapping') for x in coverage))))
    if start:
        begin_quiz(chosen, per_skill, coverage)
        if ai_enabled and mode == 'Job-driven (Member 2)':
            missing = [x for x in coverage if x['coverage'] in ('no_question_bank', 'provisional_mapping')]
            if len(missing) > max_ai_skills:
                st.warning(f'Only the first {max_ai_skills} uncovered skills will be AI-assessed; others retain self-rating fallback.')
            for entry in missing[:max_ai_skills]:
                skill_id = entry['esco_id']
                label = entry['canonical_label'] or entry['label']
                try:
                    with st.spinner(f'Generating draft questions for {label}...'):
                        generated = generate_skill_questions(skill_id, label, count=per_skill)
                    st.session_state.quiz['questions'].extend(generated)
                    st.session_state.quiz['llm_skills'].append(skill_id)
                except Exception as exc:
                    st.warning(f'Could not generate questions for {label}: {exc}. Self-assessment fallback remains available.')
        st.rerun()
    st.divider()
    st.caption('Frozen ESCO 1.2.1 vocabulary. BKT parameters are illustrative, not fitted.')

if 'quiz' not in st.session_state:
    st.subheader('Ready to measure your skills?')
    st.write('Choose the skills and question count in the sidebar, then select **Start / restart assessment**.')
    st.stop()

quiz = st.session_state.quiz
count = len(quiz['questions'])
pos = quiz['index']
if count:
    st.progress(pos / count, text=f'{pos} of {count} questions answered')
else:
    st.info('No curated quiz questions for this selection. Continue with the skill self-assessment below.')

if pos < count:
    question = quiz['questions'][pos]
    st.subheader(f'Question {pos + 1} of {count}')
    st.caption(f"{question['skill_label']} · {question['difficulty'].title()} · {question.get('topic', 'General')}")
    st.write(question['prompt'])
    with st.form(key=f"answer_form_{quiz['learner_id']}_{pos}"):
        answer = st.radio('Choose one answer', options=list(range(len(question['options']))),
                          format_func=lambda i: question['options'][i], index=None)
        submit = st.form_submit_button('Submit answer', type='primary', use_container_width=True)
    if submit:
        if answer is None:
            st.warning('Select an answer before continuing.')
        else:
            quiz['history'].append(make_interaction(quiz['learner_id'], question, answer))
            quiz['index'] += 1
            st.rerun()
else:
    st.success('Assessment complete!')
    history = quiz['history']
    bkt = estimate_skill_knowledge(quiz['learner_id'], history, method='bkt')
    accuracy = estimate_skill_knowledge(quiz['learner_id'], history, method='accuracy')
    _, unrated_skills, unknown_skills = split_coverage(quiz.get('coverage', []))
    unrated_skills = [x for x in unrated_skills if x['esco_id'] not in quiz.get('llm_skills', [])]
    if unrated_skills:
        st.subheader('Self-assessment for skills without direct quiz coverage')
        st.info('These are personal familiarity ratings, NOT objectively tested mastery. They are included only as an optional hackathon fallback.')
        with st.form('self_ratings'):
            ratings = {}
            for item in unrated_skills:
                uri = item['esco_id']
                label = item['canonical_label'] or item['label']
                previous = quiz['self_ratings'].get(uri)
                selection = st.selectbox(f'{label}',
                    ['Not assessed', 'No experience (0.0)', 'Beginner (0.25)', 'Intermediate (0.5)', 'Advanced (0.75)', 'Expert (1.0)'],
                    index=0 if previous is None else [None, 0, .25, .5, .75, 1.0].index(previous),
                    key=f'self_{uri}')
                ratings[uri] = {'Not assessed': None, 'No experience (0.0)': 0.0,
                    'Beginner (0.25)': .25, 'Intermediate (0.5)': .5,
                    'Advanced (0.75)': .75, 'Expert (1.0)': 1.0}[selection]
            if st.form_submit_button('Save self-assessment'):
                quiz['self_ratings'] = {uri: val for uri, val in ratings.items() if val is not None}
                st.rerun()
    if unknown_skills:
        st.error(f'{len(unknown_skills)} IDs are not in the frozen ESCO v1.2.1 vocabulary and cannot be exported.')
    estimates, provenance = merge_estimates(bkt, quiz['self_ratings'], quiz.get('coverage', []))
    for skill_id in quiz.get('llm_skills', []):
        if skill_id in bkt:
            provenance[skill_id] = {'method': 'llm_generated_quiz_experimental_bkt',
                                    'estimate': bkt[skill_id], 'verified': False}
    if quiz.get('llm_skills'):
        st.warning('AI-written questions and answer keys are unverified. Mastery estimates are experimental, not proof of competence.')
    st.subheader('Your estimated skill profile')
    if quiz.get('coverage'):
        with st.expander('Job requirements and assessment coverage'):
            for item in quiz['coverage']:
                st.write(f"{item['canonical_label'] or item['label']} — {item['coverage']}")
    st.caption('BKT values are based on untrained illustrative parameters, and can become extreme. Smoothed accuracy is shown for comparison.')
    for skill_id in dict.fromkeys(q['esco_id'] for q in quiz['questions']):
        label = next(q['skill_label'] for q in quiz['questions'] if q['esco_id'] == skill_id)
        n = sum(event['esco_id'] == skill_id for event in history)
        correct = sum(event['correct'] for event in history if event['esco_id'] == skill_id)
        st.write(f'**{label}** — {correct}/{n} correct')
        st.caption(f"ESCO: {skill_id} · {label}")
        st.progress(accuracy[skill_id], text=f'Smoothed accuracy: {accuracy[skill_id]:.1%}')
        st.caption(f'Experimental BKT estimate: {bkt[skill_id]:.1%}')
    st.subheader('Member 5 integration output')
    st.code(json.dumps(estimates, indent=2), language='json')
    st.download_button('Download learner_estimates.json', data=json.dumps(estimates, indent=2),
                       file_name='learner_estimates.json', mime='application/json', use_container_width=True)
    st.download_button('Download estimate_sources.json', data=json.dumps(provenance, indent=2),
                       file_name='estimate_sources.json', mime='application/json', use_container_width=True)
    with st.expander('Download anonymized interaction history'):
        st.download_button('Download interaction_history.json', data=json.dumps(history, indent=2),
                           file_name='interaction_history.json', mime='application/json')
    st.caption('The app keeps responses in the current Streamlit session. Use the download button to export; it does not automatically overwrite team files.')
