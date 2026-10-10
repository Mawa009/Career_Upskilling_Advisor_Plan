"""Pure quiz utilities; Streamlit code lives in app.py."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random


def load_questions(path):
    questions = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = set()
    for question in questions:
        required = {'question_id', 'esco_id', 'skill_label', 'prompt', 'options', 'correct_answer', 'difficulty'}
        if not required.issubset(question):
            raise ValueError(f"Missing fields in question {question.get('question_id')}")
        if question['question_id'] in ids:
            raise ValueError('Duplicate question ID')
        ids.add(question['question_id'])
        if not question['esco_id'] or not question['skill_label']:
            raise ValueError('Missing skill identity')
        if not isinstance(question['options'], list) or len(question['options']) < 2:
            raise ValueError('Each question needs answer options')
        if type(question['correct_answer']) is not int or not 0 <= question['correct_answer'] < len(question['options']):
            raise ValueError('Invalid correct_answer index')
    return questions


def select_questions(questions, skill_ids, per_skill=3, seed=None):
    """Choose unique questions; stratify by difficulty when the bank permits."""
    if per_skill <= 0:
        raise ValueError('per_skill must be positive')
    rng = random.Random(seed)
    selected = []
    for skill in skill_ids:
        available = [q for q in questions if q['esco_id'] == skill]
        if len(available) < per_skill:
            raise ValueError(f'Not enough questions for {skill}')
        levels = ['easy', 'medium', 'hard']
        groups = {level: [q for q in available if q['difficulty'] == level] for level in levels}
        for group in groups.values():
            rng.shuffle(group)
        picked = []
        # Spread across difficulty levels before selecting extras.
        while len(picked) < per_skill:
            made_progress = False
            for level in levels:
                if groups[level] and len(picked) < per_skill:
                    picked.append(groups[level].pop())
                    made_progress = True
            if not made_progress:
                remaining = [q for q in available if q not in picked]
                rng.shuffle(remaining)
                picked.extend(remaining[:per_skill - len(picked)])
        selected.extend(picked)
    rng.shuffle(selected)
    return selected


def grade_answer(question, selected_option):
    """The option index is compared with the separately stored answer key."""
    if type(selected_option) is not int or not 0 <= selected_option < len(question['options']):
        raise ValueError('Invalid answer option')
    return int(selected_option == question['correct_answer'])


def make_interaction(learner_id, question, selected_option, timestamp=None):
    result = {
        'learner_id': learner_id,
        'question_id': question['question_id'],
        'esco_id': question['esco_id'],
        'correct': grade_answer(question, selected_option),
        'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
    }
    return result
