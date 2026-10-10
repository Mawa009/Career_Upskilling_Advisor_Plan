"""Validate Member 2 skill payloads and plan an honest ESCO-keyed assessment."""
import json
from collections import Counter


def parse_job_skills(payload):
    """Accept a JSON list or {job_skills: list}; entries need esco_id and optional label/importance."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid job skills JSON: {exc.msg}') from exc
    if isinstance(payload, dict):
        payload = payload.get('job_skills')
    if not isinstance(payload, list):
        raise ValueError('Expected a list of skills or an object containing job_skills.')
    result = {}
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or not isinstance(entry.get('esco_id'), str) or not entry['esco_id'].strip():
            raise ValueError(f'Skill entry {index + 1} must contain a nonempty esco_id string.')
        uri = entry['esco_id'].strip()
        importance = entry.get('importance', entry.get('d', 1.0))
        if isinstance(importance, bool) or not isinstance(importance, (int, float)) or not 0 <= importance <= 1:
            raise ValueError(f'Importance for {uri} must be between 0 and 1.')
        old = result.get(uri)
        if old is None or importance > old['importance']:
            result[uri] = {'esco_id': uri, 'label': str(entry.get('label') or uri), 'importance': float(importance)}
    return list(result.values())


def make_coverage_plan(job_skills, questions, esco_lookup, include_provisional=False):
    """Categorize every requested skill without inventing assessments for uncovered ESCO concepts."""
    counts = Counter(q['esco_id'] for q in questions)
    status = {}
    for q in questions:
        uri = q['esco_id']
        mapping = q.get('esco_mapping_status', 'unreviewed')
        if uri in status and status[uri] != mapping:
            raise ValueError(f'Inconsistent mapping statuses in question bank for {uri}')
        status[uri] = mapping
    plan = []
    for job_skill in job_skills:
        uri = job_skill['esco_id']
        n = counts[uri]
        mapping = status.get(uri)
        if uri not in esco_lookup:
            coverage = 'unknown_esco_id'
        elif n == 0:
            coverage = 'no_question_bank'
        elif mapping != 'direct' and not include_provisional:
            coverage = 'provisional_mapping'
        else:
            coverage = 'assessable' if mapping == 'direct' else 'assessable_provisional'
        plan.append({**job_skill, 'canonical_label': esco_lookup.get(uri),
                     'question_count': n, 'mapping_status': mapping, 'coverage': coverage})
    return plan


def assessable_ids(coverage_plan):
    return [entry['esco_id'] for entry in coverage_plan if entry['coverage'].startswith('assessable')]
