"""Universal ESCO assessment coverage without pretending all skills have validated MCQs."""

def split_coverage(coverage):
    quiz_ids = [x['esco_id'] for x in coverage if x['coverage'].startswith('assessable')]
    self_rating = [x for x in coverage if x['coverage'] in ('no_question_bank', 'provisional_mapping')]
    unknown = [x for x in coverage if x['coverage'] == 'unknown_esco_id']
    return quiz_ids, self_rating, unknown


def merge_estimates(quiz_estimates, self_ratings, coverage):
    """Quiz results take precedence; self-report is explicitly flagged as unverified."""
    valid = {x['esco_id'] for x in coverage if x['coverage'] != 'unknown_esco_id'}
    result = dict(quiz_estimates)
    details = {}
    for skill in coverage:
        uri = skill['esco_id']
        if uri in quiz_estimates:
            details[uri] = {'method': 'experimental_bkt', 'estimate': quiz_estimates[uri], 'verified': False}
        elif uri in self_ratings and uri in valid:
            value = self_ratings[uri]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError('Self-ratings must be numbers from 0 to 1.')
            result[uri] = float(value)
            details[uri] = {'method': 'self_report', 'estimate': float(value), 'verified': False}
        else:
            details[uri] = {'method': 'unassessed', 'estimate': None, 'verified': False}
    return result, details
