"""Structural and ESCO-source validation for the expanded question bank."""
from collections import Counter, defaultdict
import re

VALID_LEVELS = {'easy', 'medium', 'advanced'}
VALID_STATUSES = {'direct', 'provisional_broader_than_quiz'}

def audit_question_bank(questions, esco_lookup, min_per_skill=12):
    failures=[]; warnings=[]; ids=set(); counts=Counter(); levels=defaultdict(Counter); labels={}; statuses={}
    for q in questions:
        ident=q.get('question_id','<missing>')
        if ident in ids: failures.append(f'{ident}: duplicate question ID')
        ids.add(ident)
        uri=q.get('esco_id','')
        counts[uri]+=1; levels[uri][q.get('difficulty')]+=1
        if uri not in esco_lookup: failures.append(f'{ident}: unknown ESCO URI')
        if uri in esco_lookup and q.get('esco_label')!=esco_lookup[uri]: failures.append(f'{ident}: ESCO label differs from source')
        if q.get('difficulty') not in VALID_LEVELS: failures.append(f'{ident}: invalid difficulty')
        if q.get('esco_mapping_status') not in VALID_STATUSES: failures.append(f'{ident}: unapproved mapping status')
        opts=q.get('options',[])
        if not isinstance(opts,list) or len(opts)!=4 or any(not isinstance(x,str) or not x.strip() for x in opts):failures.append(f'{ident}: four nonempty options required')
        elif len({re.sub(r'\s+',' ',x.strip()).casefold() for x in opts}) != 4:failures.append(f'{ident}: duplicate options')
        if type(q.get('correct_answer')) is not int or not 0<=q['correct_answer']<len(opts): failures.append(f'{ident}: invalid answer index')
        if not str(q.get('prompt','')).strip():failures.append(f'{ident}: empty prompt')
        if uri in statuses and statuses[uri]!=q.get('esco_mapping_status'):failures.append(f'{ident}: conflicting mapping statuses')
        statuses[uri]=q.get('esco_mapping_status')
        if uri in labels and labels[uri]!=q.get('skill_label'): failures.append(f'{ident}: conflicting local skill labels')
        labels[uri]=q.get('skill_label')
        if q.get('content_review')!='approved_by_subject_expert':warnings.append(f'{ident}: content awaits independent expert review')
        if q.get('esco_mapping_status')!='direct':warnings.append(f'{ident}: provisional mapping; excluded by default')
    for uri,count in counts.items():
        if count<min_per_skill:failures.append(f'{uri}: fewer than {min_per_skill} questions')
        if set(levels[uri])!=VALID_LEVELS: failures.append(f'{uri}: missing a difficulty level')
    return {'passed':not failures,'question_count':len(questions),'skill_count':len(counts),'per_skill':dict(counts),'by_difficulty':{k:dict(v) for k,v in levels.items()},'failures':failures,'warnings':warnings,'validation_scope':'structural and ESCO-label consistency only; not psychometric validation'}
