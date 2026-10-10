from pathlib import Path
import json
from src.quiz_engine import load_questions
from src.esco_validation import load_esco_skills
from src.question_audit import audit_question_bank
base=Path(__file__).resolve().parent
report=audit_question_bank(load_questions(base/'data/questions.json'),load_esco_skills(base/'data/esco_skills_en.csv'))
(base/'outputs/question_bank_audit.json').write_text(json.dumps(report,indent=2))
print(f"{report['question_count']} questions, {report['skill_count']} ESCO skills; {len(report['failures'])} structural failures; {len(report['warnings'])} review warnings")
if report['failures']:raise SystemExit(1)
