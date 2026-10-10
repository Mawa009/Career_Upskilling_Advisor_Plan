# FINAL API HANDOFF

For the final integrated HTTP API, request/response contracts, and deployment limitations, start with **[FINAL_HANDOFF.md](FINAL_HANDOFF.md)**. The commands are `python -m uvicorn api:app --host 127.0.0.1 --port 8000` and `python -m streamlit run app.py` (in separate terminals).

---

# Member 4 — Learner Model Starter

A runnable, dependency-free **demonstration** of the Member 4 module from the project plan. It is not a trained or validated job-readiness instrument.

## Quick start
1. Install Python 3.10+ (no additional packages required for the demo).
2. Open a terminal in this folder.
3. Run `python demo.py`.
4. Inspect `outputs/learner_estimates.json`.
5. Optional: install `pytest` and run `python -m pytest -q`.
6. Optional: open `notebooks/04_learner_model.ipynb` with Jupyter or VS Code and run all cells.

## Contract
`estimate_skill_knowledge(learner_id, interaction_history)` returns `{esco_id: mastery_value}`. Default method is `bkt`; use `method="accuracy"` for the smoothed baseline. Only assessed skills are returned. Chronological interactions are recorded as learner ID, question ID, ESCO ID, binary correctness and ISO timestamp.

## Dataset
`data/questions.json`: 36 hand-authored *starter* questions (12 each for SQL, Python and data visualization); subject-matter review recommended before a real evaluation. `data/interactions.json` and `.csv`: illustrative synthetic interactions from one fake learner, for integration only. Not suitable for fitting parameters or reporting research metrics.

## Important warnings
- `ESCO_PENDING_*` strings are **not real ESCO IDs**. Replace consistently after Member 2 supplies the chosen ESCO version and mappings.
- BKT parameters are fixed illustrative values: prior=.30, guess=.20, slip=.10, learn=0. They were **not fitted**.
- A quiz answer alone is not treated as teaching a skill, hence learn=0 by default.
- Small quiz banks, question difficulty, cheating and guessing can bias estimates.
- Predicting correctness is different from establishing occupational mastery.
- With only this synthetic sample, it would be misleading to report AUC or calibration. Acquire real held-out user response data, split by learner, and compare models.
- Each quiz item is tagged with one broad skill for the MVP; topic tags are included for future subskill analysis.

## Research foundations
- pyBKT: https://arxiv.org/abs/2105.00385 (https://github.com/CAHLR/pyBKT)
- simpleKT: https://arxiv.org/abs/2302.06881
- pyKT toolkit: https://github.com/pykt-team/pykt-toolkit

## Next development tasks
(1) Freeze ESCO version and IDs with Member 2. (2) Human-review questions, keys and skill alignment. (3) Collect sequences from multiple consenting test users and hold out users for evaluation. (4) Fit BKT parameters using pyBKT and evaluate next-response predictions plus calibration. (5) Integrate the JSON with Member 5.

## Interactive Streamlit quiz (new)
1. Install Streamlit: `python -m pip install streamlit`
2. Run in project root: `python -m streamlit run app.py`
3. In the browser, select skills and a question count, then start the assessment.
4. Submit one answer per question. When finished, review smoothed accuracy and illustrative BKT estimates.
5. Download `learner_estimates.json` to hand off to Member 5; optionally download the interaction history for analysis.

Quiz sessions are stored in Streamlit session state, not a database. Refresh/reconnect may reset the assessment. The app does not automatically persist individual histories, and download files should be handled with learner consent and care. Do not deploy this demo for real employment screening. Test the question bank with subject experts before presenting it to real learners.


## ESCO v1.2.1 integration

The quiz uses *official ESCO concept URIs* as `esco_id` fields, rather than `ESCO_PENDING_*` placeholders. The frozen `data/esco_skills_en.csv` was extracted from the team's English ESCO v1.2.1 CSV download. The `src/esco_validation.py` module checks quiz IDs and canonical labels against the CSV when Streamlit starts. Run `python -m pytest -q` to verify the integration.

Mapping: SQL → SQL (direct); Python → Python (computer programming) (direct); Data visualization → data visualisation software (**provisional**). The current chart-literacy questions do not assess hands-on use of visualization software, so this last URI should be reviewed with Member 2 and the question bank should eventually assess software-specific tasks.

No change to the agreed Member 4 output: `estimate_skill_knowledge(...)` still returns `{esco_id: float}`. The demo outputs and sample interactions now carry ESCO URIs as well. **Do not interpret fixed BKT parameter estimates as calibrated mastery.**


## Job-driven assessments (Member 2 → Member 4)

Launch using `python -m streamlit run app.py`, select **Job-driven (Member 2)**, then paste or upload a JSON skill list. Example: `data/job_skills_example.json`.

Member 2 contract: a JSON array of objects with `esco_id` (official ESCO v1.2.1 URI), `label`, `importance` in [0,1]. A wrapper of `{ "job_skills": [...] }` also works; `d` may be used as the importance key. The module de-duplicates ESCO skills, keeping the highest importance. Unknown IDs are rejected for assessment and reported in the coverage summary. Known skills without questions are marked `no_question_bank`.

The app excludes provisional mappings by default (especially chart-literacy questions currently associated with ESCO data visualisation software). A clearly warned opt-in allows demonstrations; such results are **not** validated ESCO proficiency estimates.

Only assessed ESCO IDs appear in `learner_estimates.json` (the agreed Member 5 interface). Uncovered skills are NOT assigned 0. The quiz retains illustrative BKT parameters and does not prove mastery or job readiness. Add a validated question bank and fit the model before expanding claims.

Run `python -m pytest -q` to test contracts and the existing pipeline.


## Expanded question bank (60 questions / 5 ESCO concepts)
- Direct: SQL, Python (computer programming), statistics, project management (12 questions each).
- Provisional: data visualisation software (12 chart-literacy questions; excluded by default).
- Existing questions preserved; new stats and PM questions authored as draft MCQs with explanatory answer rationales.
- Run `python audit_questions.py` to save a JSON structural/ESCO consistency report.
- Run `python -m pytest -q` to verify regressions.
- IMPORTANT: Passing structural tests is **not** expert content validation, item analysis, discrimination, reliability, calibration or evidence of professional competence. Every item is marked `draft_expert_review_required`; seek independent subject-matter review and pilot response data before deploying as a validated assessment.
- ESCO labels and IDs are from the user's frozen `skills_en.csv` v1.2.1, not a claim that ESCO endorses the quiz.

## Universal ESCO fallback (hackathon scope)
For Member 2 job skills with no question bank, job-driven mode now offers a **self-report familiarity** selector. It supports any skill ID that appears in the frozen ESCO v1.2.1 skills file. The JSON exported to Member 5 mixes experimental BKT values from quizzes with user-provided 0–1 ratings. Download `estimate_sources.json` alongside it to distinguish `experimental_bkt` from `self_report`. Neither is validated occupational competency. Absent ratings remain absent (not zero); unknown ESCO identifiers are never exported. The 60 curated questions and provisional visualization mapping remain unchanged. Do not describe this as quiz coverage for all ESCO skills.


## Optional on-demand LLM questions
1. Install: `python -m pip install -r requirements-llm.txt`.
2. In **PowerShell for the current terminal only**, run `$env:GEMINI_API_KEY="YOUR_KEY"`. Do **not** share or commit the real key.
3. Optional: `$env:GEMINI_MODEL="gemini-2.5-flash-lite"`. API calls incur separate usage charges.
4. Start: `python -m streamlit run app.py`. In Job-driven mode check **Generate missing quiz questions with AI**.
5. LLM calls happen once when starting an assessment for up to 3 uncovered skills by default (adjustable, max 8). Each generates `per_skill` questions, not per Streamlit rerun. If a generation fails, the app retains self-assessment fallback.
6. `estimate_sources.json` marks LLM-question-based estimates as `llm_generated_quiz_experimental_bkt`. These are **unverified**; structural validation does not prove answer accuracy, psychometric validity, or occupational competence.
7. For a real deployment add caching by skill+ESCO version+prompt revision, moderation and independent key review, budget limits, and persistent learner data policy.


## Gemini free-tier setup (Windows PowerShell)

Install: `python -m pip install -r requirements-llm.txt`

Set a key in the current terminal: `$env:GEMINI_API_KEY="YOUR_KEY"`

Optional model override: `$env:GEMINI_MODEL="gemini-2.5-flash-lite"`

Start: `python -m streamlit run app.py`. In job-driven mode enable AI generation.

The app generates questions per missing ESCO skill at assessment start, and uses the already generated items for that assessment. Requests consume Gemini API quota. Quota and eligible free models depend on your Google AI Studio project. The API may return 429 on quota exhaustion.

Never commit `.env`, `.streamlit/secrets.toml` or the actual API key. Free-tier input may be used by the provider to improve products; avoid sending personal learner information to generation requests.
