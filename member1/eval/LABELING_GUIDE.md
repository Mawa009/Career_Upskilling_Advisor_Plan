# Labeling guide for the ~50 gold ads (Experiment A)

Agree on these rules BEFORE anyone labels, or scores will measure disagreement, not quality.
Everyone on the team labels the same 5 ads first and compares.

1. **Label what the ad asks the candidate to be able to do or know** - tools, technologies,
   methods, and action skills ("build dashboards", "write SQL queries"). Use the ad's own words.
2. **Tools inside action phrases: label BOTH** the phrase and the tool
   ("write SQL queries" AND "SQL"). The extractor emits both, so labeling both keeps scores fair.
3. **Coordinated verbs: expand them.** "build and maintain web apps" ->
   "build web apps" and "maintain web apps".
4. **Soft skills count** when they are stated as a requirement ("strong communication skills",
   "work independently"). Drop pure filler ("team player" boilerplate in the About-us section is optional - pick one rule and keep it).
5. **Certifications and credentials** (PMP, AWS Certified ...): decide once whether they count.
   Suggested: yes, label them (ESCO has no single skill for them, so Member 2 may leave them unlinked).
6. **Do NOT label**: job titles, company names, degrees ("Bachelor's"), years of experience,
   salary/benefits, locations, generic words ("experience", "team").
7. **Job title that names a skill** ("ML Engineer"): label it only if the body also asks for it.
8. **Each skill once per ad**, in its most complete form.

Workflow: `python -m eval.prelabel ads.txt eval/gold_draft.jsonl --esco skills_en.csv`, then edit.
Label ~10 ads blind (`--blank`) to measure how much pre-labeling biases the gold toward the model.
