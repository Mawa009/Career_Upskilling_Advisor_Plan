# Member 3: Career Transition Engine (JobHop v2)

Self-contained hackathon baseline. **Not yet fitted on actual JobHop data**: download it first. Follows the plan's `predict_next_occupations(occupation_id)` -> `[{occupation_id,label,probability}]` contract.

## Setup (Windows PowerShell, inside this folder)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download BOTH official JobHop v2 splits into `data/`:

- https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_v2_train.parquet
- https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_v2_test.parquet

Then run:

```powershell
python -m src.preprocessing --data data/JobHop_v2_train.parquet --out models/transition_counts.json
python -m src.evaluation --model models/transition_counts.json --test data/JobHop_v2_test.parquet
```

Interactive demo (replace sample origin with an actual code printed in model):

```powershell
python -c "from src.career_paths import predict_next_occupations; print(predict_next_occupations('2512.2'))"
```

## Critical evaluation notes

- Check that official test/train person IDs are disjoint before trusting evaluation.
- Evaluation uses the **last adjacent pair of each held-out person's observed career**. It never fits the model to test persons.
- Outcomes represent historical transitions, NOT likelihood that an applicant succeeds or that a job is available.
- Same-quarter different occupations are excluded because their chronological order cannot be established. Distinct-start overlapping employments remain a limitation.
- Probabilities are unconditional on individual features and only sum to 1 across *all observed successors*; shown top five need not sum to 1.
- ESCO codes may need translation to labels/URIs by Member 2, because JobHop codes are not ESCO URIs.
- If the origin has fewer than min-support transitions, return `[]`, not invented probabilities.
- This package does not contain an API server yet; integrate a web endpoint after examining the team's actual repo architecture.

## Data license

JobHop v2 (Ghent University/VDAB), CC BY 4.0. Cite the JobHop dataset authors and v2 paper; source: https://huggingface.co/datasets/aida-ugent/JobHop . Do not commit raw datasets to repo.
