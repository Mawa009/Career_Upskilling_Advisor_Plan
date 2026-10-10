"""Generate the Colab notebook from the repo sources (single source of truth)."""
import json
from pathlib import Path

root = Path(__file__).parent
ext = (root / "src/extraction.py").read_text(encoding="utf-8")
evl = (root / "eval/evaluate.py").read_text(encoding="utf-8")
pre = (root / "eval/prelabel.py").read_text(encoding="utf-8")
gold = (root / "eval/sample_gold.jsonl").read_text(encoding="utf-8")
conv = (root / "eval/convert_annotations.py").read_text(encoding="utf-8")


def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


cells = [
    md("# Member 1 - Skill Extraction (v2)\n"
       "Run top to bottom on a **GPU** runtime (Runtime > Change runtime type > T4). "
       "Contract: `extract_skills(job_text) -> list[str]`; richer output via `extract_detailed`.\n\n"
       "Cells: 1 install, 2 ESCO upload, 3-5 write modules, 6 build extractor, 7 demo, 8 export mocks, 9 evaluate (Experiment A)."),
    code("!pip install -q transformers accelerate\n"
         "# torch ships with Colab. flashtext is no longer needed (replaced by a built-in trie)."),
    code("from pathlib import Path\nfrom google.colab import files\n"
         "if not Path('skills_en.csv').exists():\n"
         "    print('Upload ESCO v1.2.1 skills_en.csv (English classification CSV)')\n"
         "    files.upload()\n"
         "assert Path('skills_en.csv').exists(), 'skills_en.csv missing'\n"
         "!mkdir -p src eval\n!touch src/__init__.py eval/__init__.py"),
    code("%%writefile src/extraction.py\n" + ext),
    code("%%writefile eval/evaluate.py\n" + evl),
    code("%%writefile eval/prelabel.py\n" + pre + "\n"),
    code("%%writefile eval/convert_annotations.py\n" + conv + "\n"),
    code("import sys, time\nsys.path.insert(0, '.')\n"
         "import torch\nfrom src.extraction import SkillExtractor, ExtractorConfig, to_handoff\n"
         "t0 = time.time()\nextractor = SkillExtractor('skills_en.csv')\n"
         "print('device       :', extractor.skill_tagger.device)\n"
         "print('knowledge mdl:', 'on' if extractor.knowledge_tagger else 'OFF (see warning above)')\n"
         "print('ESCO records :', extractor.esco_records, '| ESCO phrases:', len(extractor.esco_lookup))\n"
         "print(f'load time    : {time.time()-t0:.0f}s')"),
    code("import pandas as pd\n"
         "example = ('We need a Data Analyst proficient in Python, SQL and Power BI. '\n"
         "           'Build dashboards, analyze customer data and perform statistical analysis. '\n"
         "           'Experience with TensorFlow, LangChain and Next.js is a plus.')\n"
         "skills = extractor.extract_detailed(example)\n"
         "pd.DataFrame([s.to_dict() for s in skills]).drop(columns=['spans'])"),
    code("import json, pathlib\nfrom src.extraction import to_handoff\n"
         "pathlib.Path('mock').mkdir(exist_ok=True)\n"
         "json.dump([s.text for s in skills], open('mock/extracted_skills.json','w'), indent=2, ensure_ascii=False)\n"
         "json.dump(to_handoff(skills), open('mock/extracted_skills_detailed.json','w'), indent=2, ensure_ascii=False)\n"
         "print(open('mock/extracted_skills.json').read())"),
    md("## Experiment A - evaluation\n"
       "`sample_gold.jsonl` below is a **5-ad synthetic dev set** for smoke-testing only. "
       "For the real result, hand-label ~50 ads (use `eval/prelabel.py` to draft) and point `GOLD` at that file."),
    code("%%writefile eval/sample_gold.jsonl\n" + gold),
    code("from eval.evaluate import load_gold, run_ablation, format_table\n"
         "GOLD = 'eval/sample_gold.jsonl'   # <- replace with your 50-ad file\n"
         "rows = load_gold(GOLD)\n"
         "table, details = run_ablation(extractor, rows)\n"
         "print(format_table(table))\n"
         "_, fps, fns = details['+ tech lexicon (full)']\n"
         "print('\\nTop false positives:', fps.most_common(12))\n"
         "print('Top false negatives:', fns.most_common(12))"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"name": "Member1_Skill_Extraction_v2.ipynb", "gpuType": "T4"},
      "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = root / "Member1_Skill_Extraction_v2.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, f"({len(cells)} cells)")
