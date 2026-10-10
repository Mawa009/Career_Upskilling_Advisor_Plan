import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluate import match_counts, norm, prf, score  # noqa: E402
from src.extraction import (  # noqa: E402
    ExtractorConfig, PhraseTrie, SkillExtractor, clean_text, segment, tokenize, to_handoff,
)


class FakeTagger:
    """Stands in for JobBERT: tags every occurrence of the given phrases."""

    def __init__(self, phrases, prob=0.9):
        self.phrases, self.prob, self.calls, self.seen = phrases, prob, 0, 0

    def tag(self, texts):
        self.calls += 1
        self.seen += len(texts)
        out = []
        for t in texts:
            spans = []
            for p in self.phrases:
                for m in re.finditer(re.escape(p), t):
                    spans.append((m.start(), m.end(), self.prob))
            out.append(spans)
        return out


def make(phrases=(), **cfg):
    return SkillExtractor(
        None, ExtractorConfig(use_knowledge_model=False, **cfg),
        skill_tagger=FakeTagger(list(phrases)), load_models=False,
    )


# ---------------------------------------------------------------- segmentation
def test_segment_offsets_roundtrip():
    text = "Requirements:\n- Python, SQL.\n• Git\n\nBuild things. Test things; ship e.g. Docker images."
    segs = segment(text)
    assert segs
    for s in segs:
        assert text[s.start:s.end] == s.text and s.text.strip() == s.text


def test_sections_and_inline_cues():
    text = "Requirements:\n- Python\n- SQL is a plus\nNice to have:\n- Docker\nAbout us:\n- Free lunch"
    by = {s.text: s.section for s in segment(text)}
    assert by["Python"] == "required" and by["Docker"] == "preferred" and by["Free lunch"] == "company"


def test_long_text_is_wrapped():
    text = " ".join(["word"] * 500)
    segs = segment(text, max_words=70)
    assert all(len(s.text.split()) <= 70 for s in segs) and len(segs) >= 7


def test_abbreviation_not_split():
    segs = segment("Knowledge of databases e.g. PostgreSQL is useful.")
    assert len(segs) == 1


# ---------------------------------------------------------------------- trie
def test_trie_longest_match_and_case():
    t = PhraseTrie()
    t.add(["data", "analysis"], "A")
    t.add(["data", "analysis", "tools"], "B")
    t.add(["Excel"], "X", case_sensitive=True)
    toks = tokenize("Data analysis tools and excel and Excel")
    hits = t.find_all(toks)
    assert [(i, j, p) for i, j, p in hits] == [(0, 3, ["B"]), (6, 7, ["X"])]


# ------------------------------------------------------------------- tech
def test_excel_verb_is_not_a_skill_but_noun_is():
    ex = make()
    assert "Excel" not in ex.extract_skills("Excel in a fast-paced team environment.")
    assert "Excel" in ex.extract_skills("Advanced Excel and Microsoft Excel required.")


def test_tech_canonicalization_and_dedupe():
    ex = make()
    out = ex.extract_skills("We use NextJS, next.js and Node.js. Also ReactJS.")
    assert out.count("Next.js") == 1 and "Node.js" in out and "React.js" in out


# --------------------------------------------------------- refine/consolidate
def test_filler_trim_and_generic_drop():
    ex = make(["Strong communication skills", "experience", "years"])
    out = ex.extract_skills("Strong communication skills and experience, 5 years.")
    assert "communication skills" in out and "experience" not in out and "years" not in out


def test_compound_split_on_verb():
    ex = make(["Analyze data and build dashboards"])
    assert {"analyze data", "build dashboards"} <= {x.casefold() for x in ex.extract_skills("Analyze data and build dashboards.")}


def test_noun_compound_not_split():
    ex = make(["research and development"])
    assert "research and development" in ex.extract_skills("Strong research and development background.")


def test_nested_tech_kept_and_longer_span_wins():
    ex = make(["write SQL queries", "SQL queries"])
    out = ex.extract_skills("You will write SQL queries daily.")
    assert "write SQL queries" in out and "SQL" in out and "SQL queries" not in out


def test_sources_merge_and_confidence_rises():
    ex = make(["Python"], )
    sk = {s.text: s for s in ex.extract_detailed("Python is required.")}["Python"]
    assert set(sk.sources) == {"jobbert_skill", "tech"} and sk.confidence > 0.95


def test_span_offsets_point_into_original_text():
    ad = "Intro.\nRequirements:\n- Python and SQL"
    for s in make().extract_detailed(ad):
        for a, b in s.spans:
            assert ad[a:b].casefold() in {s.text.casefold(), "python", "sql"}


# --------------------------------------------------------------- requirement
def test_requirement_levels():
    ex = make()
    ad = "Requirements:\n- Python\nNice to have:\n- Docker\n- Kubernetes is a plus\nAbout us:\n- We love Slack"
    r = {s.text: (s.requirement, s.importance_hint) for s in ex.extract_detailed(ad)}
    assert r["Python"] == ("required", 1.0)
    assert r["Docker"][0] == "preferred" and r["Kubernetes"][0] == "preferred"
    assert r["Slack"][0] == "company" and r["Slack"][1] < r["Docker"][1]


def test_strongest_requirement_wins_across_mentions():
    ex = make()
    ad = "Python is a plus.\nRequirements:\n- Python"
    sk = {s.text: s for s in ex.extract_detailed(ad)}["Python"]
    assert sk.requirement == "required" and sk.count == 2


# ------------------------------------------------------------------ contract
def test_contract_types():
    ex = make(["build dashboards"])
    assert ex.extract_skills("") == [] and ex.extract_skills("   \n ") == []
    with pytest.raises(TypeError):
        ex.extract_skills(None)
    out = ex.extract_skills("Build dashboards with Python, python and PYTHON.")
    assert all(isinstance(x, str) for x in out)
    assert len({x.casefold() for x in out}) == len(out)


def test_handoff_is_json_serialisable():
    ex = make()
    json.dumps(to_handoff(ex.extract_detailed("Python and SQL")))


def test_unicode_and_nbsp_keep_offsets():
    ad = "Skills: Python​, SQL – “quoted”"
    assert len(clean_text(ad)) == len(ad)
    assert {"Python", "SQL"} <= set(make().extract_skills(ad))


def test_extract_many_batches_once():
    tagger = FakeTagger(["Python"])
    ex = SkillExtractor(None, ExtractorConfig(use_knowledge_model=False), skill_tagger=tagger, load_models=False)
    ads = [f"We need Python skills number {i}." for i in range(20)]
    res = ex.extract_many(ads)
    assert tagger.calls == 1 and len(res) == 20 and all("Python" in [s.text for s in r] for r in res)


def test_min_model_prob_filters_weak_spans():
    ex = SkillExtractor(None, ExtractorConfig(use_knowledge_model=False, use_tech=False, min_model_prob=0.5),
                        skill_tagger=FakeTagger(["data wrangling"], prob=0.3), load_models=False)
    assert ex.extract_skills("Strong data wrangling.") == []


# ---------------------------------------------------------------------- ESCO
ESCO_CSV = (
    'conceptUri,skillType,reuseLevel,preferredLabel,altLabels,hiddenLabels,description\n'
    'http://x/1,skill,sector-specific,perform statistical analysis,"conduct statistical analysis\nrun statistical analysis",,d\n'
    'http://x/2,skill,transversal,manage budgets,"budget management",,d\n'
    'http://x/3,knowledge,sector-specific,Python,,,single word skipped\n'
    'http://x/4,skill,sector-specific,analyse data,"data analysis",,d\n'
    'http://x/5,skill,sector-specific,analyse business data,"data analysis",,d\n'
)


@pytest.fixture()
def esco_path(tmp_path):
    p = tmp_path / "skills_en.csv"
    p.write_text(ESCO_CSV, encoding="utf-8")
    return str(p)


def test_esco_loading_and_matching(esco_path):
    ex = SkillExtractor(esco_path, ExtractorConfig(use_skill_model=False, use_knowledge_model=False), load_models=False)
    assert ex.esco_records == 5
    out = ex.extract_detailed("You will run statistical analysis and budget management duties; also data analysis.")
    names = {s.text.casefold(): s for s in out}
    assert "run statistical analysis" in names and "budget management" in names
    assert set(names["run statistical analysis"].esco_candidates) == {"perform statistical analysis"}
    # ambiguous phrase is still extracted (extraction != linking); both candidates passed to Member 2
    assert set(names["data analysis"].esco_candidates) == {"analyse data", "analyse business data"}


def test_esco_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SkillExtractor(str(p), load_models=False, config=ExtractorConfig(use_skill_model=False, use_knowledge_model=False))


# ------------------------------------------------------------------ evaluation
def test_metrics():
    tp, fp, fn = match_counts(["Python", "SQL queries", "junk"], ["python", "write SQL queries", "Docker"], partial=True)
    assert tp == 2 and fp == ["junk"] and fn == ["docker"]
    tp, _, _ = match_counts(["Python", "SQL queries"], ["python", "write SQL queries"], partial=False)
    assert tp == 1
    assert prf(2, 4, 4) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    res, _, _ = score([["a"]], [["a", "b"]])
    assert res["exact"]["precision"] == 1.0 and res["exact"]["recall"] == 0.5
    assert norm("  Node.js, ") == "node.js"


def test_ablation_runs_on_dev_set():
    from eval.evaluate import load_gold, run_ablation
    rows = load_gold(str(Path(__file__).resolve().parents[1] / "eval" / "sample_gold.jsonl"))
    ex = make(["Python and SQL"])
    table, _ = run_ablation(ex, rows)
    names = [n for n, _ in table]
    assert "skill model only" in names and "dictionaries only" in names and "+ knowledge model" not in names


# ------------------------------------------------- HF decoder (no torch needed)
def _decoder():
    from src.extraction import HFTokenTagger
    t = HFTokenTagger.__new__(HFTokenTagger)
    t._letters = {0: "B", 1: "I", 2: "O"}
    return t


def test_decode_bio_with_subwords_and_confidence():
    # text: "write SQL queries now"  words: 0=write 1=SQL(2 pieces) 2=queries 3=now
    # tokens: [CLS] write S ##QL queries now [SEP] [PAD]
    word_ids = [None, 0, 1, 1, 2, 3, None, None]
    offsets = [(0, 0), (0, 5), (6, 7), (7, 9), (10, 17), (18, 21), (0, 0), (0, 0)]
    labels = [2, 0, 1, 2, 1, 2, 2, 2]   # B(write) I(S) [##QL label ignored] I(queries) O(now)
    confs = [1, .9, .8, .1, .7, .99, 1, 1]
    spans = _decoder()._decode(labels, confs, word_ids, offsets)
    assert len(spans) == 1
    s, e, p = spans[0]
    assert (s, e) == (0, 17) and abs(p - (0.9 + 0.8 + 0.7) / 3) < 1e-9


def test_decode_two_spans_and_orphan_I():
    word_ids = [None, 0, 1, 2, 3, None]
    offsets = [(0, 0), (0, 6), (7, 8), (9, 12), (13, 16), (0, 0)]
    labels = [2, 0, 2, 1, 0, 2]          # B O I(orphan -> starts span) B
    spans = _decoder()._decode(labels, [1] * 6, word_ids, offsets)
    assert [(s, e) for s, e, _ in spans] == [(0, 6), (9, 12), (13, 16)]


# ------------------------------------------------- verb-coordination repair
def test_lone_verb_gets_object_from_coordinated_phrase():
    ex = make(["Train", "deploy deep learning models"])
    out = ex.extract_skills("Train and deploy deep learning models with PyTorch.")
    assert "Train deep learning models" in out and "deploy deep learning models" in out
    assert "Train" not in out


def test_verb_run_distributes_to_all():
    ex = make(["design", "build", "test APIs"])
    out = ex.extract_skills("You will design, build and test APIs.")
    assert {"design APIs", "build APIs", "test APIs"} <= set(out)


def test_lone_verb_without_object_is_dropped():
    ex = make(["Test"])
    assert ex.extract_skills("Test it today.") == []


def test_verb_not_attached_across_unrelated_text():
    ex = make(["build", "maintain web apps"])
    out = ex.extract_skills("We build great culture while others maintain web apps.")
    assert "build" not in out and "build web apps" not in out


# ------------------------------------------------ lexicon fidelity & eval honesty
def test_html_css_not_upgraded_to_html5_css3():
    out = make().extract_skills("Knowledge of HTML and CSS; HTML5 and CSS3 preferred.")
    assert {"HTML", "CSS", "HTML5", "CSS3"} <= set(out)


def test_alias_aware_matching():
    assert norm("React") == norm("React.js") and norm("A/B tests") == norm("A/B testing")
    assert norm("HTML") != norm("HTML5")
    assert match_counts(["React.js"], ["React"], partial=False)[0] == 1


def test_lexicon_leave_out_row_is_a_lower_bound():
    from eval.evaluate import run_ablation
    rows = [{"id": "a", "text": "Strong Python and Docker skills.", "skills": ["Python", "Docker"]}]
    table, _ = run_ablation(make(), rows)
    scores = {n: r["exact"]["recall"] for n, r in table}
    low = [v for n, v in scores.items() if n.startswith("lexicon minus gold terms")]
    assert low and low[0] == 0.0 and scores["+ tech lexicon (full)"] == 1.0


def test_without_tech_terms_shares_models():
    ex = make()
    clone = ex.without_tech_terms({"Python"})
    assert clone.skill_tagger is ex.skill_tagger and "Python" not in clone.extract_skills("Python and Docker")
    assert "Python" in ex.extract_skills("Python and Docker")   # original untouched


def test_convert_annotations_validates_offsets(tmp_path):
    from eval.convert_annotations import convert
    (tmp_path / "j.csv").write_text("job_id,job_category,job_description\nA,X,Use Python daily.\n")
    (tmp_path / "a.csv").write_text("job_id,skill_phrase,start_char,end_char,annotation_status\nA,Python,4,10,draft\nA,SQL,0,3,draft\n")
    info = convert(str(tmp_path / "j.csv"), str(tmp_path / "a.csv"), str(tmp_path / "g.jsonl"))
    assert info["annotations"] == 1 and len(info["problems"]) == 1
    assert json.loads((tmp_path / "g.jsonl").read_text())["skills"] == ["Python"]


def test_convert_accepts_expanded_phrase_with_blank_offsets(tmp_path):
    from eval.convert_annotations import convert
    (tmp_path / "j.csv").write_text("job_id,job_category,job_description\nA,X,Collect and clean data.\n")
    (tmp_path / "a.csv").write_text("job_id,skill_phrase,start_char,end_char,annotation_status\nA,collect data,,,reviewed\nA,clean data,,,reviewed\n")
    info = convert(str(tmp_path / "j.csv"), str(tmp_path / "a.csv"), str(tmp_path / "g.jsonl"))
    assert info["problems"] == [] and json.loads((tmp_path / "g.jsonl").read_text())["skills"] == ["collect data", "clean data"]
