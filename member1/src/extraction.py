"""Member 1 - Skill extraction (v2).

Public contract (frozen with the team)
--------------------------------------
    extract_skills(job_text: str) -> list[str]

Extras for downstream members (all optional to use)
---------------------------------------------------
    SkillExtractor.extract_detailed(job_text) -> list[Skill]
        Skill has: text, count, sources, confidence, requirement,
        importance_hint (0..1, a seed for Member 2's `importance d`),
        esco_candidates (hint for Member 2's linker), spans (char offsets).
    SkillExtractor.extract_many(texts) -> list[list[Skill]]   (batched, fast)

Pipeline
--------
    clean -> segment (lines / bullets / sentences, section headings tracked)
          -> taggers: JobBERT skill model + JobBERT knowledge model (batched)
          -> dictionaries: ESCO phrase trie + curated tech lexicon
          -> refine (trim fillers, split "X and <verb> Y", drop junk)
          -> consolidate overlapping spans by character offsets
          -> requirement level (required / preferred / ...) + importance hint

Design notes
------------
* Character offsets always refer to the string the caller passed in.
* torch / transformers are imported lazily, so everything except the HF
  taggers can be unit-tested without them (see tests/).
* A tagger is any object with `.tag(list[str]) -> list[list[(start, end, prob)]]`.
"""
from __future__ import annotations

import copy
import csv
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ESCO_VERSION = "1.2.1"
SKILL_MODEL = "jjzha/jobbert_skill_extraction"
KNOWLEDGE_MODEL = "jjzha/jobbert_knowledge_extraction"

SRC_SKILL = "jobbert_skill"
SRC_KNOWLEDGE = "jobbert_knowledge"
SRC_ESCO = "esco"
SRC_TECH = "tech"

# Prior reliability of the non-model sources (model sources use their softmax prob).
SOURCE_PRIOR = {SRC_ESCO: 0.75, SRC_TECH: 0.95}

# --------------------------------------------------------------------------
# Tech lexicon: canonical name -> aliases. Matched on whole tokens.
# Names in CASE_SENSITIVE only match with the exact casing written here
# (avoids "excel in teamwork", "pandas" the animal, "swift response", ...).
# --------------------------------------------------------------------------
TECH_LEXICON: Dict[str, List[str]] = {
    # languages
    "Python": ["Python"], "SQL": ["SQL"], "JavaScript": ["JavaScript", "JS"],
    "TypeScript": ["TypeScript"], "Java": ["Java"], "PHP": ["PHP"],
    "C++": ["C++"], "C#": ["C#"], "Golang": ["Golang"], "Rust": ["Rust"],
    "Kotlin": ["Kotlin"], "Swift": ["Swift"], "Ruby": ["Ruby"], "Scala": ["Scala"],
    "Dart": ["Dart"], "Julia": ["Julia"], "R": ["R programming", "R language", "RStudio"],
    "Bash": ["Bash", "shell scripting"], "NoSQL": ["NoSQL"],
    "HTML": ["HTML"], "HTML5": ["HTML5", "HTML 5"], "CSS": ["CSS"], "CSS3": ["CSS3", "CSS 3"],
    # data / analytics
    "Excel": ["Excel", "EXCEL", "MS Excel", "Microsoft Excel"],
    "Google Sheets": ["Google Sheets"],
    "Pandas": ["Pandas"], "NumPy": ["NumPy"], "SciPy": ["SciPy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Matplotlib": ["Matplotlib"], "Seaborn": ["Seaborn"], "Plotly": ["Plotly"],
    "Power BI": ["Power BI", "PowerBI"], "Tableau": ["Tableau"],
    "Looker": ["Looker"], "Looker Studio": ["Looker Studio", "Data Studio"],
    "Power Query": ["Power Query"], "Power Automate": ["Power Automate"],
    "DAX": ["DAX"], "Star Schema": ["Star Schema"], "ETL": ["ETL", "ELT"],
    "data warehousing": ["data warehousing", "data warehouse"],
    "data modeling": ["data modeling", "data modelling"],
    "data transformation": ["data transformation"],
    "data cleansing": ["data cleansing", "data cleaning", "cleansing"],
    "statistics": ["statistics"], "A/B testing": ["A/B testing", "A/B tests", "A/B test"],
    "Jupyter": ["Jupyter", "Jupyter Notebook"], "Streamlit": ["Streamlit"],
    "Spark": ["Spark", "Apache Spark"], "PySpark": ["PySpark"], "Hadoop": ["Hadoop"],
    "Airflow": ["Airflow", "Apache Airflow"], "dbt": ["dbt"], "Kafka": ["Kafka"],
    "Snowflake": ["Snowflake"], "BigQuery": ["BigQuery"], "Hive": ["Hive"],
    "SSIS": ["SSIS"], "Salesforce": ["Salesforce"], "SAP": ["SAP"],
    # databases
    "PostgreSQL": ["PostgreSQL", "Postgres"], "MySQL": ["MySQL"], "SQLite": ["SQLite"],
    "SQL Server": ["SQL Server", "MSSQL"], "Oracle": ["Oracle"],
    "MongoDB": ["MongoDB"], "Redis": ["Redis"], "Elasticsearch": ["Elasticsearch"],
    "Firebase": ["Firebase"], "Supabase": ["Supabase"],
    # web / backend
    "React.js": ["React.js", "ReactJS", "React"], "React Native": ["React Native"],
    "Next.js": ["Next.js", "NextJS"], "Node.js": ["Node.js", "NodeJS"],
    "Express.js": ["Express.js", "ExpressJS"], "Vue.js": ["Vue.js", "VueJS", "Vue"],
    "Angular": ["Angular"], "Svelte": ["Svelte"], "Redux": ["Redux"],
    "Tailwind CSS": ["Tailwind CSS", "Tailwind"], "Bootstrap": ["Bootstrap"],
    "Django": ["Django"], "Flask": ["Flask"], "FastAPI": ["FastAPI"],
    "Spring Boot": ["Spring Boot"], ".NET": [".NET"], "GraphQL": ["GraphQL"],
    "REST APIs": ["REST APIs", "REST API", "RESTful APIs", "RESTful"],
    "WordPress": ["WordPress"], "Shopify": ["Shopify"], "Flutter": ["Flutter"],
    # devops / cloud / tools
    "Git": ["Git"], "GitHub": ["GitHub"], "GitLab": ["GitLab"], "Bitbucket": ["Bitbucket"],
    "Docker": ["Docker"], "Kubernetes": ["Kubernetes", "K8s"], "Terraform": ["Terraform"],
    "Jenkins": ["Jenkins"], "Linux": ["Linux"], "Nginx": ["Nginx"],
    "AWS": ["AWS", "Amazon Web Services"],
    "Azure": ["Azure", "Microsoft Azure"],
    "GCP": ["GCP", "Google Cloud Platform", "Google Cloud"],
    "CI/CD pipelines": ["CI/CD pipelines", "CI/CD pipeline", "CI/CD"],
    "automated testing": ["automated testing", "test automation"],
    "Selenium": ["Selenium"], "Cypress": ["Cypress"], "Jest": ["Jest"],
    "Pytest": ["Pytest"], "Postman": ["Postman"],
    "Jira": ["Jira"], "Confluence": ["Confluence"], "ClickUp": ["ClickUp"],
    "Notion": ["Notion"], "Slack": ["Slack"], "Figma": ["Figma"],
    "Photoshop": ["Photoshop"],
    "Scrum": ["Scrum"], "Agile": ["Agile"], "Kanban": ["Kanban"],
    "sprint planning": ["sprint planning"],
    # ML / AI
    "TensorFlow": ["TensorFlow"], "PyTorch": ["PyTorch"], "Keras": ["Keras"],
    "XGBoost": ["XGBoost"], "ONNX": ["ONNX"], "TensorRT": ["TensorRT"],
    "Triton Inference Server": ["Triton Inference Server"],
    "Hugging Face Transformers": ["Hugging Face Transformers", "Hugging Face"],
    "LangChain": ["LangChain"], "LlamaIndex": ["LlamaIndex"],
    "Pinecone": ["Pinecone"], "Weaviate": ["Weaviate"], "pgvector": ["pgvector"],
    "embeddings": ["embeddings"], "vector search": ["vector search"],
    "vector databases": ["vector databases", "vector database"],
    "machine learning": ["machine learning", "ML"],
    "deep learning": ["deep learning"],
    "NLP": ["NLP", "natural language processing"],
    "computer vision": ["computer vision"],
    "prompt engineering": ["prompt engineering"],
    "LLMs": ["LLMs", "LLM", "large language models"],
    "RAG": ["RAG", "retrieval-augmented generation"],
    "GenAI": ["GenAI", "generative AI"], "OpenAI API": ["OpenAI API"],
    # marketing
    "Google Analytics": ["Google Analytics"], "Google Ads": ["Google Ads", "AdWords"],
    "Google Tag Manager": ["Google Tag Manager"], "Meta Ads": ["Meta Ads", "Facebook Ads"],
    "SEO": ["SEO"], "SEM": ["SEM"], "keyword research": ["keyword research"],
    "on-page optimization": ["on-page optimization"], "off-page SEO": ["off-page SEO"],
    "affiliate marketing": ["affiliate marketing"],
    "email marketing": ["email marketing"], "Mailchimp": ["Mailchimp"],
    "HubSpot": ["HubSpot"], "Ahrefs": ["Ahrefs"], "SEMrush": ["SEMrush"],
}

CASE_SENSITIVE: Set[str] = {
    "Excel", "Pandas", "Git", "Swift", "Rust", "Ruby", "Scala", "Dart", "Julia",
    "Flask", "Hive", "Spark", "Looker", "Oracle", "Angular", "Notion", "Slack",
    "Jest", "Cypress", "Bootstrap", "Redux", "Vue.js", "React.js", "Snowflake",
    "Kafka", "Airflow", "ML", "RAG", "SEM", "JS", "R",
}
# canonical -> tokens that, when they directly follow the match, cancel it.
NEG_NEXT: Dict[str, Set[str]] = {
    "Excel": {"in", "at", "as", "under", "from", "beyond", "with"},
    "Spark": {"joy", "creativity", "innovation"},
}
# Aliases that are only valid when the exact casing is used even if the
# canonical is not in CASE_SENSITIVE (acronyms colliding with common words).
CASE_SENSITIVE_ALIASES: Set[str] = {"JS", "ML", "RAG", "SEM", "R", "Vue", "React", "Hive", "SAP", "DAX"}

ACTION_VERBS = set(
    "analyze analyse build create develop design manage perform preprocess implement "
    "maintain monitor optimize optimise coordinate communicate evaluate test write train "
    "deploy assess present prepare document review lead conduct collaborate clean "
    "visualize visualise automate integrate query extract report".split()
)
_STOP = set(
    "a an the and or of to in on for with at by from as is are be been being this that "
    "these those it its you we our your their they he she i not no yes can will may "
    "must should would could also more most other such any all each both etc".split()
)
_GENERIC = {
    "experience", "skills", "skill", "ability", "abilities", "team", "teams", "work",
    "working", "years", "year", "knowledge", "role", "position", "candidate", "company",
    "job", "opportunity", "environment", "responsibilities", "requirements", "plus",
    "degree", "bachelor", "master", "background", "understanding", "someone", "person",
    "people", "other", "etc", "equal opportunity employer", "benefits", "salary",
    "time", "day", "days", "month", "months", "week", "weeks", "bonus", "preferred",
}
_LEAD_FILLERS = re.compile(
    r"^(?:(?:the|a|an|and|or|with|in|of|to|for|on|as|strong|good|excellent|solid|proven)\s+)+",
    re.I,
)
_EDGE_PUNCT = " \t\r\n,;:.!?()[]{}\"'`*-–—•"


def alias_to_canonical() -> Dict[str, str]:
    """casefolded alias -> casefolded canonical, e.g. 'react' -> 'react.js' (for alias-aware scoring)."""
    out: Dict[str, str] = {}
    for canonical, aliases in TECH_LEXICON.items():
        for a in aliases + [canonical]:
            out[a.casefold()] = canonical.casefold()
    return out


# --------------------------------------------------------------------------
# Config & data classes
# --------------------------------------------------------------------------
@dataclass
class ExtractorConfig:
    use_skill_model: bool = True
    use_knowledge_model: bool = True
    use_esco: bool = True
    use_tech: bool = True
    min_model_prob: float = 0.0     # drop model spans below this mean token prob
    max_span_words: int = 8         # longer spans are almost always sentences, not skills
    max_segment_words: int = 70     # hard cap per tagger input (keeps well under 512 tokens)
    esco_min_words: int = 2         # single-word ESCO labels are too noisy; tech lexicon covers tools
    batch_size: int = 16
    split_compounds: bool = True


@dataclass
class Segment:
    start: int
    end: int
    text: str
    section: str  # required | preferred | responsibility | company | unknown


@dataclass
class Mention:
    start: int
    end: int
    text: str
    src_conf: Dict[str, float]
    label: Optional[str] = None            # display override: canonical tech name, or expanded verb phrase
    esco: Set[str] = field(default_factory=set)
    requirement: str = "unknown"

    @property
    def sources(self) -> Set[str]:
        return set(self.src_conf)

    @property
    def confidence(self) -> float:
        miss = 1.0
        for c in self.src_conf.values():
            miss *= 1.0 - max(0.0, min(1.0, c))
        return round(1.0 - miss, 4)

    @property
    def display(self) -> str:
        return self.label or self.text

    def merge(self, other: "Mention") -> None:
        for s, c in other.src_conf.items():
            self.src_conf[s] = max(self.src_conf.get(s, 0.0), c)
        self.esco |= other.esco
        if other.label and not self.label and (other.start, other.end) == (self.start, self.end):
            self.label = other.label


IMPORTANCE = {"required": 1.0, "responsibility": 0.8, "unknown": 0.7, "preferred": 0.5, "company": 0.3}


@dataclass
class Skill:
    text: str
    count: int
    sources: List[str]
    confidence: float
    requirement: str
    importance_hint: float
    esco_candidates: List[str]
    spans: List[Tuple[int, int]]

    def to_dict(self) -> dict:
        return {
            "skill": self.text, "count": self.count, "sources": self.sources,
            "confidence": self.confidence, "requirement": self.requirement,
            "importance_hint": self.importance_hint,
            "esco_candidates": self.esco_candidates, "spans": [list(s) for s in self.spans],
        }


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_NBSP_MAP = {ord(" "): " ", ord("​"): " ", ord(" "): " ", ord(" "): " ",
             ord("\r"): " ", ord("\t"): " ", ord("’"): "'", ord("‘"): "'",
             ord("“"): '"', ord("”"): '"'}


def clean_text(text: str) -> str:
    """Length-preserving cleanup, so offsets stay valid for the caller's string."""
    return text.translate(_NBSP_MAP)


def tokenize(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def normalize(text: str) -> str:
    return " ".join(text.split())


class PhraseTrie:
    """Token trie, case-insensitive with optional exact-case entries, longest match."""

    def __init__(self) -> None:
        self.root: dict = {}
        self.size = 0

    def add(self, tokens: Sequence[str], payload, case_sensitive: bool = False) -> None:
        node = self.root
        for t in tokens:
            node = node.setdefault(t.casefold(), {})
        node.setdefault(None, []).append((payload, tuple(tokens) if case_sensitive else None))
        self.size += 1

    def find_all(self, toks: List[Tuple[str, int, int]]):
        """Yield (first_token_idx, last_token_idx_exclusive, [payloads])."""
        out, i, n = [], 0, len(toks)
        while i < n:
            node, best, j = self.root, None, i
            while j < n:
                nxt = node.get(toks[j][0].casefold())
                if nxt is None:
                    break
                node, j = nxt, j + 1
                if None in node:
                    valid = [p for p, cs in node[None]
                             if cs is None or all(toks[i + k][0] == cs[k] for k in range(len(cs)))]
                    if valid:
                        best = (j, valid)
            if best:
                out.append((i, best[0], best[1]))
                i = best[0]
            else:
                i += 1
        return out


# --------------------------------------------------------------------------
# Segmentation (lines -> bullets -> sentences) with section tracking
# --------------------------------------------------------------------------
_BULLET_RE = re.compile(r"^\s*(?:[-*•▪●◦‣·–—]|\(?\d{1,2}[.)])\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")
_INLINE_BULLET = re.compile(r"\s+[•▪●◦‣·]\s+")
_ABBREV_END = ("e.g.", "i.e.", "vs.", "approx.", "etc.")
_COLON_RE = re.compile(r"^([^:\n]{2,70}?):\s*(.*)$")

_SECTION_PATTERNS = [
    ("preferred", r"nice[- ]to[- ]have|preferred|bonus|desirable|good to have|a plus|plus points?|optional|advantageous|beneficial"),
    ("company", r"about (?:us|the (?:company|team))|who we are|benefits|perks|what we offer|we offer|why (?:join|work)|our (?:mission|culture|values)|compensation|salary|equal opportunity"),
    ("responsibility", r"responsibilit|what you(?:'ll| will) do|duties|your role|the role|day[- ]to[- ]day|job description|key tasks|you will"),
    ("required", r"requirements?|qualifications?|must[- ]haves?|what you(?:'ll)? (?:bring|need)|you have|skills?|who you are|about you|minimum|essential|required|looking for|experience|tech(?:nical)? stack"),
]
_EXACT_HEADINGS = {
    "requirements": "required", "qualifications": "required", "minimum qualifications": "required",
    "required qualifications": "required", "required skills": "required", "must have": "required",
    "must haves": "required", "what you bring": "required", "what you need": "required",
    "who you are": "required", "about you": "required", "key skills": "required",
    "skills": "required", "technical skills": "required", "tech stack": "required",
    "skills and qualifications": "required", "skills & qualifications": "required",
    "preferred qualifications": "preferred", "preferred skills": "preferred",
    "nice to have": "preferred", "nice to haves": "preferred", "bonus": "preferred",
    "bonus points": "preferred", "good to have": "preferred", "desirable": "preferred",
    "responsibilities": "responsibility", "key responsibilities": "responsibility",
    "duties": "responsibility", "what you will do": "responsibility", "what you'll do": "responsibility",
    "your role": "responsibility", "the role": "responsibility", "job description": "responsibility",
    "about us": "company", "about the company": "company", "who we are": "company",
    "benefits": "company", "perks": "company", "what we offer": "company",
    "why join us": "company", "compensation": "company",
}
_PREF_CUES = re.compile(
    r"\b(?:nice[- ]to[- ]have|(?:is|are|would be|will be)\s+(?:a\s+|an\s+)?(?:plus|bonus|advantage|nice|beneficial|desirable|preferred|great|welcome)"
    r"|a plus|bonus points?|preferred|desirable|good to have|ideally|advantageous|an advantage|not required|optional)\b", re.I)
_REQ_CUES = re.compile(
    r"\b(?:must|required|requires?|essential|mandatory|needs? to|proficien\w+|expert\w*|proven|strong|solid|at least \d+|\d+\+?\s*years?)\b", re.I)


def _norm_heading(s: str) -> str:
    s = s.casefold().replace("’", "'")
    return " ".join(re.sub(r"[^a-z&' ]+", " ", s).split())


def _classify_colon_heading(h: str) -> Optional[str]:
    if len(h.split()) > 8:
        return None
    low = h.casefold()
    for sec, pat in _SECTION_PATTERNS:
        if re.search(pat, low):
            return sec
    return None


def _heading(body: str) -> Optional[Tuple[str, str, int]]:
    """Return (section, rest_text, rest_offset) when the line is/starts with a heading."""
    m = _COLON_RE.match(body)
    if m:
        sec = _classify_colon_heading(m.group(1))
        if sec:
            return sec, m.group(2), m.start(2)
    if len(body.split()) <= 5:
        sec = _EXACT_HEADINGS.get(_norm_heading(body))
        if sec:
            return sec, "", len(body)
    return None


def _strip_range(text: str, a: int, b: int) -> Tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _split_by(text: str, a: int, b: int, pattern: "re.Pattern") -> List[Tuple[int, int]]:
    out, pos = [], a
    for m in pattern.finditer(text, a, b):
        if m.end() > m.start():
            out.append(_strip_range(text, pos, m.start()))
            pos = m.end()
    out.append(_strip_range(text, pos, b))
    return [(s, e) for s, e in out if e > s]


def _merge_abbrev(text: str, pieces: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    carry: Optional[Tuple[int, int]] = None
    for s, e in pieces:
        if carry:
            s = carry[0]
            carry = None
        if text[s:e].casefold().endswith(_ABBREV_END):
            carry = (s, e)
            continue
        out.append((s, e))
    if carry:
        out.append(carry)
    return out


def _wrap_words(text: str, a: int, b: int, max_words: int) -> List[Tuple[int, int]]:
    words = [(m.start() + a, m.end() + a) for m in re.finditer(r"\S+", text[a:b])]
    if len(words) <= max_words:
        return [(a, b)]
    out, i = [], 0
    while i < len(words):
        j = min(i + max_words, len(words))
        if j < len(words):  # prefer breaking after a comma in the last third of the window
            for k in range(j, i + (2 * max_words) // 3, -1):
                if text[words[k - 1][1] - 1] == ",":
                    j = k
                    break
        out.append((words[i][0], words[j - 1][1]))
        i = j
    return out


def segment(text: str, max_words: int = 70) -> List[Segment]:
    """Split into tagger-sized segments. text[seg.start:seg.end] == seg.text always."""
    segs: List[Segment] = []
    section = "unknown"
    for m in re.finditer(r"[^\n]+", text):
        a, b = _strip_range(text, m.start(), m.end())
        if a >= b:
            continue
        bm = _BULLET_RE.match(text[a:b])   # slice: '^' would anchor at string start, not at `a`
        if bm:
            a, b = _strip_range(text, a + bm.end(), b)
            if a >= b:
                continue
        local = section
        h = _heading(text[a:b])
        if h:
            sec, rest, rest_off = h
            if not rest.strip():
                section = sec
                continue
            local = sec
            a, b = _strip_range(text, a + rest_off, b)
            if a >= b:
                continue
        pieces: List[Tuple[int, int]] = []
        for s, e in _split_by(text, a, b, _INLINE_BULLET):
            pieces.extend(_split_by(text, s, e, _SENT_SPLIT))
        for s, e in _merge_abbrev(text, pieces):
            for s2, e2 in _wrap_words(text, s, e, max_words):
                segs.append(Segment(s2, e2, text[s2:e2], local))
    return segs


def requirement_for(seg: Segment) -> str:
    if seg.section == "company":
        return "company"
    if _PREF_CUES.search(seg.text):
        return "preferred"
    if _REQ_CUES.search(seg.text):
        return "required"
    return seg.section if seg.section in IMPORTANCE else "unknown"


# --------------------------------------------------------------------------
# HuggingFace token-classification tagger (lazy torch import)
# --------------------------------------------------------------------------
class HFTokenTagger:
    """Batched B/I/O span tagger. Works with the SkillSpan JobBERT checkpoints."""

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 256, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for character offsets.")
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device).eval()
        id2label = self.model.config.id2label
        self._letters = {int(i): str(l).strip().upper()[:1] for i, l in id2label.items()}
        if not set(self._letters.values()) <= {"B", "I", "O"} or "O" not in self._letters.values():
            raise RuntimeError(f"Unexpected label set for {model_name}: {id2label}")

    def tag(self, texts: List[str]) -> List[List[Tuple[int, int, float]]]:
        torch = self.torch
        results: List[List[Tuple[int, int, float]]] = [[] for _ in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for lo in range(0, len(order), self.batch_size):
            idx = order[lo:lo + self.batch_size]
            batch = [texts[i] for i in idx]
            enc = self.tokenizer(batch, return_offsets_mapping=True, truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            with torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
                probs = torch.softmax(logits, dim=-1)
                conf, lab = probs.max(dim=-1)
            conf, lab = conf.cpu().tolist(), lab.cpu().tolist()
            for row, i in enumerate(idx):
                results[i] = self._decode(lab[row], conf[row], enc.word_ids(row), offsets[row])
        return results

    def _decode(self, labels, confs, word_ids, offsets) -> List[Tuple[int, int, float]]:
        spans: List[Tuple[int, int, float]] = []
        cur: Optional[list] = None
        prev_word = None

        def flush():
            if cur is not None:
                spans.append((cur[0], cur[1], sum(cur[2]) / len(cur[2])))

        for k, wid in enumerate(word_ids):
            if wid is None:
                continue
            s, e = offsets[k]
            if wid == prev_word:                    # sub-word piece: extend, never relabel
                if cur is not None:
                    cur[1] = max(cur[1], e)
                continue
            prev_word = wid
            tag = self._letters[labels[k]]          # label of the first sub-word decides
            if tag == "O":
                flush()
                cur = None
            elif tag == "B" or cur is None:
                flush()
                cur = [s, e, [confs[k]]]
            else:                                   # "I"
                cur[1] = e
                cur[2].append(confs[k])
        flush()
        return spans


# --------------------------------------------------------------------------
# Span refinement helpers
# --------------------------------------------------------------------------
def _trim(text: str, a: int, b: int) -> Tuple[int, int]:
    while a < b and text[a] in _EDGE_PUNCT:
        a += 1
    while b > a and text[b - 1] in _EDGE_PUNCT:
        b -= 1
    m = _LEAD_FILLERS.match(text[a:b])  # slice: see note in segment()
    if m and a + m.end() < b:
        a += m.end()
    tail = re.search(r"\s+(?:and|or)$", text[a:b], re.I)
    if tail:
        b = a + tail.start()
    while b > a and text[b - 1] in _EDGE_PUNCT:
        b -= 1
    return a, b


def _valid_span(span: str, max_words: int) -> bool:
    if len(span) < 2 or not re.search(r"[A-Za-z]", span):
        return False
    words = span.split()
    if len(words) > max_words:
        return False
    low = span.casefold()
    if low in _GENERIC:
        return False
    toks = [t.casefold() for t in re.findall(r"\w+", span)]
    if not toks or all(t in _STOP or t.isdigit() for t in toks):
        return False
    if all(t in _STOP or t in _GENERIC or t.isdigit() for t in toks):
        return False
    return True


def _split_compound(text: str, a: int, b: int) -> List[Tuple[int, int]]:
    m = list(re.finditer(r"\s+and\s+", text[a:b], re.I))
    if len(m) == 1:
        right = text[a + m[0].end():b].split()
        if right and right[0].casefold() in ACTION_VERBS:
            return [(a, a + m[0].start()), (a + m[0].end(), b)]
    return [(a, b)]



_SEP_RE = re.compile(r"\s*(?:,\s*)?(?:(?:and|or|&|/)\s*)?", re.I)


def _fix_verb_fragments(text: str, ms: List[Mention]) -> List[Mention]:
    """Repair verb coordination the tagger splits, e.g. 'build and maintain web apps'.

    The model tags 'build' and 'maintain web apps' separately. A lone action verb is
    never a useful skill, so: if it is followed (through 'and' / ',' / 'or') by a
    verb-led phrase, give it that phrase's object ('build web apps'); otherwise drop it.
    The expanded wording goes in `label`, while the span still points at the verb.
    """
    model_only = {SRC_SKILL, SRC_KNOWLEDGE}
    ms = sorted(ms, key=lambda m: m.start)
    out: List[Mention] = []
    pending: List[Mention] = []
    prev_end = 0

    def is_lone_verb(m: Mention) -> bool:
        return (len(m.text.split()) == 1 and m.text.casefold() in ACTION_VERBS
                and set(m.src_conf) <= model_only)

    for m in ms:
        gap_ok = bool(pending) and _SEP_RE.fullmatch(text[prev_end:m.start]) is not None
        if is_lone_verb(m):
            if pending and not gap_ok:
                pending = []                      # earlier verbs had nothing to attach to: dropped
            pending.append(m)
        else:
            words = m.text.split(None, 1)
            if gap_ok and len(words) == 2 and words[0].casefold() in ACTION_VERBS:
                for v in pending:
                    v.label = f"{v.text} {words[1]}"
                    out.append(v)
            pending = []
            out.append(m)
        prev_end = m.end
    return out

# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------
class SkillExtractor:
    def __init__(
        self,
        esco_csv: Optional[str] = None,
        config: Optional[ExtractorConfig] = None,
        skill_tagger=None,
        knowledge_tagger=None,
        extra_tech: Optional[Dict[str, List[str]]] = None,
        load_models: bool = True,
    ):
        self.cfg = config or ExtractorConfig()
        self._cache: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = {}
        self.skill_tagger = skill_tagger
        self.knowledge_tagger = knowledge_tagger
        if load_models:
            if self.skill_tagger is None and self.cfg.use_skill_model:
                self.skill_tagger = HFTokenTagger(SKILL_MODEL, self.cfg.batch_size)
            if self.knowledge_tagger is None and self.cfg.use_knowledge_model:
                try:
                    self.knowledge_tagger = HFTokenTagger(KNOWLEDGE_MODEL, self.cfg.batch_size)
                except Exception as exc:  # optional second model must never break the pipeline
                    warnings.warn(f"Knowledge model unavailable ({exc!r}); continuing without it.")
                    self.knowledge_tagger = None
        self.esco_trie = PhraseTrie()
        self.esco_lookup: Dict[Tuple[str, ...], Set[str]] = {}
        self.esco_records = 0
        if esco_csv:
            self._load_esco(esco_csv)
        self._lexicon = dict(TECH_LEXICON)
        for k, v in (extra_tech or {}).items():
            self._lexicon[k] = list(dict.fromkeys(self._lexicon.get(k, []) + list(v)))
        self.tech_trie = self._build_tech_trie(self._lexicon)

    @staticmethod
    def _build_tech_trie(lexicon: Dict[str, List[str]], exclude: Iterable[str] = ()) -> PhraseTrie:
        trie, skip = PhraseTrie(), set(exclude)
        for canonical, aliases in lexicon.items():
            if canonical in skip:
                continue
            for alias in aliases:
                toks = [t for t, _, _ in tokenize(alias)]
                cs = canonical in CASE_SENSITIVE or alias in CASE_SENSITIVE_ALIASES
                trie.add(toks, canonical, case_sensitive=cs)
        return trie

    def without_tech_terms(self, canonicals: Iterable[str]) -> "SkillExtractor":
        """Clone (shared models/cache) whose tech lexicon lacks the given canonical names.
        Used to bound how much of an eval score comes from lexicon entries written while looking at the eval ads."""
        clone = copy.copy(self)
        clone.tech_trie = self._build_tech_trie(self._lexicon, canonicals)
        return clone

    # ---- construction helpers ------------------------------------------
    def with_config(self, **overrides) -> "SkillExtractor":
        """Cheap copy that shares models, tries and the tagger cache (used for ablations)."""
        clone = copy.copy(self)
        clone.cfg = ExtractorConfig(**{**self.cfg.__dict__, **overrides})
        return clone

    def _load_esco(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            missing = {"preferredLabel", "altLabels"} - cols
            if missing:
                raise ValueError(f"ESCO skills CSV is missing columns: {sorted(missing)}")
            label_cols = [c for c in ("preferredLabel", "altLabels", "hiddenLabels") if c in cols]
            for row in reader:
                preferred = normalize(row.get("preferredLabel", ""))
                if not preferred:
                    continue
                self.esco_records += 1
                phrases = [preferred]
                for c in label_cols[1:]:
                    phrases += (row.get(c) or "").splitlines()
                for phrase in phrases:
                    toks = tuple(t.casefold() for t, _, _ in tokenize(normalize(phrase)))
                    if len(toks) < self.cfg.esco_min_words:
                        continue
                    self.esco_lookup.setdefault(toks, set()).add(preferred)
        for toks, labels in self.esco_lookup.items():
            self.esco_trie.add(toks, labels)

    # ---- tagging -------------------------------------------------------
    def _model_sources(self):
        if self.cfg.use_skill_model and self.skill_tagger is not None:
            yield SRC_SKILL, self.skill_tagger
        if self.cfg.use_knowledge_model and self.knowledge_tagger is not None:
            yield SRC_KNOWLEDGE, self.knowledge_tagger

    def _ensure_tagged(self, seg_texts: Iterable[str]) -> None:
        uniq = list(dict.fromkeys(seg_texts))
        for name, tagger in self._model_sources():
            todo = [t for t in uniq if (name, t) not in self._cache]
            if todo:
                for t, spans in zip(todo, tagger.tag(todo)):
                    self._cache[(name, t)] = spans

    def clear_cache(self) -> None:
        self._cache.clear()

    # ---- per-text pipeline --------------------------------------------
    def _refine_model_span(self, text, a, b, prob, source) -> List[Mention]:
        a, b = _trim(text, a, b)
        ranges = _split_compound(text, a, b) if self.cfg.split_compounds else [(a, b)]
        out = []
        for s, e in ranges:
            s, e = _trim(text, s, e)
            if e > s and _valid_span(text[s:e], self.cfg.max_span_words):
                out.append(Mention(s, e, text[s:e], {source: prob}))
        return out

    def _segment_mentions(self, text: str, seg: Segment) -> List[Mention]:
        model_ms: List[Mention] = []
        for name, _ in self._model_sources():
            for s, e, p in self._cache.get((name, seg.text), []):
                if p >= self.cfg.min_model_prob:
                    model_ms += self._refine_model_span(text, seg.start + s, seg.start + e, p, name)
        ms: List[Mention] = _fix_verb_fragments(text, model_ms) if self.cfg.split_compounds else model_ms
        toks = None
        if self.cfg.use_esco and self.esco_trie.size:
            toks = tokenize(seg.text)
            for i, j, payloads in self.esco_trie.find_all(toks):
                a, b = seg.start + toks[i][1], seg.start + toks[j - 1][2]
                labels = set().union(*payloads)
                ms.append(Mention(a, b, text[a:b], {SRC_ESCO: SOURCE_PRIOR[SRC_ESCO]}, esco=labels))
        if self.cfg.use_tech:
            toks = toks or tokenize(seg.text)
            for i, j, payloads in self.tech_trie.find_all(toks):
                canonical = payloads[0]
                if j < len(toks) and toks[j][0].casefold() in NEG_NEXT.get(canonical, ()):
                    continue
                a, b = seg.start + toks[i][1], seg.start + toks[j - 1][2]
                ms.append(Mention(a, b, text[a:b], {SRC_TECH: SOURCE_PRIOR[SRC_TECH]}, label=canonical))
        return _consolidate(ms)

    def _mentions(self, text: str) -> List[Mention]:
        text = clean_text(text)
        segs = segment(text, self.cfg.max_segment_words)
        self._ensure_tagged(s.text for s in segs)
        out: List[Mention] = []
        for seg in segs:
            req = requirement_for(seg)
            for m in self._segment_mentions(text, seg):
                m.requirement = req
                out.append(m)
        return out

    def _to_skills(self, mentions: List[Mention]) -> List[Skill]:
        groups: Dict[str, List[Mention]] = {}
        for m in sorted(mentions, key=lambda m: (m.start, -(m.end - m.start))):
            groups.setdefault(m.display.casefold(), []).append(m)
        skills = []
        for ms in groups.values():
            best = max(ms, key=lambda m: (IMPORTANCE[m.requirement], m.confidence))
            src: Dict[str, float] = {}
            labels: Set[str] = set()
            for m in ms:
                for s, c in m.src_conf.items():
                    src[s] = max(src.get(s, 0.0), c)
                labels |= m.esco
            key = tuple(t.casefold() for t, _, _ in tokenize(best.display))
            labels |= self.esco_lookup.get(key, set())
            miss = 1.0
            for c in src.values():
                miss *= 1.0 - c
            skills.append(Skill(
                text=best.display, count=len(ms), sources=sorted(src),
                confidence=round(1.0 - miss, 4), requirement=best.requirement,
                importance_hint=IMPORTANCE[best.requirement],
                esco_candidates=sorted(labels)[:5], spans=[(m.start, m.end) for m in ms],
            ))
        return skills

    # ---- public API ----------------------------------------------------
    def extract_detailed(self, job_text: str, min_confidence: float = 0.0) -> List[Skill]:
        if not isinstance(job_text, str):
            raise TypeError("job_text must be a string")
        if not job_text.strip():
            return []
        skills = self._to_skills(self._mentions(job_text))
        return [s for s in skills if s.confidence >= min_confidence]

    def extract_skills(self, job_text: str) -> List[str]:
        """Frozen contract: str -> list[str] (unique, document order)."""
        return [s.text for s in self.extract_detailed(job_text)]

    def extract_many(self, texts: Sequence[str]) -> List[List[Skill]]:
        """Same as extract_detailed per text, but tags all segments in one batched pass."""
        cleaned = []
        for t in texts:
            if not isinstance(t, str):
                raise TypeError("every job text must be a string")
            cleaned.append(clean_text(t))
        self._ensure_tagged(s.text for t in cleaned for s in segment(t, self.cfg.max_segment_words))
        return [self.extract_detailed(t) for t in texts]

    def extract_skills_many(self, texts: Sequence[str]) -> List[List[str]]:
        return [[s.text for s in skills] for skills in self.extract_many(texts)]


def _consolidate(ms: List[Mention]) -> List[Mention]:
    """Resolve overlaps by character offsets (never by string containment)."""
    by_range: Dict[Tuple[int, int], Mention] = {}
    for m in ms:
        key = (m.start, m.end)
        if key in by_range:
            by_range[key].merge(m)
        else:
            by_range[key] = m
    keep: List[Mention] = []
    for m in sorted(by_range.values(), key=lambda m: (m.start, -(m.end - m.start))):
        placed = False
        for idx, k in enumerate(keep):
            overlap = min(k.end, m.end) - max(k.start, m.start)
            if overlap <= 0:
                continue
            m_in_k = k.start <= m.start and m.end <= k.end
            if m_in_k:
                if SRC_TECH in m.src_conf:      # a tool inside a longer phrase stays linkable
                    continue
                k.merge(m)
                placed = True
                break
            shorter = min(k.end - k.start, m.end - m.start)
            if overlap / shorter >= 0.5:
                if (m.end - m.start) > (k.end - k.start) and SRC_TECH not in k.src_conf:
                    m.merge(k)
                    keep[idx] = m
                else:
                    k.merge(m)
                placed = True
                break
        if not placed:
            keep.append(m)
    return sorted(keep, key=lambda m: m.start)


# --------------------------------------------------------------------------
# Module-level handoff function (what other members import)
# --------------------------------------------------------------------------
_DEFAULT: Optional[SkillExtractor] = None


def get_default_extractor(esco_csv: str = "skills_en.csv") -> SkillExtractor:
    global _DEFAULT
    if _DEFAULT is None:
        import os
        path = os.environ.get("ESCO_SKILLS_CSV", esco_csv)
        if not os.path.exists(path):
            warnings.warn(f"ESCO CSV not found at {path!r}; running without the ESCO dictionary.")
            path = None
        _DEFAULT = SkillExtractor(path)
    return _DEFAULT


def extract_skills(job_text: str) -> List[str]:
    """Public Member 1 interface: str -> list[str]."""
    return get_default_extractor().extract_skills(job_text)


def to_handoff(skills: List[Skill]) -> List[dict]:
    """Rich JSON for Member 2 / Member 5 (superset of the frozen list[str] contract)."""
    return [s.to_dict() for s in skills]


if __name__ == "__main__":  # python -m src.extraction "job text..."  |  < ad.txt
    ad = " ".join(sys.argv[1:]) or sys.stdin.read()
    print(json.dumps(to_handoff(get_default_extractor().extract_detailed(ad)), indent=2, ensure_ascii=False))
