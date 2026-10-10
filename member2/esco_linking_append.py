
# ===================== TEAM CONTRACT (appended for integration) =====================
# map_to_esco(extracted_skills, job_text) -> list of handoff-ready rows
# Same rules as main(): only EXACT label/alias matches are returned; semantic matches go to
# LAST_REVIEW (or are returned too when include_likely=True). Nothing above this line was changed.
import os as _os

_STATE = {}
LAST_REVIEW = []


def _find_file(name):
    root = Path(_os.environ.get("REPO_ROOT", Path(__file__).resolve().parent))
    hits = [p for p in root.rglob(name) if ".git" not in p.parts and "olderversions" not in p.parts]
    return hits[0] if hits else None


def _setup():
    """Load ESCO, the model and the embeddings once (works on Kaggle: no D:\\ paths needed)."""
    global ESCO_FILE, CACHE_FILE
    if _STATE:
        return _STATE
    if not ESCO_FILE.exists():
        ESCO_FILE = _find_file("skills_en.csv")
        if ESCO_FILE is None:
            raise FileNotFoundError("skills_en.csv not found in the repo")
    if not CACHE_FILE.exists():
        CACHE_FILE = _find_file("esco_label_embeddings_cache.npz") or (ESCO_FILE.parent / CACHE_FILE.name)
    concepts = load_esco()
    model = SentenceTransformer(MODEL_NAME)
    _STATE.update(concepts=concepts, model=model, vectors=get_embeddings(model, concepts))
    return _STATE


def _decide(skill, qvec, concepts, vectors):
    exact = exact_candidates(skill, concepts)
    semantic = semantic_candidates(skill, qvec, vectors, concepts)
    if exact:
        safe = [(i, m) for i, m in exact
                if not (m == "EXACT_ALIAS"
                        and (normalize(skill), normalize(concepts[i]["label"])) in BLOCKED_ALIAS_PAIRS)]
        if len(safe) == 1:
            i, m = safe[0]
            c = concepts[i]
            return m, {"uri": c["uri"], "label": c["label"], "score": None, "method": m}, semantic
        return ("AMBIGUOUS_EXACT_ALIAS" if safe else "REVIEW_REQUIRED"), None, semantic
    if semantic:
        top = semantic[0]
        second = semantic[1]["score"] if len(semantic) > 1 else -1.0
        if top["score"] >= MIN_SCORE and top["score"] - second >= MIN_MARGIN:
            return ("LIKELY_MATCH_REQUIRES_CONFIRMATION",
                    {"uri": top["uri"], "label": top["label"], "score": top["score"], "method": "SEMANTIC"},
                    semantic)
    return "REVIEW_REQUIRED", None, semantic


def map_to_esco(extracted_skills, job_text=None, include_likely=False):
    """extracted_skills: list of str, or Member 1's dicts (skill / importance_hint / requirement)."""
    st = _setup()
    LAST_REVIEW.clear()
    items, seen = [], set()
    for it in extracted_skills:
        meta = it if isinstance(it, dict) else {}
        text = clean(it if isinstance(it, str) else (
            it.get("text") or it.get("extracted_skill") or it.get("skill") or it.get("preferredLabel")))
        if text and normalize(text) not in seen:
            seen.add(normalize(text))
            items.append((text, get_importance(meta)[0]))
    if not items:
        return []
    qv = st["model"].encode([t for t, _ in items], batch_size=32, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    rows = []
    for (text, imp), v in zip(items, qv):
        status, sel, sem = _decide(text, v, st["concepts"], st["vectors"])
        base = {"phrase": text, "importance": imp, "status": status}
        if sel and (sel["method"] != "SEMANTIC" or include_likely):
            rows.append({**base, "esco_id": sel["uri"], "label": sel["label"], "match_method": sel["method"],
                         "confidence": None if sel["score"] is None else round(sel["score"], 4)})
        else:
            LAST_REVIEW.append({**base, "suggested": sel["label"] if sel else None,
                                "candidates": [{"label": c["label"], "esco_id": c["uri"],
                                                "similarity_score": round(c["score"], 4)} for c in sem[:TOP_K]]})
    return rows
