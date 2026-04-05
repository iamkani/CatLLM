# catllm/tagging.py
from __future__ import annotations

import re
from typing import Dict, List, Set


# -----------------------------
# Domain dictionaries
# -----------------------------
BREEDS = [
    "angus", "red angus", "hereford", "charolais", "simmental", "limousin", "gelbvieh",
    "brahman", "brangus", "beefmaster", "shorthorn", "wagyu", "maine-anjou", "salers",
    "chianina", "belgian blue", "senepol", "santa gertrudis",
]

TRAITS = [
    "birth weight", "bw", "weaning weight", "ww", "yearling weight", "yw", "calving ease", "ce",
    "milk", "maternal milk", "maternal weaning weight", "mww", "marbling", "imf",
    "ribeye area", "rea", "fat thickness", "ft", "carcass weight", "cw",
    "scrotal circumference", "docility", "heifer pregnancy", "hp", "stayability", "stay",
    "feed efficiency", "residual feed intake", "rfi",
]

GENES = [
    "mstn", "myostatin", "dgat1", "k232a", "leptin", "lep", "cast", "capn1", "prlr",
    "ghr", "mc1r", "pmel", "kit", "abcg2",
]

# Synonyms used for query expansion
SYNONYMS: Dict[str, List[str]] = {
    "ebv": ["epd", "estimated breeding value"],
    "epd": ["ebv", "expected progeny difference"],
    "mstn": ["myostatin"],
    "dgat1": ["k232a"],
    "marbling": ["intramuscular fat", "imf"],
    "calving": ["calving ease", "ce"],
    "heritability": ["h2", "h²"],
    "snps": ["snp", "single nucleotide polymorphism"],
    "qtl": ["quantitative trait locus"],
    "feed efficiency": ["rfi", "residual feed intake"],
}

# Compile regexes
BREED_RE = re.compile(r"\b(" + "|".join([re.escape(x) for x in BREEDS]) + r")\b", re.I)
TRAIT_RE = re.compile(r"\b(" + "|".join([re.escape(x) for x in TRAITS]) + r")\b", re.I)
GENE_RE  = re.compile(r"\b(" + "|".join([re.escape(x) for x in GENES])  + r")\b", re.I)


# -----------------------------
# Tagging
# -----------------------------
def tag_text_for_meta(text: str) -> Dict[str, List[str]]:
    """
    Extract breed/trait/gene tags from a text chunk. Lowercase, dedup, sorted.
    """
    breeds = sorted({m.group(0).lower() for m in BREED_RE.finditer(text or "")})
    traits = sorted({m.group(0).lower() for m in TRAIT_RE.finditer(text or "")})
    genes  = sorted({m.group(0).lower() for m in GENE_RE.finditer(text or "")})
    return {"breeds": breeds, "traits": traits, "genes": genes}


def merge_meta(base: Dict, extra: Dict) -> Dict:
    """
    Merge tagging meta into existing meta dict while keeping unique lists.
    """
    out = dict(base or {})
    for key in ("breeds", "traits", "genes"):
        vals = set(out.get(key, []))
        vals.update(extra.get(key, []))
        out[key] = sorted(vals)
    return out


# -----------------------------
# Query expansion
# -----------------------------
def expand_query(q: str) -> str:
    """
    Expand a user query with domain synonyms, preserving the original query first.
    - Only adds each synonym once.
    - Matches both single-word and phrase keys.
    """
    if not q:
        return q

    q_low = q.lower()
    tokens = {t.strip(".,;:!?()[]{}\"'") for t in q_low.split() if t.strip()}
    extra_terms: List[str] = []

    # token-based
    for tok in list(tokens):
        if tok in SYNONYMS:
            extra_terms.extend(SYNONYMS[tok])

    # phrase-based
    for key, syns in SYNONYMS.items():
        if " " in key and key in q_low:
            extra_terms.extend(syns)

    # unique, preserve order
    seen: Set[str] = set()
    uniq = []
    for t in extra_terms:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    if not uniq:
        return q
    return q + " " + " ".join(uniq)


# -----------------------------
# Role-aware instruction helper
# -----------------------------
ROLE_INSTRUCTIONS = {
    "Association Analyst": (
        "Emphasize data completeness, registry compliance, pedigree integrity, and contemporary groups. "
        "Propose concrete audit/report checks for missing or invalid fields."
    ),
    "Buyer / Feeder": (
        "Prioritize profitability traits (growth, feed efficiency, carcass merit, docility) and risk notes. "
        "Give quick takeaways and validation steps where uncertain."
    ),
    "Genetic Advisor": (
        "Start from breeding objective (maternal/terminal/balanced). "
        "Translate EPDs/genomics into selection decisions and mating systems; suggest scenarios and trade-offs."
    ),
    "Independent Rancher": (
        "Use plain language and checklists. Relate traits to ranch outcomes and offer a short 'Do this next' list."
    ),
}


def role_hint(role: str) -> str:
    """Return concise, role-specific guidance string (safe default if unknown)."""
    return ROLE_INSTRUCTIONS.get(role.strip(), ROLE_INSTRUCTIONS["Independent Rancher"])


# -----------------------------
# Cluster name normalization
# -----------------------------
_CLUSTER_STRIP_RE = re.compile(r"[^a-z0-9 ]+")


def _cluster_key(raw: str) -> str:
    """Reduce a cluster name to a canonical comparison key."""
    s = raw.lower().replace("&", "and").replace("-", " ")
    s = _CLUSTER_STRIP_RE.sub(" ", s)
    return " ".join(s.split())


# Canonical names (data/Excel style) keyed by their stripped form.
# Built from the 20 clusters in the training data.
_CANONICAL_CLUSTERS: Dict[str, str] = {}

_RAW_CANONICAL = [
    "Batch Processing, Auditing, and Reporting",
    "Batch Uploads & Report Integration",
    "Breed-Wide Trend Analysis",
    "Bull Selection & Breeding Strategy",
    "Carcass Value & Terminal Trait Evaluation",
    "Client Education & Communication",
    "Cross-Breed Comparisons & Risk Assessment",
    "Data Input, Upload, and Tools",
    "Getting Started with Genetics",
    "Lot Evaluation & GMR Understanding",
    "Multi-Herd Management & Reporting",
    "Price Forecasting & Strategic Buying",
    "Research & Strategic Planning",
    "Standardization & Index Monitoring",
    "Testing & Scenario Planning",
    "Tools, Integrations, and Data Flows",
    "Tools, Visualization, and Exports",
    "Trait Evaluation & Herd Analytics",
    "Trait Prioritization & Custom Advice",
    "Understanding Genetic Reports",
]

for _c in _RAW_CANONICAL:
    _CANONICAL_CLUSTERS[_cluster_key(_c)] = _c

# Aliases for known folder-name variations (plural/singular mismatches, typos)
_CLUSTER_ALIASES = {
    "data input uploads and tools": "Data Input, Upload, and Tools",
    "tools integration and data flows": "Tools, Integrations, and Data Flows",
}
_CANONICAL_CLUSTERS.update(_CLUSTER_ALIASES)


def normalize_cluster_name(raw: str) -> str:
    """Map a raw cluster name (folder name or data value) to its canonical form.

    Handles casing, ``&`` vs ``and``, typos like 'eveluation', 'statrted', 'ans',
    and missing/extra punctuation.
    """
    if not raw or not raw.strip():
        return raw or ""
    key = _cluster_key(raw)
    # Fix known typos before lookup
    key = (
        key.replace("eveluation", "evaluation")
           .replace("statrted", "started")
           .replace("visualisation", "visualization")
           .replace(" ans ", " and ")
    )
    if key in _CANONICAL_CLUSTERS:
        return _CANONICAL_CLUSTERS[key]
    # Fuzzy fallback: find best substring match
    for ck, cv in _CANONICAL_CLUSTERS.items():
        if key in ck or ck in key:
            return cv
    # No match — return cleaned-up version
    return raw.strip()