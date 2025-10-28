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