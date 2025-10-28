# catllm/roles.py
from typing import Literal, Dict, Tuple

UserRole = Literal["Association Analyst","Buyer / Feeder","Genetic Advisor","Independent Rancher"]

ROLE_GUIDANCE: Dict[UserRole, str] = {
    "Association Analyst": "Prioritize data completeness, evaluation pipeline, and reporting standards.",
    "Buyer / Feeder": "Focus on lot value, terminal traits, uniformity, and risk mitigation.",
    "Genetic Advisor": "Provide breeding plan trade-offs, selection indexes, and evidence strength.",
    "Independent Rancher": "Keep language simple, next steps practical, and costs clear.",
}

_ALLOWED: Tuple[UserRole, ...] = (
    "Association Analyst",
    "Buyer / Feeder",
    "Genetic Advisor",
    "Independent Rancher",
)

def normalize_role(role: str) -> UserRole:
    # Accept only explicit roles; do NOT infer.
    return role if role in _ALLOWED else "Independent Rancher"

def apply_role_style(role: UserRole, answer: str) -> str:
    guidance = ROLE_GUIDANCE.get(role, "")
    if not guidance:
        return answer
    return f"**Role: {role}**\n\n_{guidance}_\n\n{answer}"