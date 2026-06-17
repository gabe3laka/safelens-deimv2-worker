"""
agentic_cpu/tools/audit_tools.py -- build audit-finding structures (draft only).
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_finding(hazard: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """One DRAFT audit finding from a hazard + profile. No clause is asserted as
    fact -- unknown references are marked 'to verify'."""
    ht = str(hazard.get("hazard_type", "unknown"))
    level = str(hazard.get("risk_level", "GREEN"))
    grade = {"RED": "major", "ORANGE": "major", "YELLOW": "minor"}.get(level, "observation")
    frameworks = company_profile.get("regulatory_frameworks") or []
    return {
        "title": f"Observed hazard: {ht.replace('_', ' ')}",
        "description": (f"A '{ht}' condition ({level}) was observed by the vision "
                        f"system and should be assessed by a competent person."),
        "nonconformity_grade": grade,
        "standard_reference": (f"{frameworks[0]} (clause to verify)" if frameworks
                               else "applicable HSE standard (to verify)"),
        "objective_evidence": (f"Vision detection: {ht}, risk_level={level}, "
                               f"score={hazard.get('risk_score', 1)} "
                               f"(no raw imagery stored)."),
        "recommended_direction": [c.get("action") for c in
                                  (hazard.get("recommended_controls") or [])][:3],
    }


def summarize_findings(findings: List[Dict[str, Any]]) -> str:
    majors = sum(1 for f in findings if f.get("nonconformity_grade") == "major")
    minors = sum(1 for f in findings if f.get("nonconformity_grade") == "minor")
    return f"{len(findings)} draft finding(s): {majors} major, {minors} minor."
