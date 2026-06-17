"""
agentic_cpu/tools/document_tools.py -- light document parsing/rendering.

Naive, dependency-free company-profile extraction from free text + a markdown
renderer. In a real deployment an LLM (agentic_cpu.llm) refines these; the
deterministic fallback keeps tests weight/key-free.
"""

from __future__ import annotations

from typing import Any, Dict, List

_INDUSTRY_HINTS = {
    "cafe": "hospitality", "restaurant": "hospitality", "kitchen": "hospitality",
    "warehouse": "logistics", "logistics": "logistics", "factory": "manufacturing",
    "manufacturing": "manufacturing", "construction": "construction",
    "office": "professional_services", "hospital": "healthcare", "clinic": "healthcare",
    "lab": "research", "school": "education", "classroom": "education",
}
_FRAMEWORK_HINTS = {
    "osha": "OSHA", "iso 45001": "ISO 45001", "iso45001": "ISO 45001",
    "hse": "UK HSE", "coshh": "COSHH", "ohsas": "OHSAS 18001",
}


def parse_company_profile_text(text: str, fields: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Best-effort structured profile from free text + explicit field overrides."""
    fields = fields or {}
    low = (text or "").lower()
    industry = None
    site_type = None
    for hint, ind in _INDUSTRY_HINTS.items():
        if hint in low:
            industry = industry or ind
            site_type = site_type or hint
    frameworks: List[str] = []
    for hint, fw in _FRAMEWORK_HINTS.items():
        if hint in low and fw not in frameworks:
            frameworks.append(fw)
    profile = {
        "company_name": fields.get("company_name"),
        "industry": fields.get("industry") or industry,
        "site_type": fields.get("site_type") or site_type,
        "country": fields.get("country"),
        "regulatory_frameworks": fields.get("regulatory_frameworks") or frameworks,
        "primary_hazards": fields.get("primary_hazards") or [],
        "headcount": fields.get("headcount"),
        "notes": fields.get("notes"),
        "source": "agentic_cpu",
        "extracted_from": "uploaded_profile_text" if text else "fields",
    }
    return profile


def render_markdown(title: str, sections: List[Dict[str, str]]) -> str:
    """Render a simple markdown document (used in previews)."""
    lines = [f"# {title}", ""]
    for sec in sections or []:
        lines.append(f"## {sec.get('heading', '')}")
        lines.append(sec.get("body", ""))
        lines.append("")
    return "\n".join(lines).strip()
