"""
agentic_cpu/agents/company_setup.py -- extract a structured company profile from
free text / explicit fields. Informational (not an approval-required action):
returns a CompanyProfile, not an AgentAction.
"""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import CompanyProfile
from ..tools import document_tools


def extract_profile(req: Dict[str, Any]) -> Dict[str, Any]:
    text = (req or {}).get("text") or ""
    fields = (req or {}).get("fields") or {}
    profile = document_tools.parse_company_profile_text(text, fields)
    # Validate/normalise through the shared schema.
    return CompanyProfile(**profile).model_dump()
