"""
shared/schemas/company_profile.py -- structured company/site profile used to
ground agent drafts (risk assessments, audits, training). Free of any PII beyond
what the operator supplies; never persisted with raw frames.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CompanyProfile(BaseModel):
    company_name: Optional[str] = None
    industry: Optional[str] = None
    site_type: Optional[str] = None            # cafe | warehouse | construction | office | ...
    country: Optional[str] = None
    regulatory_frameworks: List[str] = Field(default_factory=list)  # e.g. ["OSHA", "ISO 45001"]
    primary_hazards: List[str] = Field(default_factory=list)
    headcount: Optional[int] = None
    notes: Optional[str] = None
    source: str = "agentic_cpu"
    extracted_from: Optional[str] = None       # e.g. "uploaded_profile_text"
