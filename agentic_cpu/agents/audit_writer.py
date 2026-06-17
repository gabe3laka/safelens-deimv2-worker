"""
agentic_cpu/agents/audit_writer.py -- draft internal HSE audit findings.
Sending findings is approval-required (audit_finding_send).
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import audit_tools, vision_tools
from . import new_action


def draft(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    profile = req.get("company_profile") or {}
    hazards = vision_tools.extract_hazards(dc)
    findings = [audit_tools.build_finding(h, profile) for h in hazards]
    payload = {
        "audit_type": "internal_hse_audit",
        "findings": findings,
        "summary": audit_tools.summarize_findings(findings),
        "notes": req.get("notes"),
        "disclaimer": "AI-drafted findings; a competent auditor must approve before sending.",
    }
    return new_action(
        "audit_finding_send", req,
        title="Audit findings draft",
        summary=payload["summary"],
        payload=payload,
        preview_extra={"finding_count": len(findings)},
    )
