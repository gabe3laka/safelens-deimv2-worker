from typing import Any, TypedDict

class AgenticHSEState(TypedDict, total=False):
    detections: list[dict[str, Any]]
    company_profile: dict[str, Any]
    zone_context: dict[str, Any]
    reasoning: dict[str, Any]
    approvals: list[dict[str, Any]]
