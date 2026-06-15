"""Optional Pydantic AI typed-output helpers.

The production reasoning path is the private RunPod service. When an operator
explicitly configures ``PYDANTIC_AI_MODEL``, this module can provide a typed
text-only fallback for development or document-derived context. Imports remain
lazy so normal worker startup does not require provider credentials.
"""
from __future__ import annotations

import json
import os
from typing import Any


async def reason_with_pydantic_ai(payload: dict[str, Any]) -> dict[str, Any] | None:
    model_name = os.getenv("PYDANTIC_AI_MODEL", "").strip()
    if not model_name:
        return None

    from pydantic_ai import Agent

    from .models import ReasoningRecord

    agent = Agent(
        model_name,
        output_type=ReasoningRecord,
        system_prompt=(
            "Act as a senior QHSE manager. Return only the typed risk record. "
            "Use relational context, the 5x5 matrix, hierarchy of controls, and "
            "require human approval for scores of 10 or more."
        ),
    )
    result = await agent.run(json.dumps(payload, ensure_ascii=True))
    return result.output.model_dump(mode="json")
