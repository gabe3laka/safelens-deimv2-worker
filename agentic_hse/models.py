from pydantic import BaseModel
from typing import Literal

class ReasoningRecord(BaseModel):
    hazard: str
    object_or_condition: str
    location_context: str
    risk_state: Literal['latent', 'active']
    likelihood: int
    severity: int
    score: int
    matrix_band: Literal['low', 'medium', 'high', 'critical']
    reasoning: str
    standard_reference: str
    requires_human_approval: bool = False
